"""Lower a PyMC logp graph onto the tape, and check the module against PyMC.

Walks the PyTensor graph of `model.logp()` and writes the instruction file
`crates/tapewasm-codegen/examples/tape_from_text.rs` replays. Everything
upstream of the value variables is evaluated once and carried as constants;
everything downstream becomes tape nodes, one per scalar element.

Each model is lowered at one point and evaluated at another, so a subgraph
wrongly folded into a constant shows up as a mismatch.

    uv run --with pymc --with scipy python tools/lower_pytensor.py

A spike, not a backend: it covers elementwise ops, sums, indexing, contraction
and the bounds checks the transforms make vacuous. A `Scan` is unrolled over its
fixed step count. A `Switch` on an ordering of parameters becomes tapewasm's `pick`
(0.3.5); one on a bounds check or an equality folds at the trace point.
"""

import functools
import os
import subprocess
import sys

from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.rewriting.utils import rewrite_graph
try:
    from pytensor.graph.traversal import ancestors, io_toposort
except ImportError:
    from pytensor.graph.basic import ancestors, io_toposort


# Every instruction's value, so a lowered subgraph can be compared against what
# PyTensor computes for the same variable. Only used by `--verify`.
def _apply(op, args, vals):
    from scipy.special import gammaln
    from scipy.stats import norm

    # float64 under errstate, so a trace point where log(-1) or 1/0 happens gets the nan or
    # inf the module and PyTensor give there, rather than an exception.
    a = lambda k: np.float64(vals[int(args[k])])
    c = lambda k: np.float64(args[k])
    with np.errstate(all="ignore"):
        return float(_OPS(a, c, args, gammaln, norm)[op]())


def _OPS(a, c, args, gammaln, norm):
    return {
        "new_var": lambda: c(0),
        "add": lambda: a(0) + a(1), "sub": lambda: a(0) - a(1),
        "mul": lambda: a(0) * a(1), "div": lambda: a(0) / a(1),
        "neg": lambda: -a(0), "exp": lambda: np.exp(a(0)),
        "log": lambda: np.log(a(0)), "sin": lambda: np.sin(a(0)),
        "cos": lambda: np.cos(a(0)), "sqrt": lambda: np.sqrt(a(0)),
        "abs": lambda: abs(a(0)), "lgamma": lambda: float(gammaln(a(0))),
        "phi": lambda: float(norm.cdf(a(0))), "pow": lambda: np.power(a(0), c(1)),
        "add_c": lambda: a(0) + c(1), "sub_c": lambda: a(0) - c(1),
        "rsub_c": lambda: c(1) - a(0), "mul_c": lambda: a(0) * c(1),
        "div_c": lambda: a(0) / c(1), "rdiv_c": lambda: c(1) / a(0),
        # `dot_c <len> <node> <coeff> ...` and `sum_run <seed> <len> <node> ...`.
        "dot_c": lambda: sum(a(1 + 2 * i) * c(2 + 2 * i) for i in range(int(args[0]))),
        "sum_run": lambda: a(0) + sum(a(2 + i) for i in range(int(args[1]))),
        "gt": lambda: float(a(0) > a(1)), "ge": lambda: float(a(0) >= a(1)),
        "lt": lambda: float(a(0) < a(1)), "le": lambda: float(a(0) <= a(1)),
        "eq": lambda: float(a(0) == a(1)), "ne": lambda: float(a(0) != a(1)),
        "pick": lambda: a(1) if a(0) != 0 else 0.0,
    }


def _run(nodes):
    """The tape indices as a strictly increasing, evenly spaced run, or None.

    Both run instructions want that shape; value numbering can merge two of the
    elements into one node, which leaves a gap the tape text refuses.
    """
    xs = [int(n) for n in nodes]
    if len(xs) < 2:
        return None
    stride = xs[1] - xs[0]
    if stride <= 0:
        return None
    return stride if all(b - a == stride for a, b in zip(xs, xs[1:])) else None


class TapeWriter:
    def __init__(self):
        self.lines = []
        self.values = []
        self.n = 0

    def emit(self, *parts):
        self.lines.append(" ".join(str(p) for p in parts))
        self.values.append(_apply(str(parts[0]), parts[1:], self.values))
        self.n += 1
        return self.n - 1

    def const_node(self, v):
        """A constant that has to reach the tape as a node.

        A leaf recorded before the first non-leaf op would be taken for a parameter,
        so a graph that needs a constant first gets an op ahead of it.
        """
        if all(l.startswith("new_var") for l in self.lines):
            self.emit("mul_c", 0, "0.0")
        return self.emit("new_var", repr(float(v)))


# value: either ("c", np.ndarray of floats) or ("t", np.ndarray of node ids)
def is_tape(x):
    return x[0] == "t"


def op_name(op):
    return type(op).__name__


def static_index(op, inputs, resolve):
    """The index a Subtensor-family op means, slices and all.

    `idx_list` holds the shape of the index and the inputs hold its numbers, and
    those numbers are symbolic — the same variables everything else goes through.
    """
    from pytensor.tensor.subtensor import get_idx_list

    def scalar(v):
        if v is None:
            return None
        if isinstance(v, (int, np.integer)):
            return int(v)
        return int(np.asarray(resolve(v)))

    out = []
    for i in get_idx_list(inputs, op.idx_list):
        if isinstance(i, slice):
            out.append(slice(scalar(i.start), scalar(i.stop), scalar(i.step)))
        else:
            out.append(scalar(i))
    return tuple(out)


BINARY = {
    "Add": ("add", "add_c", "add_c", np.add),
    "Sub": ("sub", "sub_c", "rsub_c", np.subtract),
    "Mul": ("mul", "mul_c", "mul_c", np.multiply),
    "TrueDiv": ("div", "div_c", "rdiv_c", np.divide),
}

UNARY = {
    "Exp": ("exp", np.exp),
    "Log": ("log", np.log),
    "Sqrt": ("sqrt", np.sqrt),
    "Abs": ("abs", np.abs),
    "Neg": ("neg", np.negative),
    "Sin": ("sin", np.sin),
    "Cos": ("cos", np.cos),
    "GammaLn": ("lgamma", None),
}

COMPARISON = {
    "GE": np.greater_equal, "GT": np.greater, "LE": np.less_equal,
    "LT": np.less, "EQ": np.equal, "NEQ": np.not_equal,
    "AND": np.logical_and, "OR": np.logical_or, "Invert": np.logical_not,
    "IsNan": np.isnan, "IsInf": np.isinf,
}

UNARY_TEST = {"Invert", "IsNan", "IsInf"}

# PyTensor's comparisons as tapewasm's (0.3.5): 1 where they hold, else 0, no gradient.
TAPE_COMPARISON = {"GT": "gt", "GE": "ge", "LT": "lt", "LE": "le", "EQ": "eq", "NEQ": "ne"}


def compare(name, vals):
    if len(vals) == 1:
        r = COMPARISON[name](vals[0]) if name in UNARY_TEST else vals[0]
    else:
        r = vals[0]
        for v in vals[1:]:
            r = COMPARISON[name](r, v)
    return np.asarray(r).astype(float)

# Comparisons a tape value reached, folded at the trace point. Vacuous for a bounds
# check after the transforms, or for a test of a measure-zero event.
FOLDED = []

PASSTHROUGH = {
    "CheckParameterValue", "ScalarFromTensor", "TensorFromScalar", "Identity",
    "SpecifyShape", "Cast", "ViewOp", "DeepCopyOp", "Unbroadcast",
    "TensorFromXTensor", "XTensorFromTensor",
}


class Lowerer:
    def __init__(self, w, guards=None, branch_on_tape=False):
        self.w = w
        self.guards = guards
        # Whether a Switch on an ordering of parameters becomes tapewasm's `pick`
        # (0.3.5) rather than folding: a module compiled once cannot trace again.
        self.branch_on_tape = branch_on_tape
        # A comparison that reached the tape folds, but keeps how to build it, by the
        # identity of its folded array, for a Switch that has to branch on it for real.
        self.recipes = {}

    def binary(self, name, a, b):
        node_op, c_right, c_left, npf = BINARY[name]
        if not is_tape(a) and not is_tape(b):
            return ("c", npf(a[1], b[1]))
        if is_tape(a) and is_tape(b):
            ia, ib = np.broadcast_arrays(a[1], b[1])
            out = np.empty(ia.shape, dtype=object)
            for k in np.ndindex(ia.shape):
                out[k] = self.w.emit(node_op, ia[k], ib[k])
            return ("t", out)
        # One side is constant: use the fused constant form.
        if is_tape(a):
            it, cs, form = a[1], b[1], c_right
        else:
            it, cs, form = b[1], a[1], c_left
        it, cs = np.broadcast_arrays(it, np.asarray(cs, dtype=float))
        out = np.empty(it.shape, dtype=object)
        for k in np.ndindex(it.shape):
            out[k] = self.w.emit(form, it[k], repr(float(cs[k])))
        return ("t", out)

    def unary(self, name, a):
        node_op, npf = UNARY[name]
        if not is_tape(a):
            if npf is None:
                from scipy.special import gammaln

                return ("c", gammaln(a[1]))
            return ("c", npf(a[1]))
        idx = np.asarray(a[1])
        out = np.empty(idx.shape, dtype=object)
        for k in np.ndindex(idx.shape):
            out[k] = self.w.emit(node_op, idx[k])
        return ("t", out)

    def scalar_op(self, name, ins):
        if name in BINARY:  # Add and Mul are variadic in PyTensor
            if len(ins) == 1:
                return ins[0]
            acc = ins[0]
            for nxt in ins[1:]:
                acc = self.binary(name, acc, nxt)
            return acc
        if name in UNARY and len(ins) == 1:
            return self.unary(name, ins[0])
        # Compositions of ops the tape has.
        if name == "Sqr":
            return self.binary("Mul", ins[0], ins[0])
        if name == "Reciprocal":
            return self.binary("TrueDiv", ("c", np.array(1.0)), ins[0])
        if name == "Log1p":
            return self.unary("Log", self.binary("Add", ins[0], ("c", np.array(1.0))))
        if name == "Sigmoid":
            return self.unary("Exp", self.unary("Neg", self.scalar_op("Softplus", [self.unary("Neg", ins[0])])))
        if name == "Pow":
            if not is_tape(ins[1]):
                base = ins[0]
                if not is_tape(base):
                    return ("c", np.power(base[1], ins[1][1]))
                it, cs = np.broadcast_arrays(base[1], np.asarray(ins[1][1], dtype=float))
                out = np.empty(it.shape, dtype=object)
                for k in np.ndindex(it.shape):
                    out[k] = self.w.emit("pow", it[k], repr(float(cs[k])))
                return ("t", out)
        if name == "Switch":
            return self.select(ins[0], ins[1], ins[2])
        if name in ("Maximum", "Minimum"):
            return self.select(self.scalar_op("GE" if name == "Maximum" else "LE", ins), ins[0], ins[1])
        if name == "Clip":  # a covariance's squared distance, clipped at 0
            return self.scalar_op("Minimum", [self.scalar_op("Maximum", ins[:2]), ins[2]])
        if name in COMPARISON:
            # Folded to what it is at the trace point: in a logp a bounds check, constant
            # after the transforms (`value < 0` guards HalfFlat). A guard list records it,
            # for a caller that has to check the branch still holds at another point.
            vals = [np.vectorize(lambda i: self.w.values[int(i)], otypes=[float])(x[1])
                    if is_tape(x) else np.asarray(x[1]) for x in ins]
            r = compare(name, vals)
            if any(is_tape(x) or id(np.asarray(x[1])) in self.recipes for x in ins):
                FOLDED.append(name)
                if self.guards is not None:
                    self.guards.append((name, ins, r))
                self.recipes[id(r)] = (r, name, ins)
            return ("c", r)
        if name == "Second":
            return ins[1]
        if name in ("Identity", "Cast", "ScalarIdentity"):
            return ins[0]
        if name == "Sign":
            if not is_tape(ins[0]):
                return ("c", np.sign(ins[0][1]))
            return self.binary("TrueDiv", ins[0], self.unary("Abs", ins[0]))
        if name == "Softplus":
            # max(x, 0) + log(1 + exp(-|x|)): the plain form is inf past x = 709.
            x, ax = ins[0], self.unary("Abs", ins[0])
            relu = self.binary("Mul", self.binary("Add", x, ax), ("c", np.array(0.5)))
            tail = self.unary("Exp", self.unary("Neg", ax))
            return self.binary("Add", relu, self.unary("Log", self.binary("Add", tail, ("c", np.array(1.0)))))
        if name == "Log1mexp":
            inner = self.unary("Exp", ins[0])
            return self.unary("Log", self.binary("Sub", ("c", np.array(1.0)), inner))
        if name == "Erf":
            raise NotImplementedError("Erf has no emitter arm")
        raise NotImplementedError(f"scalar op {name}")

    def select(self, cond, a, b):
        # A bounds check — one side a constant infinity — holds everywhere the
        # transforms reach, so it folds at the trace point as it always has.
        # So does a test for equality: a parameter meets it on a set of measure zero.
        bounds = self.bounds(a, b)
        if self.branch_on_tape and not bounds and (is_tape(cond) or self.ordering(cond)):
            c = (self.compare_on_tape("NEQ", cond, ("c", np.array(0.0))) if is_tape(cond)
                 else self.materialize(cond))
            return self.branch(c, a, b)
        if is_tape(cond):
            raise NotImplementedError(
                "Switch on a parameter: a branch resolved while tracing would "
                "bake in the trace point"
            )
        c, ca, cb = np.broadcast_arrays(
            np.asarray(cond[1]), np.asarray(a[1]), np.asarray(b[1])
        )
        if not is_tape(a) and not is_tape(b):
            return ("c", np.where(c != 0, ca, cb))
        out = np.empty(c.shape, dtype=object)
        for k in np.ndindex(c.shape):
            taken, src = (ca[k], a) if c[k] != 0 else (cb[k], b)
            out[k] = taken if is_tape(src) else self.w.const_node(taken)
        return ("t", out)

    @staticmethod
    def bounds(a, b):
        """One side wholly a constant infinity, as a bounds check's `-inf` is."""
        return any(not is_tape(x) and np.isinf(np.asarray(x[1], dtype=float)).all() for x in (a, b))

    def check_condition(self, depends, cond, a, b):
        """Refuse a condition that depends on a parameter but reached the Switch folded
        with no record of the comparison (it went through a stack or an `all`): folding
        it would bake in the trace point's branch."""
        if (self.branch_on_tape and depends and not is_tape(cond)
                and id(cond[1]) not in self.recipes and not self.bounds(a, b)):
            raise NotImplementedError(
                "a Switch whose condition depends on a parameter through an op that keeps "
                "no record of the comparison")

    def ordering(self, value):
        """Whether a folded condition compares parameters by order, anywhere in it."""
        if is_tape(value) or id(value[1]) not in self.recipes:
            return False
        _, name, ins = self.recipes[id(value[1])]
        return name in ("GT", "GE", "LT", "LE") or any(self.ordering(x) for x in ins)

    def materialize(self, value):
        """The tape node(s) a folded comparison stood for; anything else unchanged."""
        if is_tape(value) or id(value[1]) not in self.recipes:
            return value
        _, name, ins = self.recipes[id(value[1])]
        ins = [self.materialize(x) for x in ins]
        if isinstance(name, tuple):  # a reshape of one: (rule, op, node, cx)
            rule, op, node, cx = name
            return rule(op, node, ins, cx)
        if name in TAPE_COMPARISON:
            return self.compare_on_tape(name, *ins)
        if name == "AND":
            return functools.reduce(lambda x, y: self.binary("Mul", x, y), ins)
        if name == "OR":  # both 0 or 1
            return functools.reduce(
                lambda x, y: self.binary("Sub", self.binary("Add", x, y), self.binary("Mul", x, y)), ins)
        if name == "Invert":
            return self.binary("Sub", ("c", np.array(1.0)), ins[0])
        raise NotImplementedError(f"{name} of a parameter, which the tape cannot test")

    def compare_on_tape(self, name, a, b):
        if not is_tape(a) and not is_tape(b):
            return ("c", compare(name, [np.asarray(a[1]), np.asarray(b[1])]))
        ia, ib = np.broadcast_arrays(np.asarray(a[1]), np.asarray(b[1]))
        out = np.empty(ia.shape, dtype=object)
        for k in np.ndindex(ia.shape):
            x = ia[k] if is_tape(a) else self.w.const_node(float(ia[k]))
            y = ib[k] if is_tape(b) else self.w.const_node(float(ib[k]))
            out[k] = self.w.emit(TAPE_COMPARISON[name], x, y)
        return ("t", out)

    def branch(self, c, a, b):
        """`a` where `c` is 1 and `b` where it is 0: two picks added, not a product, so
        the NaN of the side not taken stays out of the value. Its gradient is right where
        that side's partials are finite; PyTensor's switch rewrites also cover infinite ones."""
        if not is_tape(c):
            return self.select(c, a, b)
        return self.binary("Add", self.pick(c, a), self.pick(self.binary("Sub", ("c", np.array(1.0)), c), b))

    def pick(self, c, v):
        ic, iv = np.broadcast_arrays(np.asarray(c[1]), np.asarray(v[1]))
        if not is_tape(v) and not np.any(iv != 0):
            return ("c", np.zeros(ic.shape))
        out = np.empty(ic.shape, dtype=object)
        for k in np.ndindex(ic.shape):
            node = iv[k] if is_tape(v) else self.w.const_node(float(iv[k]))
            out[k] = self.w.emit("pick", ic[k], node)
        return ("t", out)

    def contract(self, coeffs, run):
        """`dot_c` for one output element, or None if the run will not serve.

        The contraction node names a run of tape values and a coefficient each,
        so the products never reach the tape. What it needs is that the tape
        side is evenly spaced — value numbering can merge two of its elements,
        and then it is not.
        """
        if len(run) < 2 or _run(run) is None:
            return None
        pairs = []
        for node, c in zip(run, coeffs):
            pairs.extend((node, float(c)))
        return self.w.emit("dot_c", len(run), *pairs)

    def dot(self, a, b):
        """A contraction node where one side is data, else products and a sum.

        `X * beta` — data on the left, parameters on the right — is one node per
        row rather than the `2K` a chain of multiplies and adds would record.
        """
        A, B = np.asarray(a[1]), np.asarray(b[1])
        # Data on one side and an evenly spaced run of tape values on the other.
        # PyTensor hands `X @ beta` over with beta as a column, so a trailing 1
        # is the same shape as a vector and the result keeps it.
        vec_b = B.ndim == 1 or (B.ndim == 2 and B.shape[1] == 1)
        if A.ndim == 2 and vec_b and not is_tape(a) and is_tape(b):
            run = list(B.ravel())
            if len(run) == A.shape[1]:
                rows = [self.contract(A[i], run) for i in range(A.shape[0])]
                if all(r is not None for r in rows):
                    out = np.empty(A.shape[0] if B.ndim == 1 else (A.shape[0], 1), dtype=object)
                    out.ravel()[:] = rows
                    return ("t", out)
        if A.ndim == 1 and B.ndim == 1:
            return self.reduce_sum(self.binary("Mul", a, b), None)
        if A.ndim == 2 and B.ndim == 1:
            return self.reduce_sum(self.binary("Mul", a, b), (1,))
        if A.ndim == 1 and B.ndim == 2:
            return self.reduce_sum(self.binary("Mul", (a[0], A[:, None]), b), (0,))
        if A.ndim == 2 and B.ndim == 2:
            wide = self.binary("Mul", (a[0], A[:, :, None]), (b[0], B[None, :, :]))
            return self.reduce_sum(wide, (1,))
        raise NotImplementedError(f"dot of {A.ndim}d and {B.ndim}d")

    def cholesky(self, a, lower=True):
        """The decomposition written out in scalars, the only form the tape has.

        `L[j][j] = sqrt(A[j][j] - sum_m L[j][m]^2)` and
        `L[i][j] = (A[i][j] - sum_m L[i][m] L[j][m]) / L[j][j]`, which is cubic
        in the dimension where the rest of a model is linear in the data.
        `Blockwise` hands over a batch, so the leading axes are mapped.
        """
        A = np.asarray(a[1])
        tape_in = is_tape(a)
        batch = A.shape[:-2]
        out = np.empty(A.shape, dtype=object if tape_in else float)
        for b in np.ndindex(batch):
            out[b] = self._cholesky_2d(A[b], tape_in, lower)
        return ("t" if tape_in else "c", out)

    def _cholesky_2d(self, A, tape_in, lower):
        k = A.shape[0]
        dt = object if tape_in else float
        L = np.zeros((k, k), dtype=dt)
        wrap = lambda v: ("t" if tape_in else "c", np.array(v, dtype=dt))
        for j in range(k):
            acc = wrap(A[j, j])
            for m in range(j):
                sq = self.binary("Mul", wrap(L[j, m]), wrap(L[j, m]))
                acc = self.binary("Sub", acc, sq)
            djj = self.unary("Sqrt", acc)
            L[j, j] = np.asarray(djj[1]).item()
            for i in range(j + 1, k):
                num = wrap(A[i, j])
                for m in range(j):
                    cross = self.binary("Mul", wrap(L[i, m]), wrap(L[j, m]))
                    num = self.binary("Sub", num, cross)
                L[i, j] = np.asarray(self.binary("TrueDiv", num, djj)[1]).item()
        if not lower:
            L = L.T
        if tape_in:
            # A structural zero still has to be a value the tape can name.
            for idx in np.ndindex(L.shape):
                if L[idx] == 0:
                    L[idx] = self.w.const_node(0.0)
        return L

    def solve_triangular(self, a, b, lower=True, b_ndim=1):
        """Forward or back substitution, one scalar at a time.

        `x[i] = (b[i] - sum_{m<i} L[i][m] x[m]) / L[i][i]`, and the mirror of it
        for an upper factor. `Blockwise` broadcasts the two operands' batch axes
        against each other, so one factor can serve many right-hand sides.
        """
        A, B = np.asarray(a[1]), np.asarray(b[1])
        tape = is_tape(a) or is_tape(b)
        dt = object if tape else float
        a_batch, b_batch = A.shape[:-2], B.shape[:-b_ndim]
        batch = np.broadcast_shapes(a_batch, b_batch)
        k = A.shape[-1]
        cols = B.shape[-1] if b_ndim == 2 else None
        out = np.empty(batch + ((k,) if cols is None else (k, cols)), dtype=dt)

        def fit(idx, shape):
            # A batch axis of length one is shared by every index along it.
            off = len(idx) - len(shape)
            return tuple(0 if shape[i] == 1 else idx[off + i] for i in range(len(shape)))

        wa = lambda v: ("t" if is_tape(a) else "c", np.array(v, dtype=object if is_tape(a) else float))
        wb = lambda v: ("t" if is_tape(b) else "c", np.array(v, dtype=object if is_tape(b) else float))
        wo = lambda v: ("t" if tape else "c", np.array(v, dtype=dt))
        for bi in np.ndindex(batch):
            ai, bj = fit(bi, a_batch), fit(bi, b_batch)
            order = range(k) if lower else range(k - 1, -1, -1)
            for c in ([None] if cols is None else range(cols)):
                for i in order:
                    tail = (i,) if c is None else (i, c)
                    acc = wb(B[bj + tail])
                    ms = range(i) if lower else range(i + 1, k)
                    for m in ms:
                        prev = (m,) if c is None else (m, c)
                        term = self.binary("Mul", wa(A[ai + (i, m)]), wo(out[bi + prev]))
                        acc = self.binary("Sub", acc, term)
                    div = self.binary("TrueDiv", acc, wa(A[ai + (i, i)]))
                    out[bi + tail] = np.asarray(div[1]).item()
        return ("t" if tape else "c", out)

    def convolve1d(self, x, k, full):
        """`numpy.convolve` along the last axis, one lag at a time.

        Valid mode is `out[t] = sum_j x[t + m-1 - j] k[j]`, and full mode is
        valid mode over `x` padded with `m-1` zeros each side. `Blockwise` hands
        over leading batch axes, which the elementwise products broadcast.
        """
        F = np.asarray(full[1])
        assert F.all() == F.any(), "a batch mixing full and valid convolution"
        m = np.shape(k[1])[-1]
        if F.any():
            X = np.asarray(x[1])
            pad = [(0, 0)] * (X.ndim - 1) + [(m - 1, m - 1)]
            if is_tape(x):
                x = ("t", np.pad(X, pad, constant_values=self.w.const_node(0.0)))
            else:
                x = ("c", np.pad(X, pad))
        X, K = np.asarray(x[1]), np.asarray(k[1])
        n = X.shape[-1]
        assert n >= m, "a kernel longer than the signal"
        L = n - m + 1
        acc = None
        for j in range(m):
            term = self.binary("Mul", (x[0], X[..., m - 1 - j : m - 1 - j + L]),
                               (k[0], K[..., j : j + 1]))
            acc = term if acc is None else self.binary("Add", acc, term)
        return acc

    def reduce_sum(self, a, axis):
        if not is_tape(a):
            return ("c", np.sum(a[1], axis=axis))
        idx = np.asarray(a[1])
        if axis is None:
            axis = tuple(range(idx.ndim))
        keep = tuple(d for d in range(idx.ndim) if d not in tuple(axis))
        moved = np.transpose(idx, keep + tuple(axis))
        out_shape = moved.shape[: len(keep)]
        out = np.empty(out_shape, dtype=object)
        for k in np.ndindex(out_shape):
            run = np.asarray(moved[k]).ravel()
            # One reduction node where the run allows it: the chain it replaces
            # carries a value from each element to the next, which is the one
            # shape a re-rolled loop cannot run two repeats of at a time.
            if len(run) >= 4 and _run(run[1:]) is not None:
                out[k] = self.w.emit("sum_run", run[0], len(run) - 1, *run[1:])
                continue
            acc = run[0]
            for nxt in run[1:]:
                acc = self.w.emit("add", acc, nxt)
            out[k] = acc
        return ("t", out if out_shape else np.array(out.item(), dtype=object))


# How each op becomes tape values, by op name: `f(op, node, ins, cx)` gets the
# inputs already lowered, as ("c", floats) or ("t", node ids), and returns the output.
LOWER = {}


def lowers(*names):
    def register(f):
        for n in names:
            LOWER[n] = f
        return f
    return register


def rule_for(op):
    name, inner = op_name(op), getattr(op, "scalar_op", None)
    if name == "CAReduce" and op_name(inner) == "Add":
        return "Sum"
    if name not in LOWER and inner is not None:
        return "Elemwise"
    return name


def _index(x):
    if is_tape(x):
        raise NotImplementedError("an index that depends on a parameter")
    return x[1] if isinstance(x[1], slice) or x[1] is None else np.asarray(x[1]).astype(int)


def _advanced_key(op, index_ins):
    # PyTensor 3 keeps slices in `idx_list`, whose integers are positions among the index
    # inputs (`x[1:, i]` is `(slice(0, None), 1)`); older ones pass everything as inputs.
    idx_list = getattr(op, "idx_list", None)
    if idx_list is None:
        return tuple(_index(x) for x in index_ins)

    def bound(k):
        return None if k is None else int(np.asarray(index_ins[k][1]).item())

    return tuple(slice(bound(e.start), bound(e.stop), bound(e.step)) if isinstance(e, slice)
                 else _index(index_ins[e]) for e in idx_list)


@lowers("All", "Any")
def _all_any(op, node, ins, cx):
    a = ins[0]
    assert not is_tape(a), "a bounds check that reached a parameter"
    reduce = np.all if op_name(op) == "All" else np.any
    axis = getattr(op, "axis", None)
    axis = tuple(axis) if isinstance(axis, (list, tuple)) else axis
    return ("c", np.asarray(reduce(np.asarray(a[1]) != 0, axis=axis), dtype=float))


@lowers("Sum")
def _sum(op, node, ins, cx):
    return cx.low.reduce_sum(ins[0], op.axis)


@lowers("Max", "Min")
def _max_min(op, node, ins, cx):
    # A reduction, as in logsumexp: the chosen axes folded pairwise by Maximum or Minimum.
    arr = np.asarray(ins[0][1])
    axes = tuple(range(arr.ndim)) if op.axis is None else tuple(a % arr.ndim for a in op.axis)
    if not is_tape(ins[0]):
        return ("c", (np.max if op_name(op) == "Max" else np.min)(arr, axis=axes))
    keep = [a for a in range(arr.ndim) if a not in axes]
    runs = np.transpose(arr, keep + list(axes)).reshape([arr.shape[a] for a in keep] + [-1])
    scalar = "Maximum" if op_name(op) == "Max" else "Minimum"
    out = np.empty(runs.shape[:-1], dtype=object)
    for k in np.ndindex(out.shape):
        acc = ("t", np.asarray(runs[k][0]))
        for v in runs[k][1:]:
            acc = cx.low.scalar_op(scalar, [acc, ("t", np.asarray(v))])
        out[k] = np.asarray(acc[1]).item()
    return ("t", out)


@lowers("Elemwise")
def _elemwise(op, node, ins, cx):
    inner = op.scalar_op
    if op_name(inner) == "Switch":
        cx.low.check_condition(_orders_parameters(node.inputs[0], cx.tainted), *ins)
    if op_name(inner) != "Composite":
        return cx.low.scalar_op(op_name(inner), ins)
    # A fused run of scalar ops: lower its own graph over the same inputs.
    if len(inner.fgraph.outputs) != 1:
        raise NotImplementedError("a Composite with several outputs")
    vals = dict(zip(inner.fgraph.inputs, ins))
    outer = dict(zip(inner.fgraph.inputs, node.inputs))
    for n in inner.fgraph.toposort():
        args = [vals[i] if i in vals else ("c", np.asarray(i.data, dtype=float)) for i in n.inputs]
        if op_name(n.op) == "Switch" and n.inputs[0] in outer:
            cx.low.check_condition(_orders_parameters(outer[n.inputs[0]], cx.tainted), *args)
        vals[n.outputs[0]] = cx.low.scalar_op(op_name(n.op), args)
    return vals[inner.fgraph.outputs[0]]


@lowers("DimShuffle")
def _dimshuffle_op(op, node, ins, cx):
    a = ins[0]
    arr = np.asarray(a[1])
    return (a[0], op.perform_shuffle(arr) if hasattr(op, "perform_shuffle") else _dimshuffle(op, arr))


@lowers("Subtensor", "AdvancedSubtensor1", "AdvancedSubtensor")
def _subtensor(op, node, ins, cx):
    a = ins[0]
    if op_name(op) == "Subtensor":
        keys = static_index(op, node.inputs, cx.resolve_scalar)
    else:
        keys = _advanced_key(op, ins[1:])
    return (a[0], np.asarray(a[1])[keys if len(keys) > 1 else keys[0]])


@lowers("Dot", "Dot22", "Gemv", "CGemv", "BatchedDot")
def _dot(op, node, ins, cx):
    # CGemv is `beta * y + alpha * A @ x`; PyTensor emits it with the
    # scaling already folded, so the two operands are the last inputs.
    return cx.low.dot(ins[-2], ins[-1])


@lowers("CumOp")
def _cumop(op, node, ins, cx):
    a = ins[0]
    arr = np.asarray(a[1])
    axis = op.axis if op.axis is not None else 0
    flat = arr if op.axis is not None else arr.ravel()
    out = np.empty(flat.shape, dtype=object if is_tape(a) else float)
    combine = "Add" if getattr(op, "mode", "add") == "add" else "Mul"
    moved = np.moveaxis(flat, axis, 0)
    res = np.moveaxis(out, axis, 0)
    for k in np.ndindex(moved.shape[1:]):
        acc = (a[0], np.array(moved[(0,) + k], dtype=moved.dtype))
        res[(0,) + k] = np.asarray(acc[1]).item()
        for i in range(1, moved.shape[0]):
            nxt = (a[0], np.array(moved[(i,) + k], dtype=moved.dtype))
            acc = cx.low.binary(combine, acc, nxt)
            res[(i,) + k] = np.asarray(acc[1]).item()
    return (a[0], out.reshape(arr.shape))


@lowers("SolveTriangular", "CholeskySolve")
def _solve_triangular(op, node, ins, cx):
    return cx.low.solve_triangular(
        ins[0], ins[1], getattr(op, "lower", True), getattr(op, "b_ndim", 1),
    )


@lowers("ExtractDiag")
def _extract_diag(op, node, ins, cx):
    a = ins[0]
    arr = np.asarray(a[1])
    offset = getattr(op, "offset", 0)
    ax1 = getattr(op, "axis1", 0)
    ax2 = getattr(op, "axis2", 1)
    return (a[0], np.diagonal(arr, offset, ax1, ax2).copy())


@lowers("AllocDiag")
def _alloc_diag(op, node, ins, cx):
    a = ins[0]
    arr = np.asarray(a[1])
    out = np.zeros(arr.shape + (arr.shape[-1],), dtype=arr.dtype)
    for b in np.ndindex(arr.shape[:-1]):
        for i in range(arr.shape[-1]):
            out[b + (i, i)] = arr[b + (i,)]
    if is_tape(a):
        for idx in np.ndindex(out.shape):
            if out[idx] == 0:
                out[idx] = cx.low.w.const_node(0.0)
    return (a[0], out)


@lowers("Cholesky")
def _cholesky(op, node, ins, cx):
    return cx.low.cholesky(ins[0], getattr(op, "lower", True))


@lowers("Convolve1d")
def _convolve1d(op, node, ins, cx):
    return cx.low.convolve1d(*ins)


@lowers("AdvancedIncSubtensor", "AdvancedIncSubtensor1", "IncSubtensor")
def _inc_subtensor(op, node, ins, cx):
    base, values = ins[0], ins[1]
    if op_name(op) == "IncSubtensor":
        key = static_index(op, node.inputs[1:], cx.resolve_scalar)
        if len(key) == 1:
            key = key[0]
    else:
        keys = _advanced_key(op, ins[2:])
        key = keys if len(keys) > 1 else keys[0]
    arr = np.asarray(base[1])
    setting = getattr(op, "set_instead_of_inc", True)
    if not is_tape(base) and not is_tape(values):
        out = np.array(arr, dtype=float)
        if setting:
            out[key] = np.asarray(values[1], dtype=float)
        else:
            # `out[key] +=` adds a repeated index once; a gradient of `x[idx]` repeats them.
            np.add.at(out, key, np.asarray(values[1], dtype=float))
        return ("c", out)
    # Mixed: the destination has to become tape nodes to hold them.
    out = np.empty(arr.shape, dtype=object)
    for k in np.ndindex(arr.shape):
        out[k] = arr[k] if is_tape(base) else cx.low.w.const_node(arr[k])
    v = np.asarray(values[1])
    promoted = np.empty(v.shape, dtype=object)
    for k in np.ndindex(v.shape):
        promoted[k] = v[k] if is_tape(values) else cx.low.w.const_node(v[k])
    if setting:
        out[key] = promoted
    else:
        # Each occurrence of an index adds in turn, as np.add.at does.
        at = np.arange(out.size).reshape(out.shape)[key]
        wide = np.broadcast_to(promoted, np.shape(at))
        flat = out.reshape(-1)
        for k in np.ndindex(np.shape(at)):
            flat[at[k]] = cx.low.w.emit("add", flat[at[k]], wide[k])
    return ("t", out)


@lowers("Alloc")
def _alloc(op, node, ins, cx):
    a = ins[0]
    shape = tuple(int(np.asarray(x[1]).item()) for x in ins[1:])
    arr = np.broadcast_to(np.asarray(a[1]), shape)
    return (a[0], np.array(arr, dtype=object) if is_tape(a) else np.array(arr, dtype=float))


@lowers("AllocEmpty")
def _alloc_empty(op, node, ins, cx):
    # Its contents are unspecified and a set_subtensor fills it; zeros are as good as any.
    return ("c", np.zeros(tuple(int(np.asarray(x[1]).item()) for x in ins)))


def _stack(low, vals):
    """One lowered value from a list of them, constants made nodes if any is not."""
    if not any(is_tape(v) for v in vals):
        return ("c", np.stack([np.asarray(v[1], dtype=float) for v in vals]))
    node = np.frompyfunc(lambda c: low.w.const_node(float(c)), 1, 1)
    return ("t", np.stack([np.asarray(v[1], dtype=object) if is_tape(v) else node(np.asarray(v[1]))
                           for v in vals]))


def _row(value, i):
    return (value[0], np.asarray(value[1])[i])


@lowers("Scan")
def _scan(op, node, ins, cx):
    """Unrolled: the inner graph lowered once per step into the tape.

    The step count is fixed when the graph is traced, and the tape grows with it.
    Sequences, sit-sot and mit-sot states, nit-sot outputs and non-sequences; not a
    `while` loop, mit-mot (gradients of a scan) or an untraced state.
    """
    info = op.info
    if info.as_while or info.mit_mot_in_slices or info.n_untraced_sit_sot:
        raise NotImplementedError("a Scan with a while condition, mit-mot or untraced state")
    if is_tape(ins[0]):
        raise NotImplementedError("a Scan whose step count depends on a parameter")
    steps = int(np.asarray(ins[0][1]).item())
    at = 1
    def take(k):
        nonlocal at
        at += k
        return ins[at - k:at]
    seqs = take(info.n_seqs)
    taps = list(info.mit_sot_in_slices) + list(info.sit_sot_in_slices)
    buffers = take(len(taps))
    take(info.n_nit_sot)  # their lengths; the steps give them
    non_seqs = take(info.n_non_seqs)

    if steps == 0:
        raise NotImplementedError("a Scan of zero steps")
    # Each state keeps its initial rows, then one row per step appended.
    past = [-min(t) for t in taps]
    if any(np.shape(b[1])[0] != p + steps for b, p in zip(buffers, past)):
        raise NotImplementedError("a Scan whose state buffer is not its initial rows and one per step")
    states = [[_row(b, i) for i in range(p)] for b, p in zip(buffers, past)]
    collected = [[] for _ in range(info.n_nit_sot)]
    inner_in, inner_out = op.inner_inputs, op.inner_outputs
    for t in range(steps):
        values = [_row(s, t) for s in seqs]
        for state, p, tap in zip(states, past, taps):
            values += [state[p + t + k] for k in tap]
        memo = dict(zip(inner_in, values + list(non_seqs)))
        _, get = _walk(cx.low, memo, inner_out)
        outs = [get(o) for o in inner_out]
        for state, out in zip(states, outs):
            state.append(out)
        for got, out in zip(collected, outs[len(states):]):
            got.append(out)
    return [_stack(cx.low, s) for s in states] + [_stack(cx.low, c) for c in collected]


@lowers("LogAddExp")
def _logaddexp(op, node, ins, cx):
    # max(a, b) + log(1 + exp(-|a - b|)), written as softplus is, so neither side overflows.
    low, (a, b) = cx.low, ins
    gap = low.unary("Abs", low.binary("Sub", a, b))
    top = low.binary("Mul", low.binary("Add", low.binary("Add", a, b), gap), ("c", np.array(0.5)))
    tail = low.unary("Exp", low.unary("Neg", gap))
    return low.binary("Add", top, low.unary("Log", low.binary("Add", tail, ("c", np.array(1.0)))))


@lowers("LogSumExp")
def _logsumexp(op, node, ins, cx):
    # c + log(sum(exp(x - c))) has the same value and gradient for any constant c, so the
    # shift PyTensor takes as a max is the trace point's max here, folded.
    x, axis = ins[0], op.axis
    vals = np.vectorize(lambda i: cx.low.w.values[int(i)], otypes=[float])(x[1]) if is_tape(x) else np.asarray(x[1])
    c = np.max(vals, axis=axis, keepdims=True)
    total = cx.low.reduce_sum(cx.low.unary("Exp", cx.low.binary("Sub", x, ("c", c))), axis)
    return cx.low.binary("Add", cx.low.unary("Log", total), ("c", np.squeeze(c, axis=axis)))


@lowers("Shape", "Shape_i")
def _shape(op, node, ins, cx):
    # Shapes are fixed once traced; a caller with another shape traces again.
    shape = np.shape(ins[0][1])
    return ("c", np.array(shape if op_name(op) == "Shape" else shape[op.i], dtype=float))


@lowers("MakeVector", "Join")
def _concat(op, node, ins, cx):
    axis = 0
    if op_name(op) == "Join":
        # Newer PyTensor holds the axis on the op; older passes it as a first, 0-d input.
        if getattr(op, "axis", None) is not None:
            axis = op.axis
        else:
            axis, ins = int(np.asarray(ins[0][1]).item()), ins[1:]
    if not any(is_tape(x) for x in ins):
        return ("c", np.concatenate([np.atleast_1d(np.asarray(x[1])) for x in ins], axis=axis))
    # A constant joined to tape values has to become nodes, or its numbers read as indices.
    node = np.frompyfunc(lambda c: cx.low.w.const_node(float(c)), 1, 1)
    arrs = [np.atleast_1d(np.asarray(x[1], dtype=object) if is_tape(x) else node(np.asarray(x[1])))
            for x in ins]
    return ("t", np.concatenate(arrs, axis=axis))


ORDERING = {"GT", "GE", "LT", "LE", "Maximum", "Minimum"}


def _orders_parameters(var, tainted):
    """Whether `var` depends on a parameter through an ordering — what folding at the
    trace point gets wrong. An equality, met on a set of measure zero, is fine to fold."""
    seen, todo = set(), [var]
    while todo:
        v = todo.pop()
        if v in seen or v not in tainted or v.owner is None:
            continue
        seen.add(v)
        scalar = getattr(v.owner.op, "scalar_op", None)
        names = ({op_name(n.op) for n in scalar.fgraph.toposort()} if op_name(scalar) == "Composite"
                 else {op_name(scalar)})
        if names & ORDERING and any(i in tainted for i in v.owner.inputs):
            return True
        # Past an equality are values, not conditions, unless it compares a condition.
        todo.extend(i for i in v.owner.inputs if not names & {"EQ", "NEQ"} or i.dtype == "bool")
    return False


# Rules that only move elements around, through which a folded comparison stays one.
RESHAPES = {_dimshuffle_op, _subtensor, _alloc, _concat}


def _eval_float(var):
    """A constant subgraph's value, computed in float64 where it is integer arithmetic.

    An `AllocEmpty` is zeros: evaluated, it is whatever memory it got.

    `StudentT(nu=3, sigma=10)` scales by an int8 `nu * sigma**2`, which wraps to 44 on
    its own; PyMC's compiled logp upcasts first and gets 300. An index cannot be a
    float, so that one keeps its integer evaluation.
    """
    import pytensor.tensor as pt

    if var.owner is not None and op_name(var.owner.op) == "AllocEmpty":
        return np.zeros(tuple(int(np.asarray(i.eval()).item()) for i in var.owner.inputs))

    def rebuild(v, memo):
        if v not in memo:
            if v.owner is None:
                memo[v] = pt.cast(v, "float64") if v.dtype.startswith(("int", "uint")) else v
            else:
                ins = [rebuild(i, memo) for i in v.owner.inputs]
                memo[v] = v.owner.op.make_node(*ins).outputs[v.owner.outputs.index(v)]
        return memo[v]

    if var.owner is not None and var.dtype.startswith(("int", "uint")):
        try:
            return np.asarray(rebuild(var, {}).eval(), dtype=float)
        except Exception:
            pass
    return np.asarray(var.eval(), dtype=float)


def _stabilize(outputs):
    """PyTensor's own rewrites of log(sigmoid(x)) and log1p(-sigmoid(x)) into softplus.

    A Bernoulli's logit likelihood is written that way, and a logit past about 37
    rounds sigmoid to 1, so log1p(-1) is -inf and the gradient NaN.
    """
    from pytensor.graph.fg import FunctionGraph
    from pytensor.graph.rewriting.basic import in2out
    from pytensor.tensor.rewriting.math import (
        log1msigm_to_softplus, log1p_neg_sigmoid, logsigm_to_softplus)

    fg = FunctionGraph(outputs=outputs, clone=False)
    for rewrite in (logsigm_to_softplus, log1msigm_to_softplus, log1p_neg_sigmoid):
        in2out(rewrite).rewrite(fg)
    return fg.outputs


def lower_graph(inputs, outputs, at, guards=None, on_node=None, branch_on_tape=False):
    """Lower `outputs` over `inputs`, each a tape leaf traced at its value in `at`.

    Returns the writer and each output lowered. A comparison that reaches an input
    folds at `at`; `guards`, a list, collects each as `(op name, inputs, result)`.
    """
    w = TapeWriter()
    low = Lowerer(w, guards, branch_on_tape)

    # Leaves first, in the order the raveled parameter vector uses.
    memo = {}
    for v, val in zip(inputs, at):
        val = np.asarray(val, dtype=float)
        ids = np.empty(val.shape, dtype=object)
        for k in np.ndindex(val.shape):
            ids[k] = w.emit("new_var", repr(float(val[k])))
        memo[v] = ("t", ids)

    nodes, get = _walk(low, memo, outputs)
    if on_node is not None:
        for node in nodes:
            if node.outputs[0] in memo:
                on_node(node, memo[node.outputs[0]], w)
    return w, [get(o) for o in outputs]


def _walk(low, memo, outputs):
    """Lower every node between `memo`'s variables and `outputs` into `memo`.

    Returns the nodes walked and a lookup for any variable; a Scan walks its inner
    graph this way once per step, into the same tape.
    """
    # Fixed once: `memo` grows to hold every intermediate, and asking again
    # would treat those as graph inputs and walk nothing.
    nodes = io_toposort(list(memo), outputs)

    tainted = set(memo)
    for node in nodes:
        if any(i in tainted for i in node.inputs):
            tainted.update(node.outputs)

    def get(var):
        if var in memo:
            return memo[var]
        if not hasattr(var.type, "dtype"):  # a slice or None in an index
            return ("c", var.eval())
        return ("c", _eval_float(var))

    def resolve_scalar(var):
        got = get(var)
        assert not is_tape(got), "an index that depends on a parameter"
        return got[1]

    cx = SimpleNamespace(low=low, resolve_scalar=resolve_scalar, tainted=tainted)
    for node in nodes:
        if not any(i in tainted for i in node.inputs):
            continue
        ins = [get(i) for i in node.inputs]
        if op_name(node.op) in PASSTHROUGH:
            memo[node.outputs[0]] = ins[0]
            continue
        op = getattr(node.op, "core_op", None) or node.op
        name = rule_for(op)
        if name not in LOWER:
            raise NotImplementedError(f"op {name}")
        out = LOWER[name](op, node, ins, cx)
        # Reshaping a folded comparison keeps its record, so a Switch after it can still
        # build the comparison and reshape that.
        if (LOWER[name] in RESHAPES and not is_tape(out)
                and any(id(np.asarray(x[1])) in low.recipes for x in ins if not is_tape(x))):
            low.recipes[id(out[1])] = (out[1], (LOWER[name], op, node, cx), ins)
        # A rule for an op with several outputs returns one value per output.
        memo.update(zip(node.outputs, out) if isinstance(out, list) else [(node.outputs[0], out)])
    return nodes, get


def lower(model, out_path, trace_at=None, test_at=None, on_node=None, log_lik=True):
    """Write the model's tape. Returns the parameter count and what `outputs` named.

    With `log_lik`, each observed variable's own log-likelihood is lowered beside
    the density, elementwise, and named on an `outputs` line — the terms
    `az.loo` reads. The two graphs share their subexpressions and the tape
    numbers equal expressions into one node, so on five of the seven models here
    the terms add no node at all and the module grows 1.35–1.43x, for the second
    forward pass alone. Where the density contracts (`matrix_regression`,
    `lkj_mvnormal`) the per-observation terms are their own nodes and it is
    2.35–2.65x, which is the reason this can be turned off.
    """
    _refuse_discrete(model)
    logp = model.logp(sum=True)
    pointwise = model.logp(vars=model.observed_RVs, sum=False) if log_lik else []
    logp, *pointwise = _stabilize([logp, *pointwise])
    value_vars = model.value_vars
    ip = trace_at if trace_at is not None else model.initial_point()
    w, (root, *terms) = lower_graph(
        value_vars, [logp, *pointwise], [ip[v.name] for v in value_vars], on_node=on_node,
        branch_on_tape=True)
    n_params = sum(np.size(ip[v.name]) for v in value_vars)

    assert is_tape(root), "logp folded to a constant"
    root_id = int(np.asarray(root[1]).item())

    # One entry per observed variable, in the order their terms are written.
    outputs, groups = [], []
    for rv, (kind, vals) in zip(model.observed_RVs, terms):
        vals = np.atleast_1d(np.asarray(vals))
        for k in np.ndindex(vals.shape):
            # A term that does not reach a parameter is a constant, and the
            # module still has to report it in place.
            outputs.append(int(vals[k]) if kind == "t" else w.const_node(float(vals[k])))
        groups.append({"name": rv.name, "shape": list(vals.shape)})

    at = test_at if test_at is not None else ip
    flat = np.concatenate([np.asarray(at[v.name], dtype=float).ravel() for v in value_vars])
    header = [f"n_params {n_params}", "test_params " + " ".join(repr(float(x)) for x in flat)]
    tail = [f"root {root_id}"]
    if outputs:
        tail.append("outputs " + " ".join(str(i) for i in outputs))
    with open(out_path, "w") as f:
        f.write("\n".join(header + w.lines + tail) + "\n")

    return n_params, groups


def _refuse_discrete(model):
    if model.discrete_value_vars:
        names = ", ".join(v.name for v in model.discrete_value_vars)
        raise NotImplementedError(f"discrete parameters ({names}): NUTS samples continuous ones only")


def lower_expansion(model, point, var_names=None):
    """The tape mapping the unconstrained vector to what nutpie calls the expansion.

    Each selected variable of `model.unobserved_value_vars` — a free variable in its
    own space, or a deterministic — raveled and concatenated in `var_names` order,
    free variables then deterministics by default, as nuts-rs-wasm's Numba path lays
    them out. Returns the tape text and one `{name, shape, size, dims}` per name.
    """
    _refuse_discrete(model)
    names = list(var_names) if var_names is not None else [
        v.name for v in [*model.free_RVs, *model.deterministics]]
    available = {v.name: v for v in model.unobserved_value_vars}
    if not names or len(set(names)) != len(names) or set(names) - available.keys():
        raise ValueError("var_names must be unique unobserved model variable names")
    selected = [pt.as_tensor(available[k], allow_xtensor_conversion=True) for k in names]
    # Deterministics written with dims stay XTensor ops until compiled; model.logp()
    # is lowered already, these are not.
    selected = rewrite_graph(selected, include=("lower_xtensor",))

    # Comparisons fold at `point` as in `lower`: here the bounds CheckParameterValue tests.
    value_vars = model.value_vars
    w, lowered = lower_graph(value_vars, selected, [point[v.name] for v in value_vars],
                             branch_on_tape=True)

    outputs, layout = [], []
    for name, (kind, vals) in zip(names, lowered):
        vals = np.asarray(vals)
        if kind == "c" and not any(not l.startswith("new_var") for l in w.lines):
            w.emit("mul_c", "0", "0.0")  # const_node needs an op before it
        outputs += [int(vals[k]) if kind == "t" else w.const_node(float(vals[k]))
                    for k in np.ndindex(vals.shape)]
        dims = list(model.named_vars_to_dims.get(name, ()))
        layout.append({
            "name": name, "shape": list(vals.shape), "size": int(vals.size),
            "dims": [dims[i] if i < len(dims) and dims[i] is not None else f"{name}_dim_{i}"
                     for i in range(vals.ndim)],
        })

    n_params = sum(np.size(point[v.name]) for v in value_vars)
    lines = [f"n_params {n_params}", *w.lines, f"root {outputs[0]}",
             "outputs " + " ".join(map(str, outputs))]
    return "\n".join(lines) + "\n", layout


def _dimshuffle(op, arr):
    # `new_order` names the input's own axes, so the dropped ones go last and then
    # away, rather than being squeezed first and shifting the numbering.
    arr = np.asarray(arr)
    order = [o for o in op.new_order if o != "x"]
    arr = np.transpose(arr, order + list(getattr(op, "drop", [])))
    arr = arr.reshape(arr.shape[:len(order)])
    for i, o in enumerate(op.new_order):
        if o == "x":
            arr = np.expand_dims(arr, axis=i)
    return arr


def linear_regression(n=20):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = 1.0 + 1.8 * x + rng.normal(scale=0.5, size=n)
    with pm.Model() as m:
        alpha = pm.Normal("alpha", 0, 10)
        beta = pm.Normal("beta", 0, 10)
        sigma = pm.Exponential("sigma", 1)
        pm.Normal("y", mu=alpha + beta * x, sigma=sigma, observed=y)
    return m


def logistic(n=30):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 + 1.2 * x)))).astype(int)
    with pm.Model() as m:
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        pm.Bernoulli("y", logit_p=a + b * x, observed=y)
    return m


def eight_schools():
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    s = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu, tau, shape=8)
        pm.Normal("obs", theta, s, observed=y)
    return m


def matrix_regression(n=60, k=5):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(n, k))
    y = X @ np.arange(1.0, k + 1) + rng.normal(scale=0.4, size=n)
    with pm.Model() as m:
        beta = pm.Normal("beta", 0, 5, shape=k)
        sigma = pm.Exponential("sigma", 1)
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
    return m


def lkj_mvnormal(n=30, k=3):
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(n, k))
    with pm.Model() as m:
        chol, _, _ = pm.LKJCholeskyCov(
            "chol_cov", n=k, eta=2.0, sd_dist=pm.Exponential.dist(1.0)
        )
        mu = pm.Normal("mu", 0, 5, shape=k)
        pm.MvNormal("obs", mu=mu, chol=chol, observed=Y)
    return m


def varying_intercepts(n=40, g=4):
    rng = np.random.default_rng(3)
    idx = rng.integers(0, g, size=n)
    x = rng.normal(size=n)
    y = idx * 0.5 + 1.2 * x + rng.normal(scale=0.4, size=n)
    with pm.Model() as m:
        a = pm.Normal("a", 0, 1, shape=g)
        b = pm.Normal("b", 0, 1)
        sigma = pm.Exponential("sigma", 1)
        pm.Normal("y", mu=a[idx] + b * x, sigma=sigma, observed=y)
    return m


def student_t(n=25):
    rng = np.random.default_rng(2)
    y = rng.standard_t(5, size=n)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 5)
        sigma = pm.HalfNormal("sigma", 2)
        pm.StudentT("y", nu=5.0, mu=mu, sigma=sigma, observed=y)
    return m


KNOWN_WRONG = set()

MODELS = {
    "linear_regression": linear_regression,
    "logistic": logistic,
    "eight_schools": eight_schools,
    "varying_intercepts": varying_intercepts,
    "matrix_regression": matrix_regression,
    "lkj_mvnormal": lkj_mvnormal,
    "student_t": student_t,
}

# Only the offline artifact builder needs a tapewasm checkout — inside Pyodide
# the emitter is the published wasm, so this must not fail on import.
REPO = os.environ.get("TAPEWASM")


def jitter(ip, rng, scale):
    return {k: np.asarray(v, dtype=float) + rng.normal(scale=scale, size=np.shape(v))
            for k, v in ip.items()}


def check(name, build):
    model = build()
    rng = np.random.default_rng(11)
    ip = model.initial_point()
    # Trace at one point, evaluate at another: anything wrongly folded into a
    # constant shows up as a mismatch at the second.
    trace_at = jitter(ip, rng, 0.3)
    test_at = jitter(ip, rng, 0.7)
    path = os.path.join(REPO, "target", f"{name}.tape")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_params, log_lik = lower(model, path, trace_at, test_at)

    out = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "tapewasm-codegen",
         "--example", "tape_from_text", "--", os.path.abspath(path)],
        cwd=REPO, capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"{name}: FAILED\n{out.stderr[-800:]}")
        return
    got_lp = None
    got_g, got_ll = [], []
    for line in out.stdout.splitlines():
        if line.startswith("lp "):
            got_lp = float(line.split()[1])
        elif line.startswith("grad "):
            got_g.append(float(line.split()[1]))
        elif line.startswith("ll "):
            got_ll.append(float(line.split()[1]))
    want_lp = float(model.compile_logp()(test_at))
    want_g = np.asarray(model.compile_dlogp()(test_at), dtype=float)
    got_g = np.asarray(got_g)
    # The module's own log-likelihood terms, against PyMC's for the same point.
    terms = model.logp(vars=model.observed_RVs, sum=False)
    want_ll = np.concatenate([
        np.atleast_1d(np.asarray(v, dtype=float)).ravel()
        for v in model.compile_fn(terms, inputs=model.value_vars,
                                  on_unused_input="ignore", point_fn=True)(test_at)
    ]) if terms else np.zeros(0)

    def rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1.0)

    lp_err = rel(got_lp, want_lp)
    g_err = max(rel(a, b) for a, b in zip(got_g, want_g)) if len(got_g) else float("nan")
    if len(got_ll) != len(want_ll):
        print(f"{name}: {len(got_ll)} log_lik terms, PyMC has {len(want_ll)}")
        return
    ll_err = max((rel(a, b) for a, b in zip(got_ll, want_ll)), default=0.0)
    nodes = [l for l in out.stderr.splitlines() if "replayed" in l]
    flag = "  KNOWN WRONG" if name in KNOWN_WRONG else ""
    print(f"{name:<20} params {n_params:>3}  lp rel {lp_err:.2e}  grad rel {g_err:.2e}"
          f"  log_lik {len(got_ll):>4} rel {ll_err:.2e}   {nodes[0] if nodes else ''}{flag}")


if __name__ == "__main__":
    for name, build in MODELS.items():
        check(name, build)
