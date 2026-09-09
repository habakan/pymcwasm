"""Lower a PyMC logp graph onto the tape, and check the module against PyMC.

Walks the PyTensor graph of `model.logp()` and writes the instruction file
`crates/stanwasm-codegen/examples/tape_from_text.rs` replays. Everything
upstream of the value variables is evaluated once and carried as constants;
everything downstream becomes tape nodes, one per scalar element.

Each model is lowered at one point and evaluated at another, so a subgraph
wrongly folded into a constant shows up as a mismatch.

    uv run --with pymc --with scipy python tools/lower_pytensor.py

A spike, not a backend: it covers elementwise ops, sums, indexing, contraction
and the bounds checks the transforms make vacuous. `Cholesky` and `Scan` are
not lowered, and a `Switch` on a parameter is refused rather than resolved at
the trace point.
"""

import os
import subprocess
import sys

import numpy as np
import pymc as pm
try:
    from pytensor.graph.traversal import ancestors, io_toposort
except ImportError:
    from pytensor.graph.basic import ancestors, io_toposort


class TapeWriter:
    def __init__(self):
        self.lines = []
        self.n = 0

    def emit(self, *parts):
        self.lines.append(" ".join(str(p) for p in parts))
        self.n += 1
        return self.n - 1

    def const_node(self, v):
        """A constant that has to reach the tape as a node.

        A leaf recorded after the first non-leaf op is a constant, not a
        parameter, so this must never run before one exists.
        """
        assert any(not l.startswith("new_var") for l in self.lines), "constant before any op"
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

PASSTHROUGH = {
    "CheckParameterValue", "ScalarFromTensor", "TensorFromScalar", "Identity",
    "SpecifyShape", "Cast", "ViewOp", "DeepCopyOp", "Unbroadcast",
}


class Lowerer:
    def __init__(self, w):
        self.w = w

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
            neg = self.unary("Neg", ins[0])
            den = self.binary("Add", self.unary("Exp", neg), ("c", np.array(1.0)))
            return self.binary("TrueDiv", ("c", np.array(1.0)), den)
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
        if name in COMPARISON:
            # On constants this is a real test the data decides. On a tape value
            # it is a bounds check, true by construction after the transforms.
            if all(not is_tape(x) for x in ins):
                vals = [np.asarray(x[1]) for x in ins]
                if len(vals) == 1:
                    r = COMPARISON[name](vals[0]) if name in UNARY_TEST else vals[0]
                else:
                    r = vals[0]
                    for v in vals[1:]:
                        r = COMPARISON[name](r, v)
                return ("c", np.asarray(r).astype(float))
            return ("c", np.array(1.0))
        if name == "Second":
            return ins[1]
        if name in ("Identity", "Cast", "ScalarIdentity"):
            return ins[0]
        if name == "Sign":
            if not is_tape(ins[0]):
                return ("c", np.sign(ins[0][1]))
            return self.binary("TrueDiv", ins[0], self.unary("Abs", ins[0]))
        if name == "Softplus":
            return self.unary("Log", self.binary(
                "Add", self.unary("Exp", ins[0]), ("c", np.array(1.0))))
        if name == "Log1mexp":
            inner = self.unary("Exp", ins[0])
            return self.unary("Log", self.binary("Sub", ("c", np.array(1.0)), inner))
        if name == "Erf":
            raise NotImplementedError("Erf has no emitter arm")
        raise NotImplementedError(f"scalar op {name}")

    def select(self, cond, a, b):
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

    def dot(self, a, b):
        """Contraction as elementwise products and a sum over the shared axis.

        The tape has a contraction node, but it wants a contiguous run of tape
        values and a stride, which the instruction stream cannot name — equal
        expressions are numbered into one node, so positions do not survive.
        Written this way the re-rolled loop finds the per-row block instead.
        """
        A, B = np.asarray(a[1]), np.asarray(b[1])
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

    def solve_triangular(self, a, b, lower=True):
        """Forward or back substitution, one scalar at a time.

        `x[i] = (b[i] - sum_{m<i} L[i][m] x[m]) / L[i][i]`, and the mirror of it
        for an upper factor. `Blockwise` batches, and the right-hand side is a
        vector or a matrix of columns.
        """
        A, B = np.asarray(a[1]), np.asarray(b[1])
        tape = is_tape(a) or is_tape(b)
        dt = object if tape else float
        batch = A.shape[:-2]
        k = A.shape[-1]
        cols = B.shape[-1] if B.ndim > A.ndim - 1 else None
        out_shape = batch + ((k,) if cols is None else (k, cols))
        out = np.empty(out_shape, dtype=dt)
        wa = lambda v: ("t" if is_tape(a) else "c", np.array(v, dtype=object if is_tape(a) else float))
        wb = lambda v: ("t" if is_tape(b) else "c", np.array(v, dtype=object if is_tape(b) else float))
        wo = lambda v: ("t" if tape else "c", np.array(v, dtype=dt))
        for bi in np.ndindex(batch):
            order = range(k) if lower else range(k - 1, -1, -1)
            for c in ([None] if cols is None else range(cols)):
                for i in order:
                    rhs = bi + ((i,) if c is None else (i, c))
                    acc = wb(B[rhs])
                    ms = range(i) if lower else range(i + 1, k)
                    for m in ms:
                        at = bi + ((m,) if c is None else (m, c))
                        term = self.binary("Mul", wa(A[bi + (i, m)]), wo(out[at]))
                        acc = self.binary("Sub", acc, term)
                    div = self.binary("TrueDiv", acc, wa(A[bi + (i, i)]))
                    out[rhs] = np.asarray(div[1]).item()
        return ("t" if tape else "c", out)

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
            acc = run[0]
            for nxt in run[1:]:
                acc = self.w.emit("add", acc, nxt)
            out[k] = acc
        return ("t", out if out_shape else np.array(out.item(), dtype=object))


def lower(model, out_path, trace_at=None, test_at=None):
    logp = model.logp(sum=True)
    value_vars = model.value_vars
    ip = trace_at if trace_at is not None else model.initial_point()

    w = TapeWriter()
    low = Lowerer(w)

    # Leaves first, in the order the raveled parameter vector uses.
    point, memo, n_params = {}, {}, 0
    for v in value_vars:
        val = np.asarray(ip[v.name], dtype=float)
        point[v.name] = val
        ids = np.empty(val.shape, dtype=object)
        for k in np.ndindex(val.shape):
            ids[k] = w.emit("new_var", repr(float(val[k])))
        memo[v] = ("t", ids)
        n_params += val.size

    tainted = set(memo)
    for node in io_toposort(list(memo), [logp]):
        if any(i in tainted for i in node.inputs):
            tainted.update(node.outputs)

    def get(var):
        if var in memo:
            return memo[var]
        return ("c", np.asarray(var.eval(), dtype=float))

    def resolve_scalar(var):
        got = get(var)
        assert not is_tape(got), "an index that depends on a parameter"
        return got[1]

    for node in io_toposort(list(memo), [logp]):
        if not any(i in tainted for i in node.inputs):
            continue
        op = node.op
        name = op_name(op)
        ins = [get(i) for i in node.inputs]

        if name in PASSTHROUGH:
            memo[node.outputs[0]] = ins[0]
            continue
        core = getattr(op, "core_op", None)
        if core is not None:
            op, name = core, op_name(core)
        inner = getattr(op, "scalar_op", None)
        if name == "Sum" or (name == "CAReduce" and op_name(inner) == "Add"):
            memo[node.outputs[0]] = low.reduce_sum(ins[0], op.axis)
            continue
        if inner is not None:  # Elemwise
            memo[node.outputs[0]] = low.scalar_op(op_name(inner), ins)
            continue
        if name == "DimShuffle":
            a = ins[0]
            arr = np.asarray(a[1])
            memo[node.outputs[0]] = (a[0], op.perform_shuffle(arr) if hasattr(op, "perform_shuffle") else _dimshuffle(op, arr))
            continue
        if name in ("Subtensor", "AdvancedSubtensor1", "AdvancedSubtensor"):
            a = ins[0]
            keys = tuple(np.asarray(x[1]).astype(int) if not is_tape(x) else None for x in ins[1:])
            memo[node.outputs[0]] = (a[0], np.asarray(a[1])[keys if len(keys) > 1 else keys[0]])
            continue
        if name in ("Dot", "Dot22", "Gemv", "CGemv", "BatchedDot"):
            # CGemv is `beta * y + alpha * A @ x`; PyTensor emits it with the
            # scaling already folded, so the two operands are the last inputs.
            memo[node.outputs[0]] = low.dot(ins[-2], ins[-1])
            continue
        if name == "CumOp":
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
                    acc = low.binary(combine, acc, nxt)
                    res[(i,) + k] = np.asarray(acc[1]).item()
            memo[node.outputs[0]] = (a[0], out.reshape(arr.shape))
            continue
        if name in ("SolveTriangular", "CholeskySolve"):
            memo[node.outputs[0]] = low.solve_triangular(
                ins[0], ins[1], getattr(op, "lower", True)
            )
            continue
        if name == "ExtractDiag":
            a = ins[0]
            arr = np.asarray(a[1])
            offset = getattr(op, "offset", 0)
            ax1 = getattr(op, "axis1", 0)
            ax2 = getattr(op, "axis2", 1)
            memo[node.outputs[0]] = (a[0], np.diagonal(arr, offset, ax1, ax2).copy())
            continue
        if name == "AllocDiag":
            a = ins[0]
            arr = np.asarray(a[1])
            out = np.zeros(arr.shape + (arr.shape[-1],), dtype=arr.dtype)
            for b in np.ndindex(arr.shape[:-1]):
                for i in range(arr.shape[-1]):
                    out[b + (i, i)] = arr[b + (i,)]
            if is_tape(a):
                for idx in np.ndindex(out.shape):
                    if out[idx] == 0:
                        out[idx] = low.w.const_node(0.0)
            memo[node.outputs[0]] = (a[0], out)
            continue
        if name == "Cholesky":
            memo[node.outputs[0]] = low.cholesky(ins[0], getattr(op, "lower", True))
            continue
        if name in ("AdvancedIncSubtensor", "AdvancedIncSubtensor1", "IncSubtensor"):
            base, values = ins[0], ins[1]
            if name == "IncSubtensor":
                key = static_index(op, node.inputs[1:], resolve_scalar)
                if len(key) == 1:
                    key = key[0]
            else:
                keys = tuple(np.asarray(x[1]).astype(int) for x in ins[2:])
                key = keys if len(keys) > 1 else keys[0]
            arr = np.asarray(base[1])
            setting = getattr(op, "set_instead_of_inc", True)
            if not is_tape(base) and not is_tape(values):
                out = np.array(arr, dtype=float)
                if setting:
                    out[key] = np.asarray(values[1], dtype=float)
                else:
                    out[key] += np.asarray(values[1], dtype=float)
                memo[node.outputs[0]] = ("c", out)
                continue
            # Mixed: the destination has to become tape nodes to hold them.
            out = np.empty(arr.shape, dtype=object)
            for k in np.ndindex(arr.shape):
                out[k] = arr[k] if is_tape(base) else low.w.const_node(arr[k])
            v = np.asarray(values[1])
            promoted = np.empty(v.shape, dtype=object)
            for k in np.ndindex(v.shape):
                promoted[k] = v[k] if is_tape(values) else low.w.const_node(v[k])
            if setting:
                out[key] = promoted
            else:
                selected = np.asarray(out[key], dtype=object)
                wide = np.broadcast_to(promoted, selected.shape)
                summed = np.empty(selected.shape, dtype=object)
                for k in np.ndindex(selected.shape):
                    summed[k] = low.w.emit("add", selected[k], wide[k])
                out[key] = summed
            memo[node.outputs[0]] = ("t", out)
            continue
        if name == "Alloc":
            a = ins[0]
            shape = tuple(int(np.asarray(x[1]).item()) for x in ins[1:])
            arr = np.broadcast_to(np.asarray(a[1]), shape)
            memo[node.outputs[0]] = (a[0], np.array(arr, dtype=object) if is_tape(a) else np.array(arr, dtype=float))
            continue
        if name in ("MakeVector", "Join"):
            arrs = [np.asarray(x[1]) for x in ins if not (len(ins) > 1 and x is ins[0] and name == "Join")]
            if any(is_tape(x) for x in ins):
                memo[node.outputs[0]] = ("t", np.concatenate([np.atleast_1d(a) for a in arrs]))
            else:
                memo[node.outputs[0]] = ("c", np.concatenate([np.atleast_1d(a) for a in arrs]))
            continue
        raise NotImplementedError(f"op {name}")

    root = memo[logp]
    assert is_tape(root), "logp folded to a constant"
    root_id = int(np.asarray(root[1]).item())

    at = test_at if test_at is not None else point
    flat = np.concatenate([np.asarray(at[v.name], dtype=float).ravel() for v in value_vars])
    header = [f"n_params {n_params}", "test_params " + " ".join(repr(float(x)) for x in flat)]
    with open(out_path, "w") as f:
        f.write("\n".join(header + w.lines + [f"root {root_id}"]) + "\n")

    return n_params


def _dimshuffle(op, arr):
    arr = np.asarray(arr)
    for d in sorted(getattr(op, "drop", []), reverse=True):
        arr = np.squeeze(arr, axis=d)
    order = [o for o in op.new_order if o != "x"]
    if order:
        arr = np.transpose(arr, order)
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


# Lowers, and is wrong: see NOTES.md. The linear algebra it needs is exact
# against numpy on its own, so the fault is in the plumbing around the LKJ
# transform, not in the decomposition. Listed so the check keeps reporting it
# rather than leaving it looking unattempted.
KNOWN_WRONG = {"lkj_mvnormal"}

MODELS = {
    "linear_regression": linear_regression,
    "logistic": logistic,
    "eight_schools": eight_schools,
    "varying_intercepts": varying_intercepts,
    "matrix_regression": matrix_regression,
    "lkj_mvnormal": lkj_mvnormal,
    "student_t": student_t,
}

# The emitter is a cargo example in the stanwasm checkout: this step needs Rust
# and that repository, which is why the artifacts are committed and the demo
# does not.
REPO = os.environ.get("STANWASM")
if not REPO:
    raise SystemExit("set STANWASM to a checkout of github.com/habakan/stanwasm")


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
    n_params = lower(model, path, trace_at, test_at)

    out = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "stanwasm-codegen",
         "--example", "tape_from_text", "--", os.path.abspath(path)],
        cwd=REPO, capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"{name}: FAILED\n{out.stderr[-800:]}")
        return
    got_lp = None
    got_g = []
    for line in out.stdout.splitlines():
        if line.startswith("lp "):
            got_lp = float(line.split()[1])
        elif line.startswith("grad "):
            got_g.append(float(line.split()[1]))
    want_lp = float(model.compile_logp()(test_at))
    want_g = np.asarray(model.compile_dlogp()(test_at), dtype=float)
    got_g = np.asarray(got_g)

    def rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1.0)

    lp_err = rel(got_lp, want_lp)
    g_err = max(rel(a, b) for a, b in zip(got_g, want_g)) if len(got_g) else float("nan")
    nodes = [l for l in out.stderr.splitlines() if "replayed" in l]
    flag = "  KNOWN WRONG" if name in KNOWN_WRONG else ""
    print(f"{name:<20} params {n_params:>3}  lp rel {lp_err:.2e}  grad rel {g_err:.2e}"
          f"   {nodes[0] if nodes else ''}{flag}")


if __name__ == "__main__":
    for name, build in MODELS.items():
        check(name, build)
