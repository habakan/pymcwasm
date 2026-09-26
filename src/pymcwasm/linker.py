"""Run a PyTensor graph as a tapewasm module: `pytensor.function(..., mode="WASM")`.

    import pymcwasm.linker
    f = pytensor.function([x], pt.exp(x).sum(), mode="WASM")

Each call with new input shapes traces the graph at those inputs, emits a module with
npm's `tapewasm` (Node, as `pymcwasm-build` does) and runs it under `wasmtime`. What it
computes is the forward pass alone. The tape has no branch, so a comparison on an input
is taken as it was traced and checked on every call.
"""

import math
import os
import tempfile

import numpy as np
from pytensor.compile.mode import Mode, predefined_linkers, predefined_modes, register_linker, register_mode
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.basic import JITLinker

from .build import compile_tape
from .lowering import compare, lower_graph


class Module:
    """One emitted module in a `wasmtime` store, with the buffers its calls use."""

    def __init__(self, wasm_path, n_params, scratch_init, n_outputs):
        import wasmtime
        from scipy.special import digamma, gammaln
        from scipy.stats import norm

        self.store = wasmtime.Store()
        module = wasmtime.Module.from_file(self.store.engine, wasm_path)
        self.n, self.n_out = n_params, n_outputs
        self.scratch = np.asarray(scratch_init, dtype="<f8").tobytes()
        self.params, self.grads = 0, 8 * n_params
        self.out = self.grads + 8 * n_params
        self.scratch_at = self.out + 8 * max(n_outputs, 1)
        pages = (self.scratch_at + len(self.scratch)) // 65536 + 1
        self.memory = wasmtime.Memory(self.store, wasmtime.MemoryType(wasmtime.Limits(pages, None)))

        f64 = wasmtime.ValType.f64()
        unary = {"exp": math.exp, "log": _log, "sin": math.sin, "cos": math.cos, "tan": math.tan,
                 "asin": math.asin, "acos": math.acos, "atan": math.atan,
                 "lgamma": lambda x: float(gammaln(x)), "digamma": lambda x: float(digamma(x)),
                 "phi": lambda x: float(norm.cdf(x))}
        imports = []
        for imp in module.imports:
            if imp.name == "memory":
                imports.append(self.memory)
            elif imp.name == "pow":
                imports.append(wasmtime.Func(self.store, wasmtime.FuncType([f64, f64], [f64]), _pow))
            else:
                imports.append(wasmtime.Func(self.store, wasmtime.FuncType([f64], [f64]), unary[imp.name]))
        self.exports = wasmtime.Instance(self.store, module, imports).exports(self.store)

    def _call(self, name, x, out_at, k):
        self.memory.write(self.store, np.asarray(x, dtype="<f8").tobytes(), self.params)
        self.memory.write(self.store, self.scratch, self.scratch_at)
        root = self.exports[name](self.store, self.params, out_at, self.n, self.scratch_at)
        return root, np.frombuffer(self.memory.read(self.store, out_at, out_at + 8 * k), "<f8").copy()

    def evaluate(self, x):
        return self._call("evaluate", x, self.out, self.n_out)[1]

    def log_prob_grad(self, x):
        return self._call("log_prob_grad", x, self.grads, self.n)


def _log(x):
    return math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan)


def _pow(a, b):
    try:
        return float(a ** b)
    except (OverflowError, ZeroDivisionError):
        return math.inf
    except ValueError:
        return math.nan


class Traced:
    """The graph's function: traced and compiled once per input shape and branch taken.

    A comparison on an input folds at the trace point; its operands are reported
    beside the outputs, and a call where one of them comes out the other way
    traces again at that call's inputs.
    """

    def __init__(self, fgraph):
        self.fgraph = fgraph
        self.cache = {}

    def compile(self, inputs):
        fg = self.fgraph
        guards = []
        w, outs = lower_graph(fg.inputs, fg.outputs, inputs, guards=guards)
        named = []

        def name(kind, vals):
            vals = np.asarray(vals)
            if kind != "t":
                return ("c", vals)
            named.extend(int(i) for i in vals.ravel())
            return ("t", slice(len(named) - vals.size, len(named)), vals.shape)

        plan = [name(*o) for o in outs]
        checks = [(op, [name(*x) for x in ins], r) for op, ins, r in guards]
        module = None
        if named:
            n_params = sum(np.size(x) for x in inputs)
            tape = "\n".join([f"n_params {n_params}", *w.lines, f"root {named[0]}",
                              "outputs " + " ".join(map(str, named))]) + "\n"
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "graph.wasm")
                built = compile_tape(tape, path)
                module = Module(path, n_params, built["scratchInit"], built["nOutputs"])
        return module, plan, checks

    def __call__(self, *inputs):
        for var, x in zip(self.fgraph.inputs, inputs):
            if not np.issubdtype(np.asarray(x).dtype, np.floating):
                raise NotImplementedError(f"input {var} is {np.asarray(x).dtype}: only float inputs become tape leaves")
        x = np.concatenate([np.ravel(v) for v in inputs]) if inputs else np.zeros(0)
        variants = self.cache.setdefault(tuple(np.shape(v) for v in inputs), [])
        for module, plan, checks in variants:
            flat = module.evaluate(x) if module else None
            if all(np.array_equal(compare(op, [_value(p, flat) for p in ins]), r) for op, ins, r in checks):
                break
        else:
            module, plan, checks = self.compile(inputs)
            variants.append((module, plan, checks))
            flat = module.evaluate(x) if module else None
        return [_value(p, flat).astype(var.type.dtype) for var, p in zip(self.fgraph.outputs, plan)]


def _value(p, flat):
    return p[1] if p[0] == "c" else flat[p[1]].reshape(p[2])


class WasmLinker(JITLinker):
    """A `Linker` that runs the whole graph as one tapewasm module."""

    incompatible_rewrites = ("cxx_only", "BlasOpt", "fusion", "inplace", "scan_save_mem_prealloc")

    def fgraph_convert(self, fgraph, **kwargs):
        return Traced(fgraph)

    def jit_compile(self, fn):
        return fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]


if "wasm" not in predefined_linkers:
    register_linker("wasm", WasmLinker())
if "WASM" not in predefined_modes:
    register_mode("WASM", Mode(WasmLinker(), RewriteDatabaseQuery(include=["fast_run"])))
