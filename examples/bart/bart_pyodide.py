"""What pymc-bart needs patched before it imports and builds a BART under Pyodide."""

import sys
import types


def install():
    # pymc_bart.utils imports numba's jit for plotting helpers the sampler never reaches.
    if "numba" not in sys.modules:
        numba = types.ModuleType("numba")
        numba.jit = lambda *a, **k: (lambda f: f)
        sys.modules["numba"] = numba

    import pymc_bart.bart

    # BART() starts a multiprocessing Manager, which Pyodide cannot spawn; one chain needs a list.
    pymc_bart.bart.Manager = lambda: types.SimpleNamespace(list=list)
