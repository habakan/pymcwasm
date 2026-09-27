// Pyodide in a worker, as JupyterLite and marimo run it: `mode="WASM"` against the
// Python linker a C-compiler-less PyTensor falls back to.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/pyodide.mjs";

const say = (msg) => postMessage({ msg });

onmessage = async ({ data: { models, draws } }) => {
  try {
    const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/" });
    say("installing PyMC");
    await py.loadPackage(["micropip", "numpy", "scipy", "pandas", "setuptools"]);
    const mp = py.pyimport("micropip");
    for (const p of ["filelock", "cachetools", "typing-extensions", "cloudpickle", "cons",
                     "logical-unification", "etuples", "miniKanren", "rich", "threadpoolctl",
                     "xarray", "arviz"]) { try { await mp.install(p); } catch {} }
    for (const p of ["pytensor", "pymc"]) await mp.install.callKwargs(p, { deps: false });

    py.FS.mkdirTree("/pkg/pymcwasm");
    for (const f of ["__init__.py", "_bridge.py", "lowering.py", "build.py", "linker.py"]) {
      py.FS.writeFile(`/pkg/pymcwasm/${f}`, await (await fetch(`/pkg/pymcwasm/${f}`)).text());
    }
    py.globals.set("TAPEWASM", new URL("/vendor/pkg/tapewasm.js", location.href).href);
    py.globals.set("MODELS", py.toPy(models));
    py.globals.set("DRAWS", draws);
    say("running");
    const out = await py.runPythonAsync(`
import sys, time, json
sys.path.insert(0, "/pkg")
import numpy as np, pytensor
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.linker = "py"
import pymc as pm
import pymcwasm.linker
from pymcwasm.lowering import MODELS as ALL, jitter
await pymcwasm.linker.load(TAPEWASM)

rows = []
for name in MODELS:
    m = ALL[name]()
    point = jitter(m.initial_point(), np.random.default_rng(3), 0.5)
    row = {"model": name}
    for what in ("logp", "dlogp"):
        f = getattr(m, "compile_" + what)
        want = np.asarray(f()(point))
        t = time.perf_counter(); fw = f(mode="WASM"); got = np.asarray(fw(point))
        row[what + "_compile_s"] = time.perf_counter() - t
        if got.shape != want.shape:
            raise ValueError(f"{name} {what}: WASM {got.shape} {got} vs {want.shape} {want}")
        row[what + "_err"] = float(np.max(np.abs(got - want) / np.maximum(1, np.abs(want))))
    for mode in ("py", "WASM"):
        kw = {} if mode == "py" else {"compile_kwargs": {"mode": "WASM"}}
        with m:
            t = time.perf_counter()
            idata = pm.sample(draws=DRAWS, tune=DRAWS, chains=1, cores=1, random_seed=1,
                              progressbar=False, compute_convergence_checks=False, **kw)
        row["sample_" + mode + "_s"] = time.perf_counter() - t
        row["mean_" + mode] = {k: float(v.mean()) for k, v in idata.posterior.data_vars.items() if v.ndim == 2}
    rows.append(row)
json.dumps(rows)
`);
    postMessage({ done: JSON.parse(out) });
  } catch (e) {
    postMessage({ error: String(e && e.stack || e) });
  }
};
