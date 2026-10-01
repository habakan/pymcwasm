// nuts-rs-wasm's demo MMM written in the page: Pyodide, PyMC and PyMC-Marketing from their
// CDNs, the model built, lowered and compiled here, then sampled as bench/mmm.py's rows are.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/pyodide.mjs";

const say = (msg) => postMessage({ msg });
onmessage = async ({ data: { reroll, seeds } }) => {
  try {
    const t0 = performance.now(), at = () => (performance.now() - t0) / 1000;
    const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/" });
    await py.loadPackage(["micropip", "numpy", "scipy", "pandas", "setuptools", "matplotlib",
                          "scikit-learn", "pydantic", "pyyaml"]);
    const mp = py.pyimport("micropip"), skipped = [];
    say("installing PyMC and PyMC-Marketing");
    for (const p of ["filelock", "cachetools", "typing-extensions", "cloudpickle", "cons",
                     "logical-unification", "etuples", "miniKanren", "rich", "threadpoolctl", "xarray",
                     "arviz", "arviz-base", "arviz-stats", "arviz-plots", "narwhals", "patsy",
                     "pyprojroot", "seaborn", "tqdm", "xarray-einstats"]) {
      try { await mp.install(p); } catch { skipped.push(p); }
    }
    // The versions `bench/mmm.py native` measured under CPython.
    for (const p of ["pytensor==3.3.2", "pymc==6.3.2", "pytensor-distributions==0.3.2", "preliz==0.28.0",
                     "pymc-extras==0.15.1", "better-optimize==0.4.2", "pymc-marketing==1.2.0"]) {
      await mp.install.callKwargs(p, { deps: false });
    }
    py.FS.mkdirTree("/pkg/pymcwasm");
    for (const f of ["__init__.py", "_bridge.py", "lowering.py", "build.py", "linker.py"]) {
      py.FS.writeFile(`/pkg/pymcwasm/${f}`, await (await fetch(`../pkg/pymcwasm/${f}`)).text());
    }
    for (const f of ["model.py", "mmm_example.csv"]) {
      py.FS.writeFile(`/${f}`, await (await fetch(`models/mmm/${f}`)).text());
    }
    py.globals.set("TAPEWASM", new URL("../vendor/pkg/tapewasm.js", location.href).href);
    py.globals.set("REROLL", reroll);
    py.globals.set("SEEDS", py.toPy(seeds));
    // Loading ends once everything is imported and tapewasm is up; the model starts after.
    await py.runPythonAsync(`
import sys, types
sys.path.insert(0, "/pkg")
# preliz imports numba for helpers the model does not call.
numba = types.ModuleType("numba")
numba.njit = numba.jit = numba.vectorize = numba.guvectorize = (
    lambda *a, **k: a[0] if len(a) == 1 and callable(a[0]) and not k else (lambda f: f))
numba.prange = range
sys.modules["numba"] = numba
import numpy as np, pytensor
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.linker = "py"
import pymcwasm, pymc_marketing.mmm
await pymcwasm._bridge.load(TAPEWASM)
`);
    const loaded = at();
    say("building the model");
    const out = await py.runPythonAsync(`
import json, time
import importlib.metadata as md
import numpy as np, pymcwasm
t = time.perf_counter()
ns = {"DATA_PATH": "/mmm_example.csv", "SEASONALITY": True}
exec(open("/model.py").read(), ns)
model_s = time.perf_counter() - t
t = time.perf_counter()
compiled = await pymcwasm.compile(ns["model"], tapewasm_path=TAPEWASM, log_lik=False, reroll=REROLL)
compile_s = time.perf_counter() - t
fits = []
for seed in SEEDS:
    f = await compiled.sample(draws=500, warmup=750, seed=seed, chains=2, log_lik=False,
                              target_accept=0.9, grad_based_estimate=True)
    fits.append({"seed": seed, "sampling_seconds": f.ms / 1000,
                 "evals": int(np.asarray(f.stats["n_steps"]).sum()),
                 "divergences": int(np.asarray(f.stats["diverging"])[:, 750:].sum()),
                 "draws": f.chain_draws.ravel().tolist()})
json.dumps({"versions": {p: md.version(p) for p in ("pymc", "pytensor", "pymc-marketing")},
            "layout_id": int(compiled._handle.built.layoutId), "init": compiled.init.tolist(),
            "model_seconds": model_s, "compile_seconds": compile_s, "lower_ms": compiled.lower_ms,
            "emit_ms": compiled.compile_ms, "module_bytes": compiled.module_bytes,
            "nParams": len(compiled.names), "fits": fits})
`);
    postMessage({ done: { loaded_seconds: loaded, skipped, ...JSON.parse(out) } });
  } catch (e) {
    postMessage({ error: String((e && e.stack) || e) });
  }
};
