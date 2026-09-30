// Pyodide in a worker: PyMC's own CompoundStep, PGBART on the trees and NUTS on sigma,
// with every logp compiled by `mode="WASM"`. bartrs is the wheel build-wheel.sh makes.
// A per-draw callback posts mu and the forest as sampling runs, at most every 80 ms.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/pyodide.mjs";

const BARTRS = "bartrs-0.4.0-cp313-cp313-pyodide_2025_0_wasm32.whl";
const say = (msg) => postMessage({ msg });
let py;

async function load() {
  py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/" });
  say("installing PyMC");
  await py.loadPackage(["micropip", "numpy", "scipy", "pandas", "setuptools", "matplotlib"]);
  const mp = py.pyimport("micropip");
  for (const p of ["filelock", "cachetools", "typing-extensions", "cloudpickle", "cons",
                   "logical-unification", "etuples", "miniKanren", "rich", "threadpoolctl",
                   "xarray", "arviz", "arviz-base", "arviz-stats"]) { try { await mp.install(p); } catch {} }
  for (const p of ["pytensor", "pymc", "pymc-bart", new URL(`wheels/${BARTRS}`, location.href).href]) {
    await mp.install.callKwargs(p, { deps: false });
  }
  py.FS.mkdirTree("/pkg/pymcwasm");
  for (const f of ["__init__.py", "_bridge.py", "lowering.py", "build.py", "linker.py"]) {
    py.FS.writeFile(`/pkg/pymcwasm/${f}`, await (await fetch(`../../pkg/pymcwasm/${f}`)).text());
  }
  py.FS.writeFile("/pkg/bart_pyodide.py", await (await fetch("bart_pyodide.py")).text());
  py.globals.set("TAPEWASM", new URL("../../vendor/pkg/tapewasm.js", location.href).href);
  say("loading pymc-bart");
  await py.runPythonAsync(`
import sys
sys.path.insert(0, "/pkg")
import pytensor
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.linker = "py"
import bart_pyodide
bart_pyodide.install()
import pymcwasm.linker
await pymcwasm.linker.load(TAPEWASM)
`);
}

onmessage = async ({ data: { n, trees, draws, seed } }) => {
  try {
    if (!py) await load();
    say("sampling");
    py.globals.set("ARGS", py.toPy({ n, trees, draws, seed }));
    py.globals.set("emit", (s) => postMessage({ tick: JSON.parse(s) }));
    const out = await py.runPythonAsync(`
import json, time
import numpy as np, pymc as pm, pymc_bart as pmb
from bartrs import PGBART

def forest_of(step):
    finite = lambda vs: [None if v != v else v for v in vs]
    return [{"var": list(t.split_var), "val": finite(t.split_val), "leaf": finite(t.leaf_val)}
            for t in step.pg_bart.forest]

last = [0.0]
def on_draw(trace, draw):
    now = time.perf_counter()
    if now - last[0] < 0.08 and not draw.is_last:
        return
    last[0] = now
    emit(json.dumps({"i": draw.draw_idx, "tuning": draw.tuning, "mu": draw.point["mu"].tolist(),
                     "trees": forest_of(bart_step)}))

a = ARGS
rng = np.random.default_rng(a["seed"])
x = np.sort(rng.uniform(0, 10, a["n"]))
y = np.sin(x) + rng.normal(0, 0.2, a["n"])
emit(json.dumps({"x": x.tolist(), "y": y.tolist()}))
with pm.Model():
    mu = pmb.BART("mu", x[:, None], y, m=a["trees"])
    sigma = pm.HalfNormal("sigma", 1)
    pm.Normal("y", mu, sigma, observed=y)
    bart_step = PGBART([mu], compile_kwargs={"mode": "WASM"})
    t = time.perf_counter()
    idata = pm.sample(draws=a["draws"], tune=a["draws"], chains=1, cores=1, random_seed=a["seed"],
                      step=[bart_step], callback=on_draw, progressbar=False,
                      compute_convergence_checks=False, compile_kwargs={"mode": "WASM"})
    seconds = time.perf_counter() - t
m = idata.posterior["mu"].values[0]
json.dumps({"x": x.tolist(), "y": y.tolist(), "trees": forest_of(bart_step), "mean": m.mean(0).tolist(),
            "lo": np.quantile(m, 0.05, 0).tolist(), "hi": np.quantile(m, 0.95, 0).tolist(),
            "sigma": float(idata.posterior["sigma"].mean()),
            "rmse": float(np.sqrt(np.mean((m.mean(0) - np.sin(x)) ** 2))), "seconds": seconds})
`);
    postMessage({ done: JSON.parse(out) });
  } catch (e) {
    postMessage({ error: String((e && e.stack) || e) });
  }
};
