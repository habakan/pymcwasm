// Pyodide in a worker: pymcwasm.fit (tapewasm's mean-field ADVI) on the seven models and
// on the live-decoder's network written in PyMC.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/pyodide.mjs";

const say = (msg) => postMessage({ msg });

onmessage = async ({ data: { models, n } }) => {
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
    py.FS.writeFile("/digits.json", await (await fetch("/examples/live-decoder/digits.json")).text());
    py.globals.set("TAPEWASM", new URL("/vendor/pkg/tapewasm.js", location.href).href);
    py.globals.set("MODELS", py.toPy(models));
    py.globals.set("N_ITERS", n);
    postMessage({ msg: "running" });
    const out = await py.runPythonAsync(`
import sys, time, json
sys.path.insert(0, "/pkg")
import numpy as np, pytensor
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.linker = "py"
import pymc as pm
import pymcwasm
from pymcwasm.lowering import MODELS as ALL

rows = []
for name in MODELS:
    t = time.perf_counter()
    approx = await pymcwasm.fit(ALL[name](), n=N_ITERS, tapewasm_path=TAPEWASM)
    rows.append({"model": name, "names": approx.names, "mean": approx.mean.tolist(), "std": approx.std.tolist(),
                 "seconds": time.perf_counter() - t, "fit_ms": approx.ms, "loss_first": float(approx.hist[:100].mean()),
                 "loss_last": float(approx.hist[-100:].mean())})

raw = json.load(open("/digits.json"))
x = np.asarray(raw["pixels"], dtype=float).reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196)
with pm.Model() as decoder:
    z = pm.Normal("z", 0, 1, shape=(len(x), 4))
    w1 = pm.Normal("w1", 0, 1, shape=(4, 20)); b1 = pm.Normal("b1", 0, 1, shape=20)
    w2 = pm.Normal("w2", 0, 1, shape=(20, 196)); b2 = pm.Normal("b2", 0, 1, shape=196)
    mu = pm.math.sigmoid(pm.math.sigmoid(z @ w1 + b1) @ w2 + b2)
    pm.Normal("x", mu, 0.05, observed=x)
t = time.perf_counter()
compiled = await pymcwasm.compile(decoder, tapewasm_path=TAPEWASM, log_lik=False)
compile_s = time.perf_counter() - t
approx = await compiled.fit(n=4000, learning_rate=0.02, mc_samples=3, seed=1)
post = approx.sample(draws=1, seed=0)["posterior"]
m = {v: approx.mean[[i for i, nm in enumerate(approx.names) if nm.split("[")[0] == v]] for v in ("z", "w1", "b1", "w2", "b2")}
sig = lambda a: 1 / (1 + np.exp(-a))
rec = sig(sig(m["z"].reshape(-1, 4) @ m["w1"].reshape(4, 20) + m["b1"]) @ m["w2"].reshape(20, 196) + m["b2"])
rows.append({"model": "decoder", "params": len(approx.names), "compile_s": compile_s, "fit_ms": approx.ms,
             "mse": float(((rec - x) ** 2).mean()), "sample_shape": list(post["z"].shape)})
json.dumps(rows)
`);
    postMessage({ done: JSON.parse(out) });
  } catch (e) {
    postMessage({ error: String(e && e.stack || e) });
  }
};
