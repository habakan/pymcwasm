// The browser half: Pyodide in a worker, as JupyterLite runs it, one model per worker —
// Pyodide's process_time() stops working past about 2,147 s of CPU in one instance.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/pyodide.mjs";

onmessage = async ({ data: { model: name, budget } }) => {
  try {
    const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/" });
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
    py.FS.writeFile("/pkg/bench.py", await (await fetch("/bench/bench.py")).text());
    py.FS.mkdirTree(`/models/${name}`);
    for (const f of ["model.py", "data.json"]) {
      py.FS.writeFile(`/models/${name}/${f}`, await (await fetch(`/bench/models/${name}/${f}`)).text());
    }
    py.globals.set("TAPEWASM", new URL("/vendor/pkg/tapewasm.js", location.href).href);
    py.globals.set("NAME", name);
    py.globals.set("BUDGET", budget);
    py.globals.set("report", (row) => postMessage({ row: JSON.parse(row) }));
    const versions = await py.runPythonAsync(`
import sys, json, time
sys.path.insert(0, "/pkg")
import pytensor
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.linker = "py"
import pymc as pm, arviz
import bench, pymcwasm, pymcwasm.linker
await pymcwasm.linker.load(TAPEWASM)

source, data = open(f"/models/{NAME}/model.py").read(), json.load(open(f"/models/{NAME}/data.json"))
evals = None
for sampler in ("wasm", "pymcwasm", "py"):
    row = {"model": NAME, "env": "pyodide", "sampler": sampler}
    try:
        model = bench.load(source, data)
        if sampler == "wasm":
            row.update(bench.pymc_nuts(model, compile_kwargs={"mode": "WASM"}))
            evals = row["grad_evals"]
        elif sampler == "pymcwasm":
            row.update(await bench.pymcwasm_sample(model, TAPEWASM))
        elif evals and (floor := evals * bench.seconds_per_gradient(model)) > BUDGET:
            # A floor: the gradients alone, before PyMC's own Python per step.
            row["skipped"] = f"at least {floor:.0f} s projected"
        else:
            row.update(bench.pymc_nuts(model))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    report(json.dumps(row))
json.dumps({"pyodide": sys.version.split()[0], "pymc": pm.__version__,
            "pytensor": pytensor.__version__, "arviz": arviz.__version__})
`);
    postMessage({ done: JSON.parse(versions) });
  } catch (e) {
    postMessage({ error: String(e && e.stack || e) });
  }
};
