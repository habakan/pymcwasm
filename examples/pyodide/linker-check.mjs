// `mode="WASM"` under Pyodide in a worker, against the Python linker, in Chromium.
// Slow — it installs PyMC from PyPI — so it is not part of `npm test`.

import { chromium } from "playwright";
import { server } from "../../serve.mjs";

const PORT = Number(process.env.PORT ?? 8140);
const models = (process.env.MODELS ?? "linear_regression,eight_schools,logistic").split(",");
const draws = Number(process.env.DRAWS ?? 300);

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/examples/pyodide/`);
const rows = await page.evaluate(({ models, draws }) => new Promise((done, fail) => {
  const w = new Worker("/examples/pyodide/linker-worker.js", { type: "module" });
  w.onmessage = ({ data }) => {
    if (data.msg) console.log(data.msg);
    if (data.done) done(data.done);
    if (data.error) fail(new Error(data.error));
  };
  w.postMessage({ models, draws });
}), { models, draws });
await browser.close();
server.close();

let failed = false;
for (const r of rows) {
  const speedup = r.sample_py_s / r.sample_WASM_s;
  console.log(`${r.model.padEnd(18)} logp err ${r.logp_err.toExponential(1)}  dlogp err ${r.dlogp_err.toExponential(1)}` +
    `  pm.sample ${r.sample_py_s.toFixed(1)}s (py) vs ${r.sample_WASM_s.toFixed(1)}s (WASM), ${speedup.toFixed(1)}x`);
  console.log(`${"".padEnd(18)} means py ${JSON.stringify(r.mean_py)}\n${"".padEnd(18)} means WASM ${JSON.stringify(r.mean_WASM)}`);
  if (!(r.logp_err < 1e-9 && r.dlogp_err < 1e-9)) failed = true;
}
process.exit(failed ? 1 : 0);
