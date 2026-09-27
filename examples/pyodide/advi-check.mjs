// pymcwasm.fit under Pyodide in a worker, in Chromium; writes the fits for scripts/advi_compare.py.
// Slow — it installs PyMC from PyPI — so it is not part of `npm test`.

import { chromium } from "playwright";
import { server } from "../../serve.mjs";

const PORT = Number(process.env.PORT ?? 8140);
const models = (process.env.MODELS ?? "linear_regression,logistic,eight_schools,varying_intercepts,matrix_regression,lkj_mvnormal,student_t").split(",");
const n = Number(process.env.N_ITERS ?? 10000);

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/examples/pyodide/`);
const rows = await page.evaluate(({ models, n }) => new Promise((done, fail) => {
  const w = new Worker("/examples/pyodide/advi-worker.js", { type: "module" });
  w.onmessage = ({ data }) => {
    if (data.msg) console.log(data.msg);
    if (data.done) done(data.done);
    if (data.error) fail(new Error(data.error));
  };
  w.postMessage({ models, n });
}), { models, n });
await browser.close();
server.close();

import { writeFileSync } from "node:fs";
writeFileSync(process.env.OUT ?? "advi-pyodide.json", JSON.stringify(rows));
for (const r of rows) console.log(JSON.stringify({ ...r, names: undefined, mean: undefined, std: undefined }));
process.exit(0);
