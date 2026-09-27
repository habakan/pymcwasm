// Runs bench/worker.js in Chromium and writes bench/results-browser.json.
//
//   STANWASM=../tapewasm node bench/browser.mjs      # serve a tapewasm checkout's ts/
//
// Slow: it installs PyMC from PyPI, and the Python linker is the slowest thing timed.

import { writeFileSync, readFileSync } from "node:fs";
import { chromium } from "playwright";
import { server } from "../serve.mjs";

const PORT = Number(process.env.PORT ?? 8140);
const budget = Number(process.env.BUDGET_S ?? 300);
const models = process.env.MODELS?.split(",")
  ?? JSON.parse(readFileSync(new URL("models/index.json", import.meta.url)));

const browser = await chromium.launch();
const page = await browser.newPage();
page.on("console", (m) => console.log(m.text()));
await page.goto(`http://127.0.0.1:${PORT}/`);
const rows = [];
let versions;
for (const model of models) {
  const out = await page.evaluate(({ model, budget }) => new Promise((done, fail) => {
    const rows = [];
    const w = new Worker("/bench/worker.js", { type: "module" });
    w.onmessage = ({ data }) => {
      if (data.row) {
        rows.push(data.row);
        const { means, sds, ...shown } = data.row;
        console.log(JSON.stringify(shown));
      }
      if (data.done) { w.terminate(); done({ rows, versions: data.done }); }
      if (data.error) { w.terminate(); fail(new Error(data.error)); }
    };
    w.postMessage({ model, budget });
  }), { model, budget });
  rows.push(...out.rows);
  versions = out.versions;
}
versions.browser = `chromium ${browser.version()}`;
await browser.close();
server.close();
writeFileSync(new URL("results-browser.json", import.meta.url), JSON.stringify({ versions, rows }));
