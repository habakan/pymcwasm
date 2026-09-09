// Drives the Pyodide page in a real browser. Slow — it installs PyMC from PyPI
// on every run — so it is separate from `npm test`.

import { chromium } from "playwright";
import { server } from "../../serve.mjs";

const PORT = Number(process.env.PORT ?? 8140);
const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

await page.goto(`http://127.0.0.1:${PORT}/examples/pyodide/`);
await page.waitForFunction(() => !document.getElementById("go").disabled,
  null, { timeout: 300_000 });
console.log("page ready:", await page.textContent("#status"));

await page.click("#go");
await page.waitForFunction(() => window.done || window.failed, null, { timeout: 300_000 });

const failed = await page.evaluate(() => window.failed);
const out = await page.textContent("#out");
const status = await page.textContent("#status");
await browser.close();
server.close();

console.log(status);
console.log(out);
if (failed || errors.length) {
  console.error("FAILED", failed ?? "", errors.slice(0, 3).join(" | "));
  process.exit(1);
}
console.log("OK");
