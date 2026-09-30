// Drives the page in Chromium: installs PyMC and the bartrs wheel, samples, and fails if
// sigma or the fit is off. Slow, and needs build-wheel.sh first, so not part of `npm test`.

import { chromium } from "playwright";
import { server } from "../../serve.mjs";

const PORT = Number(process.env.PORT ?? 8140);
const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(`http://127.0.0.1:${PORT}/examples/bart/`);
await page.click("#run");
await page.waitForFunction(() => {
  const s = document.getElementById("status").textContent;
  return s.startsWith("sampled in") || !document.getElementById("run").disabled;
}, null, { timeout: 600_000 });
const status = await page.textContent("#status");
const cells = await page.locator("#forest button").count();
const trees = Number(await page.inputValue("#trees"));
await browser.close();
server.close();

// Measured sigma 0.191 and RMSE 0.090 at the page defaults; the bounds leave room for another seed.
const sigma = parseFloat(status.match(/sigma ([\d.]+)/)?.[1]);
const rmse = parseFloat(status.match(/RMSE to sin\(x\) ([\d.]+)/)?.[1]);
const ok = !errors.length && Math.abs(sigma - 0.2) < 0.04 && rmse < 0.15 && cells === trees;
console.log(`${ok ? "ok" : "FAIL"} ${status} · ${cells} trees shown${errors.length ? " errors: " + errors.join("; ") : ""}`);
process.exit(ok ? 0 : 1);
