// The demo MMM in Chromium three ways, written to bench/models/mmm/browser.json for
// `bench/mmm.py table`: nuts-rs-wasm's hosted benchmark page (Numba, reused mode), and
// tapewasm in this repository's pages, written in the page and compiled beforehand.
//
//   node bench/mmm-browser.mjs          # after `bench/mmm.py native`; serves the repository

import { chromium } from "playwright";
import fs from "node:fs";
import { server } from "../serve.mjs";

const PORT = Number(process.env.PORT ?? 8140);
const OUT = new URL("models/mmm/browser.json", import.meta.url);
const NUMBA = "https://pymc-labs.github.io/nuts-rs-wasm/benchmark.html";
const browser = await chromium.launch();

// Every request the page and its workers make, from a fresh context.
async function run(url, until, label) {
  const context = await browser.newContext();
  let bytes = 0, requests = 0;
  context.on("requestfinished", async (r) => {
    requests++;
    const s = await r.sizes().catch(() => null);
    if (s) bytes += s.responseBodySize + s.responseHeadersSize;
  });
  const page = await context.newPage();
  const t0 = Date.now();
  await page.goto(url);
  const result = await until(page);
  const out = { ...result, network: { MB: bytes / 1e6, requests }, wall_seconds: (Date.now() - t0) / 1000 };
  console.log(label, JSON.stringify({ ...out.network, wall: out.wall_seconds }));
  await context.close();
  return out;
}

const result = (page) => page.waitForFunction(() => window.RESULT, null, { timeout: 1800_000 })
  .then(() => page.evaluate(() => window.RESULT))
  .then((r) => { if (r.error) throw new Error(r.error); return r; });
const status = (page, prefix) => page.waitForFunction(
  (p) => document.querySelector("#status").textContent.startsWith(p), prefix, { timeout: 1800_000 });

const numbaCold = await run(NUMBA, async (page) => {
  const t0 = Date.now();
  await page.selectOption("#mode", "reused");
  await page.click("#run");
  await status(page, "Preparing");
  return { seconds_to_prepare: (Date.now() - t0) / 1000 };
}, "numba cold");
const numba = await run(NUMBA, async (page) => {
  await page.selectOption("#mode", "reused");
  await page.click("#run");
  await status(page, "Complete");
  return JSON.parse(await page.textContent("#results"));
}, "numba");
const inpage = await run(`http://127.0.0.1:${PORT}/bench/mmm-inpage.html?reroll=never`, result, "in the page");
const precompiled = await run(`http://127.0.0.1:${PORT}/bench/mmm-precompiled.html`, result, "compiled beforehand");

fs.writeFileSync(OUT, JSON.stringify({ browser: `chromium ${browser.version()}`, numbaCold, numba, inpage, precompiled }));
await browser.close();
server.close();
