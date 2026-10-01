// The demo MMM in Chromium three ways, written to bench/models/mmm/browser.json for
// `bench/mmm.py table`: nuts-rs-wasm's hosted benchmark page (Numba, reused mode), and
// tapewasm in this repository's pages, written in the page and compiled beforehand.
//
//   node bench/mmm-browser.mjs          # after `bench/mmm.py native`; serves the repository

import { chromium } from "playwright";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { server } from "../serve.mjs";

const PORT = Number(process.env.PORT ?? 8140);
const OUT = new URL("models/mmm/browser.json", import.meta.url);
const NUMBA = "https://pymc-labs.github.io/nuts-rs-wasm/benchmark.html";
// The hosted page is not pinned: refuse a version whose fits are not the ones described,
// and record which one ran.
const hosted = await (await fetch(NUMBA)).text();
for (const want of ["chains:2,tune:750,draws:500", "targetAccept:.9", "[42,142,242,342,442]"]) {
  if (!hosted.includes(want)) throw new Error(`the hosted benchmark page no longer has ${want}`);
}
const hostedSha256 = createHash("sha256").update(hosted).digest("hex");
const browser = await chromium.launch();

// Every request the page and its workers make, from a fresh context.
async function run(url, until, label) {
  const context = await browser.newContext();
  // Bytes as they crossed the wire: the CDNs compress, the local server does not.
  const sizes = [];
  context.on("requestfinished", (r) => sizes.push(r.sizes().catch(() => null)));
  const page = await context.newPage();
  const t0 = Date.now();
  await page.goto(url);
  const result = await until(page);
  const done = (await Promise.all(sizes)).filter(Boolean);
  const bytes = done.reduce((a, s) => a + s.responseBodySize + s.responseHeadersSize, 0);
  const out = { ...result, network: { MB: bytes / 1e6, requests: sizes.length },
                wall_seconds: (Date.now() - t0) / 1000 };
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

fs.writeFileSync(OUT, JSON.stringify({ browser: `chromium ${browser.version()}`, hostedSha256,
  numbaCold, numba, inpage, precompiled }));
await browser.close();
server.close();
