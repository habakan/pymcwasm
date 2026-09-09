// Records the Pyodide page and writes demo.gif.
//
//   node serve.mjs &
//   node scripts/record-demo.mjs
//
// The interesting part is what happens after PyMC has installed, and that takes
// most of a minute — so the page is brought all the way to ready in one context
// which is thrown away, and the recording starts on a second one that reuses the
// same browser's HTTP cache. Trimming afterwards would leave the first frames of
// the gif showing a spinner.

import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, readdirSync, renameSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..");
const URL = process.argv[2] ?? "http://127.0.0.1:8140/examples/pyodide/";
const W = 1280, H = 820;

const out = mkdtempSync(join(tmpdir(), "pymcwasm-demo-"));
const browser = await chromium.launch();

async function ready(page) {
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => {
    const b = document.getElementById("compile");
    return b && !b.disabled;
  }, null, { timeout: 300_000 });
}

// Warm the cache: Pyodide and every wheel come from a CDN, and paying for them
// on camera is most of the runtime.
const warm = await browser.newContext({ viewport: { width: W, height: H } });
console.log("warming…");
await ready(await warm.newPage());
await warm.close();

const context = await browser.newContext({
  viewport: { width: W, height: H },
  recordVideo: { dir: out, size: { width: W, height: H } },
  deviceScaleFactor: 2,
});
const page = await context.newPage();
const t0 = Date.now();
const at = () => ((Date.now() - t0) / 1000).toFixed(1);
const beat = (ms) => page.waitForTimeout(ms);

console.log("recording…");
await ready(page);
// Even warm, Pyodide takes seconds to come up; the gif starts here instead.
const from = Number(at());
console.log(`${at()}s  ready`);
await beat(1200);

await page.click("#compile");
await page.waitForFunction(() => window.done, null, { timeout: 120_000 });
console.log(`${at()}s  compiled and sampled`);
await beat(2500);

// The second draw is the point: compiling is the slow half and it is already
// done, so this returns in milliseconds.
await page.fill("#seed", "7");
await beat(500);
await page.click("#draw");
await beat(1800);
console.log(`${at()}s  sampled again`);

await page.selectOption("#preset", "eight schools (centred)");
await beat(1400);
await page.evaluate(() => { window.done = false; });
await page.click("#compile");
await page.waitForFunction(() => window.done, null, { timeout: 120_000 });
console.log(`${at()}s  a different model`);
await beat(3200);

await context.close();
await browser.close();

const webm = readdirSync(out).find((f) => f.endsWith(".webm"));
const src = join(out, webm);
const gif = resolve(repo, "demo.gif");

// One shared palette for the whole clip; a per-frame one shimmers on the plots.
const palette = join(out, "palette.png");
const start = Math.max(0, from + 0.2).toFixed(2);
execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-ss", start, "-i", src,
  "-vf", "fps=10,scale=900:-1:flags=lanczos,palettegen=stats_mode=diff", palette]);
execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-ss", start, "-i", src, "-i", palette,
  "-lavfi", "fps=10,scale=900:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3",
  gif]);
renameSync(src, resolve(repo, "demo.webm"));
rmSync(out, { recursive: true, force: true });

const size = (p) => (execFileSync("stat", ["-f%z", p]).toString().trim() / 1024 / 1024).toFixed(1);
console.log(`demo.gif ${size(gif)} MB, demo.webm ${size(resolve(repo, "demo.webm"))} MB`);
