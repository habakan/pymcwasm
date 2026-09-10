// Five seeds of the MMM, each in a fresh browser context: a cold fit, the same
// fit again on the loaded page, and a third run that only counts module calls.
//
//   VENDOR=/path/to/tapewasm/ts ENGINE=chromium ART=artifact node bench/mmm/bench.mjs
//
// Responses are gzipped when the browser asks, so transfer is what a static host sends.
import { createServer } from "node:http";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import { chromium, firefox, webkit } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const vendor = process.env.VENDOR ?? resolve(here, "../../node_modules/tapewasm");
const engineName = process.env.ENGINE ?? "chromium";
const art = process.env.ART ?? "artifact";
const settings = { targetAccept: Number(process.env.TA ?? 0.9), gradBased: (process.env.GB ?? "1") === "1" };
const PORT = Number(process.env.PORT ?? 8171);

const types = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const server = createServer(async (req, res) => {
  const p = new URL(req.url, "http://x").pathname;
  const file = p.startsWith("/vendor/") ? resolve(vendor, "." + p.slice(7)) : resolve(here, "." + p);
  try {
    let body = await readFile(file);
    const headers = { "content-type": types[extname(file)] ?? "application/octet-stream", "cache-control": "no-store" };
    if (/gzip/.test(req.headers["accept-encoding"] ?? "")) {
      body = gzipSync(body, { level: 9 });
      headers["content-encoding"] = "gzip";
    }
    res.writeHead(200, headers).end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(PORT);

const browser = await { chromium, firefox, webkit }[engineName].launch();
const runs = [];
for (const seed of [42, 142, 242, 342, 442]) {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  await page.goto(`http://127.0.0.1:${PORT}/bench.html?a=${art}`);
  await page.waitForFunction(() => window.ready);
  const fit = (extra = {}) => page.evaluate((a) => window.fit(a), { seed, ...settings, ...extra });
  const cold = await fit();
  const reused = await fit();
  const counted = await fit({ count: true });
  runs.push({ seed, cold, reusedMs: reused.sampleMs, calls: counted.calls });
  console.log(`${engineName} ${art} seed ${seed}: sample ${cold.sampleMs.toFixed(0)} ms, ` +
    `calls ${counted.calls}, div ${cold.divergences}, navigation→posterior ${cold.sinceNavMs.toFixed(0)} ms`);
  await ctx.close();
}
await browser.close();
server.close();
await mkdir(resolve(here, "raw"), { recursive: true });
await writeFile(resolve(here, "raw", `${engineName}-${art}.json`), JSON.stringify({ engine: engineName, art, settings, runs }));
