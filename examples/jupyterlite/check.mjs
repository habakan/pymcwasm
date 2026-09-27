// Opens the built JupyterLite in Chromium, runs every cell of the notebook, and prints
// what each printed. Slow — the kernel installs PyMC from PyPI — so not in `npm test`.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "_output");
const types = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".ipynb": "application/json", ".wasm": "application/wasm",
  ".svg": "image/svg+xml", ".png": "image/png", ".woff": "font/woff", ".woff2": "font/woff2",
  ".whl": "application/zip",
};
const server = createServer(async (req, res) => {
  let path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (path.endsWith("/")) path += "index.html";
  try {
    const body = await readFile(resolve(root, "." + path));
    res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(Number(process.env.PORT ?? 8150));

const port = server.address().port;
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/lab/index.html?path=PyMC-in-the-browser.ipynb`);
// JupyterLite does not expose its app object, so this drives the menu as a reader would.
await page.waitForSelector("text=Python (Pyodide) | Idle", { timeout: 180_000 });
await page.click("li.lm-MenuBar-item >> text=Run");
await page.click("li.lm-Menu-item >> text=Run All Cells");
// Done once every prompt reads [n], or once a cell has raised, which stops Run All.
const finished = await page.waitForFunction(() => {
  const prompts = [...document.querySelectorAll(".jp-CodeCell .jp-InputPrompt")].map((p) => p.innerText);
  const raised = [...document.querySelectorAll(".jp-OutputArea-output")].some((o) => /Traceback/.test(o.innerText));
  return raised || (prompts.length && prompts.every((p) => /\[\d+\]/.test(p)));
}, null, { timeout: Number(process.env.TIMEOUT_MS ?? 1_200_000), polling: 2000 }).then(() => true, () => false);
if (process.env.SCREENSHOT) await page.screenshot({ path: process.env.SCREENSHOT, fullPage: true });
const prompts = await page.$$eval(".jp-CodeCell .jp-InputPrompt", (ps) => ps.map((p) => p.innerText));

const outputs = await page.evaluate(() =>
  [...document.querySelectorAll(".jp-CodeCell")].map((c) =>
    c.querySelector(".jp-OutputArea")?.innerText.trim() ?? ""));
await browser.close();
server.close();

outputs.forEach((o, i) => console.log(`--- cell ${i + 1} ${prompts[i] ?? ""}\n${/Traceback/.test(o) ? o.slice(-1500) : o.slice(0, 1500)}`));
if (!finished) console.log("timed out before every cell ran");
process.exit(!finished || outputs.some((o) => /Traceback|Error/.test(o)) ? 1 : 0);
