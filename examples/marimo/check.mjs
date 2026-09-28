// Exports the notebook with `marimo export html-wasm`, runs it in Chromium against this
// checkout's wheel and npm's tapewasm, and prints each sampling log line. Slow, so not in `npm test`.

import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../..");
const out = mkdtempSync(join(tmpdir(), "marimo-"));
const port = Number(process.env.PORT ?? 8151);

// The notebook fetches the wheel and tapewasm from the published site; point it here instead.
const nb = readFileSync(join(here, "simpsons-paradox.py"), "utf8")
  .replace(/SITE = "[^"]*"/, `SITE = "http://127.0.0.1:${port}/files"`);
writeFileSync(join(out, "nb.py"), nb);
execFileSync("uvx", ["marimo", "export", "html-wasm", "nb.py", "-o", "site", "--mode", "run", "--show-code"], { cwd: out, stdio: "ignore" });
const wheels = join(out, "wheels");
mkdirSync(join(out, "site/files"));
execFileSync("uv", ["build", "-q", "--wheel", "-o", wheels, repo]);
cpSync(join(wheels, readdirSync(wheels).find((f) => f.endsWith(".whl"))), join(out, "site/files/pymcwasm-0.1.0-py3-none-any.whl"));
cpSync(join(repo, "node_modules/tapewasm"), join(out, "site/files/tapewasm"), { recursive: true });

const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".wasm": "application/wasm", ".whl": "application/zip" };
const server = createServer((req, res) => {
  let path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (path.endsWith("/")) path += "index.html";
  try {
    const body = readFileSync(join(out, "site", path));
    res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" }).end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(port);

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/`);
// Done once all three models have sampled, or once a cell has raised.
const finished = await page.waitForFunction(() => {
  const t = document.body.innerText;
  return /Traceback/.test(t) || (t.match(/took \d+ seconds/g) ?? []).length >= 3;
}, null, { timeout: Number(process.env.TIMEOUT_MS ?? 900_000), polling: 2000 }).then(() => true, () => false);
const text = await page.evaluate(() => document.body.innerText);
await browser.close();
server.close();
rmSync(out, { recursive: true, force: true });

console.log(text.split("\n").filter((l) => /took \d+ seconds|rhat|divergen/.test(l)).join("\n"));
const raised = /Traceback/.test(text);
if (raised) console.log(text.slice(text.indexOf("Traceback"), text.indexOf("Traceback") + 1500));
if (!finished) console.log("timed out before the three models sampled\n" + text.slice(-800));
process.exit(!finished || raised ? 1 : 0);
