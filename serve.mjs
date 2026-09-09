// Static files only. A page that samples a precompiled model needs no more than
// this, which is the point.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const types = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".json": "application/json",
};
const port = Number(process.env.PORT ?? 8140);

export const server = createServer(async (req, res) => {
  const path = new URL(req.url, "http://x").pathname;
  const file = path === "/" ? "/demo/index.html" : path;
  // `stanwasm` is served straight out of node_modules: no bundler, so what the
  // page loads is the published package.
  const root = file.startsWith("/vendor/")
    ? resolve(here, "node_modules/stanwasm")
    : here;
  const rel = file.startsWith("/vendor/") ? file.slice("/vendor".length) : file;
  try {
    const body = await readFile(resolve(root, "." + rel));
    res.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`http://127.0.0.1:${port}/`);
});
