// Static files for the page: the artifacts, this directory, and the published
// `stanwasm` package straight out of node_modules. No bundler — what the page
// loads is what npm published.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";

const repo = dirname(fileURLToPath(import.meta.url));
const types = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".json": "application/json",
};
const port = Number(process.env.PORT ?? 8140);

// `/vendor` is the published stanwasm package, `/pkg` the Python sources a
// Pyodide page installs, and everything else is a file in the repository.
//
// `STANWASM=<checkout>` serves that checkout's `ts/` instead, for trying a
// change before it is published.
const vendor = process.env.STANWASM
  ? resolve(process.env.STANWASM, "ts")
  : resolve(repo, "node_modules/stanwasm");

function locate(path) {
  if (path.startsWith("/vendor/")) {
    return resolve(vendor, "." + path.slice("/vendor".length));
  }
  if (path.startsWith("/pkg/")) {
    return resolve(repo, "src", "." + path.slice("/pkg".length));
  }
  return resolve(repo, "." + path);
}

export const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://x").pathname;
  // A directory URL means its index, and bare "/" means the first example.
  const path = url === "/"
    ? "/index.html"
    : url.endsWith("/") ? `${url}index.html` : url;
  try {
    const body = await readFile(locate(path));
    res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
});

server.listen(port, "127.0.0.1", () => console.log(`http://127.0.0.1:${port}/`));
