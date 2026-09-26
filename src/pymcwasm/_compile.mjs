// Tape on stdin, module to argv[2], what a host needs as JSON on stdout.
// `tapewasm` resolves from the working directory, so a project's own pinned copy is the one used.
import fs from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(pathToFileURL(process.cwd() + "/").href);
const entry = require.resolve("tapewasm");
const tw = await import(pathToFileURL(entry).href);
await tw.default({ module_or_path: fs.readFileSync(new URL("pkg/tapewasm_bg.wasm", pathToFileURL(entry))) });

const built = tw.compileTape(fs.readFileSync(0, "utf8"), process.argv[3] ?? "auto");
fs.writeFileSync(process.argv[2], built.wasm);
console.log(JSON.stringify({
  nParams: built.nParams,
  layoutId: built.layoutId,
  scratchInit: Array.from(built.scratchInit),
  nOutputs: built.nOutputs,
  tapewasm: tw.tapewasmVersion(),
}));
