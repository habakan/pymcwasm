// Samples the MMM modules bench/mmm.py built with tapewasm in V8, in the four combinations of
// re-rolled or straight-line module and the gradient-based metric estimate off or on, and
// writes bench/models/mmm/tapewasm.json for `bench/mmm.py table`.
//
//   STANWASM=../tapewasm node bench/mmm.mjs     # a tapewasm checkout's ts/ instead of npm's

import fs from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const entry = process.env.STANWASM
  ? `${process.env.STANWASM}/ts/index.js`
  : createRequire(import.meta.url).resolve("tapewasm");
const tw = await import(pathToFileURL(entry).href);
await tw.default({ module_or_path: fs.readFileSync(new URL("pkg/tapewasm_bg.wasm", pathToFileURL(entry))) });

const OUT = new URL("models/mmm/", import.meta.url);
const FITS = 10, TUNE = 750, DRAWS = 500, CHAINS = 2;
const MATH = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow, tan: Math.tan,
  asin: Math.asin, acos: Math.acos, atan: Math.atan, lgamma: () => NaN, digamma: () => NaN, phi: () => NaN,
};

const runs = [];
let nParams;
for (const reroll of ["auto", "never"]) {
  const meta = JSON.parse(fs.readFileSync(new URL(`${reroll}/meta.json`, OUT)));
  const aot = await WebAssembly.instantiate(fs.readFileSync(new URL(`${reroll}/model.wasm`, OUT)),
    { tapewasm: { memory: tw.sharedMemory() }, Math: MATH });
  tw.setAotExports(aot.instance.exports);
  const n = nParams = meta.nParams;
  for (const gradBased of [false, true]) {
    const fits = [];
    for (let f = 0; f < FITS; f++) {
      const fit = { evals: 0, divergences: 0, step_size: 0, draws: [] };
      const t = performance.now();
      for (let c = 0; c < CHAINS; c++) {
        const s = new tw.AotSampler(n, new Float64Array(meta.scratchInit), meta.layoutId, meta.paramNames);
        s.setTargetAccept(0.9);
        s.setGradBasedEstimate(gradBased);
        const r = s.sampleWithStats(new Float64Array(meta.initialPoint), TUNE, DRAWS, BigInt(1000 * (f + 1) + c), c);
        for (let i = 0; i < TUNE + DRAWS; i++) {
          fit.evals += r.numSteps[i];
          if (i >= TUNE) { fit.divergences += r.diverging[i]; fit.step_size += r.stepSize[i] / (DRAWS * CHAINS); }
        }
        fit.draws.push(...r.draws.slice(TUNE * n));
        s.free();
      }
      fit.seconds = (performance.now() - t) / 1000;
      fits.push(fit);
    }
    runs.push({ reroll, gradBased, fits });
    console.log(reroll, gradBased, fits.map((f) => f.seconds.toFixed(2)).join(" "));
  }
}
const versions = { node: process.version, tapewasm: tw.tapewasmVersion() };
fs.writeFileSync(new URL("tapewasm.json", OUT), JSON.stringify({ versions, nParams, runs }));
