// Everything a page has to do with a precompiled model. Nothing here knows what
// PyMC is: it loads a module, the buffer sizes recorded beside it, and samples.

import init, { AotSampler, setAotExports, sharedMemory } from "../../vendor/index.js";

export const MODELS = [
  "linear_regression", "logistic", "eight_schools",
  "varying_intercepts", "matrix_regression", "student_t", "lkj_mvnormal",
];

let ready;
export function start() {
  ready ??= init({ module_or_path: new URL("../../vendor/pkg/stanwasm_bg.wasm", import.meta.url) });
  return ready;
}

// Only the ones a tape reaches are imported, but a module asks for what it
// asks for, so all of them are here.
const MATH = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
  tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  lgamma: () => NaN, digamma: () => NaN, phi: () => NaN,
};

export async function sample(name, { warmup = 1000, draws = 1000, seed = 42 } = {}) {
  await start();
  const meta = await (await fetch(`../../artifacts/${name}/meta.json`)).json();
  const bytes = await (await fetch(`../../artifacts/${name}/model.wasm`)).arrayBuffer();

  const aot = await WebAssembly.instantiate(bytes, {
    stan: { memory: sharedMemory() },
    Math: MATH,
  });
  setAotExports(aot.instance.exports);

  const sampler = new AotSampler(
    meta.nParams, new Float64Array(meta.scratchInit), meta.layoutId, meta.paramNames,
  );
  const t0 = performance.now();
  const flat = sampler.sample(new Float64Array(meta.initialPoint), warmup, draws, BigInt(seed));
  const ms = performance.now() - t0;

  const n = meta.nParams;
  const post = flat.subarray(warmup * n);
  const mean = new Array(n).fill(0);
  for (let i = 0; i < draws; i++) {
    for (let k = 0; k < n; k++) mean[k] += post[i * n + k] / draws;
  }
  // `post` is a view into wasm memory; the caller keeps it, so hand over a copy.
  return { meta, mean, draws: post.slice(), nDraws: draws, ms, moduleBytes: bytes.byteLength };
}

export async function compare(name, options) {
  const { meta, mean, draws, nDraws, ms, moduleBytes } = await sample(name, options);
  const reference = await (await fetch(`../../artifacts/${name}/reference.json`)).json();
  const rows = meta.paramNames.map((label, k) => {
    const ref = reference[label];
    return {
      label, ours: mean[k], theirs: ref.mean,
      gap: Math.abs(mean[k] - ref.mean) / Math.max(ref.sd, 1e-12),
    };
  });
  return {
    name, ms, moduleBytes, rows, draws, nDraws, meta,
    worst: Math.max(...rows.map((r) => r.gap)),
  };
}
