// Everything a page has to do with a precompiled model. Nothing here knows what
// PyMC is: it loads a module, the buffer sizes recorded beside it, and samples.

import init, { AotSampler, setAotExports, sharedMemory } from "../../vendor/index.js";

export const MODELS = [
  "linear_regression", "logistic", "eight_schools",
  "varying_intercepts", "matrix_regression", "student_t", "lkj_mvnormal",
];

let ready;
export function start() {
  ready ??= init({ module_or_path: new URL("../../vendor/pkg/tapewasm_bg.wasm", import.meta.url) });
  return ready;
}

// Only the ones a tape reaches are imported, but a module asks for what it
// asks for, so all of them are here.
const MATH = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
  tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  lgamma: () => NaN, digamma: () => NaN, phi: () => NaN,
};

export async function sample(name, { warmup = 1000, draws = 1000, seed = 42, chains = 4 } = {}) {
  await start();
  const meta = await (await fetch(`../../artifacts/${name}/meta.json`)).json();
  const bytes = await (await fetch(`../../artifacts/${name}/model.wasm`)).arrayBuffer();

  const aot = await WebAssembly.instantiate(bytes, {
    tapewasm: { memory: sharedMemory() },
    Math: MATH,
  });
  setAotExports(aot.instance.exports);

  const n = meta.nParams;
  const runs = [];
  const t0 = performance.now();
  for (let c = 0; c < chains; c++) {
    // One sampler per chain, seeded apart, as the Python side does.
    const sampler = new AotSampler(
      meta.nParams, new Float64Array(meta.scratchInit), meta.layoutId, meta.paramNames,
    );
    const args = [new Float64Array(meta.initialPoint), warmup, draws, BigInt(seed + c)];
    const r = sampler.sampleWithStats(...args, c);
    // `draws` may be a view into wasm memory that the next chain overwrites.
    const post = r.draws.slice(warmup * n);
    runs.push({
      draws: post,
      diverging: r.diverging.slice(warmup),
      // The pointwise log-likelihood, a row per draw, which is what PSIS-LOO
      // reads. The module carries the terms only when it was compiled with
      // them, and a tapewasm before 0.4.0 cannot read them back.
      logLik: meta.logLik?.length && typeof sampler.evaluate === "function"
        ? pointwise(sampler, post, n, draws) : null,
    });
    sampler.free();
  }
  const ms = performance.now() - t0;

  const mean = new Array(n).fill(0);
  for (const { draws: post } of runs) {
    for (let i = 0; i < draws; i++) {
      for (let k = 0; k < n; k++) mean[k] += post[i * n + k] / (draws * chains);
    }
  }
  return {
    meta, mean, chains: runs.map((r) => r.draws),
    diverging: runs.map((r) => r.diverging),
    logLikelihood: runs[0].logLik ? runs.map((r) => r.logLik) : null,
    nDraws: draws, nChains: chains, ms, moduleBytes: bytes.byteLength,
  };
}

/** One `evaluate` per draw, laid out `nDraws * nObs` row-major per chain. */
function pointwise(sampler, post, n, draws) {
  let out = null;
  for (let i = 0; i < draws; i++) {
    const row = sampler.evaluate(post.subarray(i * n, (i + 1) * n));
    out ??= new Float64Array(draws * row.length);
    out.set(row, i * row.length);
  }
  return out;
}

/** How far a parameter's mean may sit from the reference, in its sds.
 * eight_schools is written centred: in the funnel the two samplers disagree by
 * more than anywhere else, and the reference's own error is larger there too. */
export const tolerance = (name) => (name === "eight_schools" ? 0.5 : 0.3);

export async function compare(name, options) {
  const { meta, mean, ms, moduleBytes } = await sample(name, options);
  const reference = await (await fetch(`../../artifacts/${name}/reference.json`)).json();
  const rows = meta.paramNames.map((label, k) => {
    const ref = reference[label];
    return {
      label, ours: mean[k], theirs: ref.mean,
      gap: Math.abs(mean[k] - ref.mean) / Math.max(ref.sd, 1e-12),
    };
  });
  return {
    name, ms, moduleBytes, rows, meta,
    worst: Math.max(...rows.map((r) => r.gap)),
    tolerance: tolerance(name),
  };
}
