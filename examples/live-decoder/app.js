// A small neural decoder, z (4D) -> Dense -> sigmoid -> Dense -> sigmoid, written
// as a tape by hand and fit in this tab by tapewasm's compileTape() + advi().
// compileDecoder/trainDecoder run in worker.js; the rest is the page's too.

import init, { AotSampler, compileTape, setAotExports, sharedMemory } from "../../vendor/index.js";

let ready;
export function start() {
  ready ??= init({ module_or_path: new URL("../../vendor/pkg/tapewasm_bg.wasm", import.meta.url) });
  return ready;
}

const MATH = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
  tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  lgamma: () => NaN, digamma: () => NaN, phi: () => NaN,
};

function makeRng(seed) {
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  const gauss = () => {
    const u1 = Math.max(rnd(), 1e-12), u2 = rnd();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  return { rnd, gauss };
}

/** 2x2 average pooling: at 28x28 the same iteration budget converges slower and
 * to a worse fit. */
export function downsample14(px28) {
  const out = new Float64Array(14 * 14);
  for (let r = 0; r < 14; r++) {
    for (let c = 0; c < 14; c++) {
      let s = 0;
      for (let dr = 0; dr < 2; dr++) for (let dc = 0; dc < 2; dc++) s += px28[(r * 2 + dr) * 28 + (c * 2 + dc)];
      out[r * 14 + c] = s / 4;
    }
  }
  return out;
}

/** sigmoid(x) as `1/(1+exp(-x))`: the tape has no sigmoid op, and PyMC's lowering
 * decomposes `pm.math.sigmoid` the same way. */
function sigmoidNode(op, x) {
  const negx = op(`neg ${x}`);
  const e = op(`exp ${negx}`);
  const den = op(`add_c ${e} 1.0`);
  return op(`rdiv_c ${den} 1.0`);
}

export function buildDecoderTape({ N, L, D_out, H, sigma = 0.05, pixel, seed = 1 }) {
  const lines = [];
  let next = 0;
  const op = (text) => { lines.push(text); return next++; };
  const nf = (v) => (Number.isFinite(v) ? v.toFixed(8) : "0.0");
  const { gauss } = makeRng(seed);

  const totalParams = L * H + H + H * D_out + D_out + N * L;
  lines.push(`n_params ${totalParams}`);

  const W1 = Array.from({ length: L }, () => []);
  for (let d = 0; d < L; d++) for (let k = 0; k < H; k++) W1[d].push(op(`new_var ${nf(0.7 * gauss())}`));
  const b1 = [];
  for (let k = 0; k < H; k++) b1.push(op(`new_var ${nf(0)}`));
  const W2 = [];
  for (let k = 0; k < H; k++) {
    const row = [];
    for (let j = 0; j < D_out; j++) row.push(op(`new_var ${nf(0.7 * gauss())}`));
    W2.push(row);
  }
  const b2 = [];
  for (let j = 0; j < D_out; j++) b2.push(op(`new_var ${nf(0)}`));
  const z = [];
  for (let i = 0; i < N; i++) z.push(Array.from({ length: L }, () => op(`new_var ${nf(0.5 * gauss())}`)));

  let acc = null;
  const addTerm = (t) => { acc = acc === null ? t : op(`add ${acc} ${t}`); };

  // N(0,1) priors throughout, matching the offline recipe this mirrors.
  for (let d = 0; d < L; d++) for (let k = 0; k < H; k++) addTerm(op(`mul_c ${op(`mul ${W1[d][k]} ${W1[d][k]}`)} -0.5`));
  for (let k = 0; k < H; k++) addTerm(op(`mul_c ${op(`mul ${b1[k]} ${b1[k]}`)} -0.5`));
  for (let k = 0; k < H; k++) for (let j = 0; j < D_out; j++) addTerm(op(`mul_c ${op(`mul ${W2[k][j]} ${W2[k][j]}`)} -0.5`));
  for (let j = 0; j < D_out; j++) addTerm(op(`mul_c ${op(`mul ${b2[j]} ${b2[j]}`)} -0.5`));
  for (let i = 0; i < N; i++) for (let d = 0; d < L; d++) addTerm(op(`mul_c ${op(`mul ${z[i][d]} ${z[i][d]}`)} -0.5`));

  const invSigma2 = -0.5 / (sigma * sigma);
  const hidden = [];
  for (let i = 0; i < N; i++) {
    const row = [];
    for (let k = 0; k < H; k++) {
      let lin = op(`mul ${z[i][0]} ${W1[0][k]}`);
      for (let d = 1; d < L; d++) lin = op(`add ${lin} ${op(`mul ${z[i][d]} ${W1[d][k]}`)}`);
      const pre = op(`add ${lin} ${b1[k]}`);
      row.push(sigmoidNode(op, pre));
    }
    hidden.push(row);
  }
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < D_out; j++) {
      let s = op(`mul ${hidden[i][0]} ${W2[0][j]}`);
      for (let k = 1; k < H; k++) s = op(`add ${s} ${op(`mul ${hidden[i][k]} ${W2[k][j]}`)}`);
      const pre = op(`add ${s} ${b2[j]}`);
      const pred = sigmoidNode(op, pre);
      const diff = op(`rsub_c ${pred} ${nf(pixel(i, j))}`);
      addTerm(op(`mul_c ${op(`mul ${diff} ${diff}`)} ${nf(invSigma2)}`));
    }
  }

  lines.push(`root ${acc}`);
  return { text: lines.join("\n"), totalParams };
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

/** Splits one flat `advi()` parameter vector into the shapes this decoder
 * actually is. */
export function decomposeMu(vec, { N, L, D_out, H }) {
  let q = 0;
  const W1 = Array.from({ length: L }, () => []);
  for (let d = 0; d < L; d++) for (let k = 0; k < H; k++) W1[d].push(vec[q++]);
  const b1 = [];
  for (let k = 0; k < H; k++) b1.push(vec[q++]);
  const W2 = [];
  for (let k = 0; k < H; k++) { const row = []; for (let j = 0; j < D_out; j++) row.push(vec[q++]); W2.push(row); }
  const b2 = [];
  for (let j = 0; j < D_out; j++) b2.push(vec[q++]);
  const z = [];
  for (let i = 0; i < N; i++) { z.push(Array.from(vec.subarray(q, q + L))); q += L; }
  return { W1, b1, W2, b2, z, L, H, D_out };
}

export function decodeWith({ W1, b1, W2, b2, L, H, D_out }, z) {
  const hidden = new Float64Array(H);
  for (let k = 0; k < H; k++) {
    let s = b1[k];
    for (let d = 0; d < L; d++) s += z[d] * W1[d][k];
    hidden[k] = sigmoid(s);
  }
  const row = new Float64Array(D_out);
  for (let j = 0; j < D_out; j++) {
    let s = b2[j];
    for (let k = 0; k < H; k++) s += hidden[k] * W2[k][j];
    row[j] = sigmoid(s);
  }
  return row;
}

function initVector({ N, L, D_out, H, totalParams, seed }) {
  const { gauss } = makeRng(seed);
  const vec = new Float64Array(totalParams);
  let p = 0;
  for (let i = 0; i < L * H; i++) vec[p++] = 0.7 * gauss();
  for (let k = 0; k < H; k++) vec[p++] = 0;
  for (let i = 0; i < H * D_out; i++) vec[p++] = 0.7 * gauss();
  for (let j = 0; j < D_out; j++) vec[p++] = 0;
  for (let i = 0; i < N * L; i++) vec[p++] = 0.5 * gauss();
  return vec;
}

/** Writes the tape (the pixels are baked in as constants) and compiles it to a
 * wasm module; `trainDecoder` can then fit it any number of times. */
export async function compileDecoder({ pixels, gridSize, L = 4, H = 20, seed = 1 }) {
  await start();
  const N = pixels.length, D_out = gridSize * gridSize;
  const { text, totalParams } = buildDecoderTape({
    // At sd 0.15 the N(0,1) priors outweigh 32 images and the fit blurs (2D latent: MSE ~0.021 vs ~0.006).
    N, L, D_out, H, sigma: 0.05, pixel: (i, j) => pixels[i][j], seed,
  });

  const t0 = performance.now();
  const built = compileTape(text);
  const compileMs = performance.now() - t0;

  const aot = await WebAssembly.instantiate(built.wasm, {
    tapewasm: { memory: sharedMemory() },
    Math: MATH,
  });
  setAotExports(aot.instance.exports);
  const sampler = new AotSampler(built.nParams, built.scratchInit, built.layoutId, []);

  return {
    sampler, totalParams, shape: { N, L, D_out, H },
    compileMs,
    moduleBytes: built.wasm.length,
    nParams: built.nParams,
    tapeLines: text.split("\n").length,
  };
}

/** One `advi()` run; `onSnapshot(iter, mu, elbo)` fires at each snapshot as it goes. */
export function trainDecoder(compiled, {
  numIters = 4000, mcSamples = 3, learningRate = 0.02,
  seed = Date.now() & 0xffff, snapshotEvery = 60, onSnapshot,
} = {}) {
  const { sampler, totalParams, shape } = compiled;
  const initVec = initVector({ ...shape, totalParams, seed: seed + 1 });
  const t0 = performance.now();
  const r = sampler.advi(initVec, numIters, mcSamples, learningRate, BigInt(seed), snapshotEvery, onSnapshot);
  return { mu: r.mu, trainMs: performance.now() - t0 };
}
