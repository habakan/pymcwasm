// A small neural decoder, z (4D) -> Dense -> sigmoid -> Dense -> sigmoid, written in
// PyMC (model.py), built ahead of time by pymcwasm-build, and fit in this tab by
// tapewasm's advi(). loadDecoder/trainDecoder run in worker.js; the rest is the page's too.

import init, { AotSampler, setAotExports, sharedMemory } from "../../vendor/index.js";

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

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

/** Splits one flat `advi()` parameter vector into the shapes this decoder
 * actually is, in model.py's order: z, w1, b1, w2, b2. */
export function decomposeMu(vec, { N, L, D_out, H }) {
  let q = 0;
  const z = [];
  for (let i = 0; i < N; i++) { z.push(Array.from(vec.subarray(q, q + L))); q += L; }
  const W1 = Array.from({ length: L }, () => []);
  for (let d = 0; d < L; d++) for (let k = 0; k < H; k++) W1[d].push(vec[q++]);
  const b1 = [];
  for (let k = 0; k < H; k++) b1.push(vec[q++]);
  const W2 = [];
  for (let k = 0; k < H; k++) { const row = []; for (let j = 0; j < D_out; j++) row.push(vec[q++]); W2.push(row); }
  const b2 = [];
  for (let j = 0; j < D_out; j++) b2.push(vec[q++]);
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
  for (let i = 0; i < N * L; i++) vec[p++] = 0.5 * gauss();
  for (let i = 0; i < L * H; i++) vec[p++] = 0.7 * gauss();
  for (let k = 0; k < H; k++) vec[p++] = 0;
  for (let i = 0; i < H * D_out; i++) vec[p++] = 0.7 * gauss();
  for (let j = 0; j < D_out; j++) vec[p++] = 0;
  return vec;
}

/** Fetches the module pymcwasm-build wrote from model.py (the pixels are in it as
 * constants) and binds it; `trainDecoder` can then fit it any number of times. */
export async function loadDecoder({ N = 32, L = 4, H = 20, D_out = 196 } = {}) {
  await start();
  const t0 = performance.now();
  const base = new URL("../../artifacts/live-decoder/", import.meta.url);
  const [meta, bytes] = await Promise.all([
    fetch(new URL("meta.json", base)).then((r) => r.json()),
    fetch(new URL("model.wasm", base)).then((r) => r.arrayBuffer()),
  ]);
  const aot = await WebAssembly.instantiate(bytes, { tapewasm: { memory: sharedMemory() }, Math: MATH });
  setAotExports(aot.instance.exports);
  const sampler = new AotSampler(meta.nParams, new Float64Array(meta.scratchInit), meta.layoutId,
    meta.paramNames);
  return {
    sampler, totalParams: meta.nParams, shape: { N, L, D_out, H },
    loadMs: performance.now() - t0, moduleBytes: bytes.byteLength, nParams: meta.nParams,
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
