// Two small canvas plots, no dependency. A histogram says whether the posterior
// is a posterior; a trace says whether the chain moved.

const INK = "#1f2933";
const BAR = "#8fb6d1";
const REF = "#c0392b";
const GRID = "#d8dde2";

function fit(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px ui-monospace, monospace";
  return ctx;
}

export function histogram(canvas, values, { refMean, w = 260, h = 96 } = {}) {
  const ctx = fit(canvas, w, h);
  const pad = { l: 4, r: 4, t: 6, b: 16 };
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(hi > lo)) { hi = lo + 1; lo -= 1; }

  const bins = 34;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const k = Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins));
    counts[k] += 1;
  }
  const peak = Math.max(...counts);
  const x = (v) => pad.l + ((v - lo) / (hi - lo)) * (w - pad.l - pad.r);
  const bw = (w - pad.l - pad.r) / bins;

  ctx.fillStyle = BAR;
  for (let k = 0; k < bins; k++) {
    const bh = (counts[k] / peak) * (h - pad.t - pad.b);
    ctx.fillRect(pad.l + k * bw, h - pad.b - bh, Math.max(bw - 1, 1), bh);
  }

  if (Number.isFinite(refMean)) {
    // Where nutpie put the mean. Drawn over the bars so a shifted posterior is
    // obvious rather than something to read off a table.
    ctx.strokeStyle = REF;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    ctx.moveTo(x(refMean), pad.t);
    ctx.lineTo(x(refMean), h - pad.b);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = GRID;
  ctx.beginPath();
  ctx.moveTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();

  ctx.fillStyle = INK;
  ctx.fillText(lo.toFixed(2), pad.l, h - 4);
  const hiLabel = hi.toFixed(2);
  ctx.fillText(hiLabel, w - pad.r - ctx.measureText(hiLabel).width, h - 4);
}

export function trace(canvas, values, { w = 160, h = 96 } = {}) {
  const ctx = fit(canvas, w, h);
  const pad = { l: 3, r: 3, t: 6, b: 16 };
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(hi > lo)) { hi = lo + 1; lo -= 1; }

  // One pixel column per step once there are more draws than pixels.
  const span = w - pad.l - pad.r;
  const step = Math.max(1, Math.ceil(values.length / span));
  ctx.strokeStyle = "#5b7c95";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0, k = 0; i < values.length; i += step, k++) {
    const px = pad.l + (k / Math.ceil(values.length / step)) * span;
    const py = h - pad.b - ((values[i] - lo) / (hi - lo)) * (h - pad.t - pad.b);
    if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  ctx.fillStyle = INK;
  ctx.fillText(`${values.length} draws`, pad.l, h - 4);
}
