// Shared by the note evaluations (eval_notes.mjs, eval_band.mjs): simulated amps, the note
// model's decoding as the pipeline calls it, and mir_eval-style onset matching.
import fs from 'node:fs';
import path from 'node:path';
import { parseChain, loadParts } from '../engine/demucs.js';
import { hrModel, hrNotes } from '../engine/hrgt.js';
import { cleanNotes } from '../engine/notes.js';
import { DEFAULT_OPTS } from '../engine/pipeline.js';

// biquads (RBJ cookbook), applied in place
function biquad(x, sr, type, f0, q, gainDb = 0) {
  const A = Math.pow(10, gainDb / 40), w = (2 * Math.PI * f0) / sr, c = Math.cos(w), s = Math.sin(w), al = s / (2 * q);
  let b0, b1, b2, a0, a1, a2;
  if (type === 'lp') { b0 = (1 - c) / 2; b1 = 1 - c; b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
  else if (type === 'hp') { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
  else if (type === 'peak') { b0 = 1 + al * A; b1 = -2 * c; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * c; a2 = 1 - al / A; }
  else if (type === 'hshelf') {
    const r = 2 * Math.sqrt(A) * al;
    b0 = A * ((A + 1) + (A - 1) * c + r); b1 = -2 * A * ((A - 1) + (A + 1) * c); b2 = A * ((A + 1) + (A - 1) * c - r);
    a0 = (A + 1) - (A - 1) * c + r; a1 = 2 * ((A - 1) - (A + 1) * c); a2 = (A + 1) - (A - 1) * c - r;
  } else throw new Error(type);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i], y = (b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1; x1 = v; y2 = y1; y1 = y; x[i] = y;
  }
  return x;
}
export const peakOf = (x) => { let p = 1e-9; for (const v of x) p = Math.max(p, Math.abs(v)); return p; };
export const rmsOf = (x) => { let s = 0; for (const v of x) s += v * v; return Math.sqrt(s / Math.max(1, x.length)); };

/**
 * fx  tanh:G   soft clipper and a 4.5 kHz low-pass (the "simulated amp" of eval_gs_pipeline.mjs)
 *     amp:G    an amp-like chain: low cut, presence boost before a slightly asymmetric clipper,
 *              mid scoop, a speaker-like band-pass with a 2.5 kHz bump, then level back to -14 dB
 */
export function applyFx(x, sr, fx) {
  if (!fx) return x;
  const [kind, gs] = fx.split(':');
  const g = Number(gs);
  const y = Float32Array.from(x);
  if (kind === 'tanh') {
    const pk = peakOf(y), a = Math.exp((-2 * Math.PI * 4500) / sr);
    let lp = 0;
    for (let i = 0; i < y.length; i++) { const d = Math.tanh((g * y[i]) / pk) / Math.tanh(g); lp = (1 - a) * d + a * lp; y[i] = 0.5 * lp; }
    return y;
  }
  if (kind === 'amp') {
    biquad(y, sr, 'hp', 120, 0.7);
    biquad(y, sr, 'hshelf', 900, 0.7, 6);
    const pk = peakOf(y);
    for (let i = 0; i < y.length; i++) { const v = (g * y[i]) / pk; y[i] = v >= 0 ? Math.tanh(v) : Math.tanh(0.8 * v) / 0.8; }
    biquad(y, sr, 'peak', 650, 0.8, -5);
    biquad(y, sr, 'hp', 85, 0.8);
    biquad(y, sr, 'peak', 2500, 1.2, 4);
    biquad(y, sr, 'lp', 4800, 0.9);
    biquad(y, sr, 'lp', 6000, 0.6);
    const k = Math.pow(10, -14 / 20) / Math.max(1e-9, rmsOf(y));
    for (let i = 0; i < y.length; i++) y[i] *= k;
    return y;
  }
  throw new Error(`unknown fx ${fx}`);
}

/** The note model (hrgt.chain + hrgt.weights.bin in `dir`) on onnxruntime-node. */
export async function loadNoteModel(ort, dir, threads = Number(process.env.THREADS || 4)) {
  return hrModel(await loadParts(ort, parseChain(fs.readFileSync(path.join(dir, 'hrgt.chain'))), new Uint8Array(fs.readFileSync(path.join(dir, 'hrgt.weights.bin'))), { intraOpNumThreads: threads }));
}

/** Activations cached as planar float32 (onset, offset, frame, velocity). */
export function readActs(p) {
  const all = new Float32Array(fs.readFileSync(p).buffer.slice(0));
  const T = all.length / (88 * 4);
  const sl = (i) => all.subarray(i * T * 88, (i + 1) * T * 88);
  return { T, onset: sl(0), offset: sl(1), frame: sl(2), velocity: sl(3) };
}
export function writeActs(p, act) {
  const all = new Float32Array(act.T * 88 * 4);
  ['onset', 'offset', 'frame', 'velocity'].forEach((k, i) => all.set(act[k], i * act.T * 88));
  fs.writeFileSync(p, Buffer.from(all.buffer));
}

/** Notes from the model's outputs, as the pipeline decodes and cleans them. */
export function decodeNotes(act, opts = {}) {
  return cleanNotes(hrNotes(act, { onsetThreshold: opts.onset ?? DEFAULT_OPTS.hrOnset, offsetThreshold: DEFAULT_OPTS.hrOffset, frameThreshold: DEFAULT_OPTS.hrFrame })
    .map((n) => ({ ...n, bends: [], hr: true })), { idPrefix: 'x:', mergeFragments: false, mergeRetrigger: false });
}

/** mir_eval-style onset matching: same pitch, onsets within 50 ms, maximum matching */
export function matchNotes(est, ref, tol = 0.05) {
  const ri = new Int32Array(ref.length).fill(-1), ej = new Int32Array(est.length).fill(-1);
  const byPitch = new Map();
  ref.forEach((r, i) => { if (!byPitch.has(r.pitch)) byPitch.set(r.pitch, { r: [], e: [] }); byPitch.get(r.pitch).r.push(i); });
  est.forEach((e, j) => { const g = byPitch.get(e.pitch); if (g) g.e.push(j); });
  for (const { r, e } of byPitch.values()) {
    const adj = r.map((i) => e.filter((j) => Math.abs(est[j].t0 - ref[i].t0) <= tol));
    const aug = (k, seen) => {
      for (const j of adj[k]) {
        if (seen.has(j)) continue;
        seen.add(j);
        if (ej[j] < 0 || aug(r.indexOf(ej[j]), seen)) { ej[j] = r[k]; ri[r[k]] = j; return true; }
      }
      return false;
    };
    for (let k = 0; k < r.length; k++) aug(k, new Set());
  }
  return { ri, ej };
}

export const f1 = (tp, ne, nr) => { const P = tp / Math.max(1, ne), R = tp / Math.max(1, nr); return { P, R, F: P + R > 0 ? (2 * P * R) / (P + R) : 0 }; };
