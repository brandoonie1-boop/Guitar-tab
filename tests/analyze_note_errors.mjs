// Note-detection error analysis on GuitarSet from cached basic-pitch model outputs
// (/home/claude/work/data/gs/bpraw, made from the eval_bp_gs.py cache).
// usage: node tests/analyze_note_errors.mjs [--opts '{"onsetThresh":0.6}'] [--subset comp|solo] [--quiet]
import fs from 'node:fs';
import { outputToNotes, addPitchBends, frameTime, BP } from '../engine/basicpitch.js';
import { cleanNotes } from '../engine/notes.js';
import { hrNotes } from '../engine/hrgt.js';
import { DEFAULT_OPTS } from '../engine/pipeline.js';

const RAW = '/home/claude/work/data/gs/bpraw';
const ANN = JSON.parse(fs.readFileSync('/home/claude/work/data/gs/guitarset_annotations.json', 'utf8'));
const args = process.argv.slice(2);
const oi = args.indexOf('--opts');
const opts = { ...DEFAULT_OPTS, ...(oi >= 0 ? JSON.parse(args[oi + 1]) : {}) };
const si = args.indexOf('--subset');
const subset = si >= 0 ? args[si + 1] : null;
const quiet = args.includes('--quiet');
// --hr <dir>: notes from cached guitar transcription model activations instead (eval_hrgt_gs.mjs cache)
const hi = args.indexOf('--hr');
const hrDir = hi >= 0 ? args[hi + 1] : null;
let index = JSON.parse(fs.readFileSync(`${RAW}/index.json`, 'utf8'));
if (subset) index = index.filter(([n]) => n.endsWith(subset));
if (hrDir) index = index.filter(([n]) => fs.existsSync(`${hrDir}/${n}.f32`));
const li = args.indexOf('--limit');
if (li >= 0) index = index.slice(0, Number(args[li + 1]));

const rd = (p) => new Float32Array(fs.readFileSync(p).buffer.slice(0));

export function decode(model, o = opts) {
  const minLen = Math.max(3, Math.round((o.minNoteMs / 1000) * (BP.sr / BP.hop)));
  let notes = outputToNotes(model, { onsetThresh: o.onsetThresh, frameThresh: o.frameThresh, minNoteLen: minLen, energyTol: o.energyTol, melodiaTrick: o.melodia });
  notes = addPitchBends(model, notes);
  const nF = BP.nNotes;
  const raw = notes.map((n) => {
    let op = 0;
    for (let t = Math.max(0, n.s - 2); t <= Math.min(model.nT - 1, n.s + 2); t++) op = Math.max(op, model.onset[t * nF + (n.pitch - 21)]);
    return { t0: frameTime(n.s), t1: frameTime(n.e), pitch: n.pitch, amp: n.amp, bends: n.bends, onsetProb: op };
  });
  return cleanNotes(raw, { idPrefix: 'x:', mergeRetrigger: o.mergeRetrigger, retriggerRatio: o.retriggerRatio, retriggerAbs: o.retriggerAbs });
}

export function match(est, ref, tol = 0.05) {
  // greedy by time distance, same pitch, onset within tol (mir_eval-like)
  const pairs = [];
  for (let i = 0; i < ref.length; i++) for (let j = 0; j < est.length; j++) {
    if (est[j].pitch !== ref[i].pitch) continue;
    const d = Math.abs(est[j].t0 - ref[i].t0);
    if (d <= tol) pairs.push([d, i, j]);
  }
  pairs.sort((a, b) => a[0] - b[0]);
  const ri = new Int32Array(ref.length).fill(-1), ej = new Int32Array(est.length).fill(-1);
  for (const [, i, j] of pairs) if (ri[i] < 0 && ej[j] < 0) { ri[i] = j; ej[j] = i; }
  return { ri, ej };
}

const fnCat = {}, fpCat = {};
const bump = (o, k) => { o[k] = (o[k] || 0) + 1; };
let TP = 0, NE = 0, NR = 0, f1sum = 0;
const perFile = [];
for (const [name, nT] of index) {
  const model = { nT, note: rd(`${RAW}/${name}.note.f32`), onset: rd(`${RAW}/${name}.onset.f32`), contour: rd(`${RAW}/${name}.contour.f32`) };
  let est;
  if (hrDir) {
    const all = rd(`${hrDir}/${name}.f32`), T = all.length / 352;
    const sl = (i) => all.subarray(i * T * 88, (i + 1) * T * 88);
    est = cleanNotes(hrNotes({ T, onset: sl(0), offset: sl(1), frame: sl(2), velocity: sl(3) }, opts.hr || {}).map((n) => ({ ...n, bends: [] })), { idPrefix: 'x:', mergeRetrigger: false });
  } else est = decode(model);
  const ref = ANN[name].notes.map(([t, d, v]) => ({ t0: t, t1: t + d, pitch: Math.round(v) }));
  const { ri, ej } = match(est, ref);
  let tp = 0;
  for (let i = 0; i < ref.length; i++) {
    if (ri[i] >= 0) { tp++; continue; }
    const r = ref[i];
    // why was it missed?
    if (est.some((e) => e.pitch === r.pitch && e.t0 < r.t0 - 0.05 && e.t1 > r.t0 + 0.02)) bump(fnCat, 'held over (no new attack)');
    else if (est.some((e, j) => ej[j] < 0 && e.pitch === r.pitch && Math.abs(e.t0 - r.t0) <= 0.15)) bump(fnCat, 'late/early (50-150 ms)');
    else if (est.some((e) => Math.abs(e.pitch - r.pitch) === 12 && Math.abs(e.t0 - r.t0) <= 0.05)) bump(fnCat, 'octave');
    else if (est.some((e) => Math.abs(e.pitch - r.pitch) <= 2 && Math.abs(e.t0 - r.t0) <= 0.05)) bump(fnCat, 'semitone or tone');
    else if (r.t1 - r.t0 < 0.06) bump(fnCat, 'very short ref note');
    else bump(fnCat, 'absent');
  }
  for (let j = 0; j < est.length; j++) {
    if (ej[j] >= 0) continue;
    const e = est[j];
    if (ref.some((r) => r.pitch === e.pitch && Math.abs(r.t0 - e.t0) <= 0.15)) bump(fpCat, 'late/early (50-150 ms)');
    else if (ref.some((r) => r.pitch === e.pitch && r.t0 < e.t0 && r.t1 > e.t0)) bump(fpCat, 'split (re-attack inside a held note)');
    else if (ref.some((r) => [12, 19, 24].includes(Math.abs(r.pitch - e.pitch)) && Math.abs(r.t0 - e.t0) <= 0.08)) bump(fpCat, 'harmonic ghost (oct/12th)');
    else if (ref.some((r) => Math.abs(r.pitch - e.pitch) <= 2 && Math.abs(r.t0 - e.t0) <= 0.08)) bump(fpCat, 'wrong by 1-2 semitones');
    else if (e.t1 - e.t0 < 0.08) bump(fpCat, 'short blip');
    else bump(fpCat, 'other');
  }
  const p = tp / Math.max(1, est.length), r2 = tp / Math.max(1, ref.length);
  const f = (2 * p * r2) / Math.max(1e-9, p + r2);
  f1sum += f;
  TP += tp; NE += est.length; NR += ref.length;
  perFile.push([name, f, p, r2]);
}
const P = TP / NE, R = TP / NR;
if (!quiet) {
  console.log('missed notes:', JSON.stringify(Object.entries(fnCat).sort((a, b) => b[1] - a[1])));
  console.log('extra notes: ', JSON.stringify(Object.entries(fpCat).sort((a, b) => b[1] - a[1])));
}
console.log(`files ${index.length}${subset ? ' (' + subset + ')' : ''}: P ${(100 * P).toFixed(1)} R ${(100 * R).toFixed(1)} F1 ${(200 * P * R / (P + R)).toFixed(1)} (mean per-file F1 ${(100 * f1sum / index.length).toFixed(1)})  refs ${NR} est ${NE}`);
