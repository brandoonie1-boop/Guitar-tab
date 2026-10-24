// Fits engine/tone.js's TONE_RULES (which sound the tab player starts with) on labelled guitar
// recordings: a small softmax model over the four tone features for acoustic / clean / distorted,
// and, among distorted guitars, a line between crunch and high gain ("lead").
//   node tools/fit_tone.mjs <label>:<wav file or folder> ... [--max n] [--windows n]
// label: acoustic | clean | distorted. Prints the cross-validated accuracy and the rules to paste
// into engine/tone.js.
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { toneFeatures } from '../engine/tone.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? Number(args[i + 1]) : d; };
const MAX = opt('--max', 1e9), WINDOWS = opt('--windows', 2);
const CLASSES = ['acoustic', 'clean', 'distorted'];
const KEYS = ['hf', 'flat', 'crest', 'decay'];

const sources = args.filter((a, i) => /^[a-z]+:/.test(a) && !/^--/.test(args[i - 1] || ''));
const rows = []; // { x: [..], y: class index, file }
for (const src of sources) {
  const label = src.slice(0, src.indexOf(':')), p = src.slice(src.indexOf(':') + 1);
  const y = CLASSES.indexOf(label);
  if (y < 0) throw new Error(`unknown label ${label}`);
  let files = fs.statSync(p).isDirectory() ? fs.readdirSync(p).filter((f) => /\.wav$/i.test(f)).sort().map((f) => path.join(p, f)) : [p];
  if (files.length > MAX) files = Array.from({ length: MAX }, (_, i) => files[Math.floor((i * files.length) / MAX)]);
  for (const f of files) {
    const w = decodeWav(fs.readFileSync(f));
    let x = w.channels[0];
    if (w.channels.length > 1) { x = new Float32Array(x.length); for (const c of w.channels) for (let i = 0; i < x.length; i++) x[i] += c[i] / w.channels.length; }
    if (w.sr !== 44100) x = resample(x, w.sr, 44100);
    // up to WINDOWS stretches of 30 s, spread over the recording
    const L = 30 * 44100, n = Math.max(1, Math.min(WINDOWS, Math.floor(x.length / L)));
    for (let k = 0; k < n; k++) {
      const o = n === 1 ? 0 : Math.floor((k * (x.length - L)) / (n - 1));
      const ft = toneFeatures(x.subarray(o, o + L), 44100);
      if (ft.frames < 20) continue;
      rows.push({ x: KEYS.map((key) => ft[key]), y, file: f, label });
    }
  }
  console.error(`${src}: ${files.length} files`);
}

const mean = KEYS.map((_, j) => rows.reduce((s, r) => s + r.x[j], 0) / rows.length);
const sd = KEYS.map((_, j) => Math.sqrt(rows.reduce((s, r) => s + (r.x[j] - mean[j]) ** 2, 0) / rows.length) || 1);
const z = (x) => x.map((v, j) => (v - mean[j]) / sd[j]);

// softmax regression, every class weighted equally, L2 on the weights
function fit(data) {
  const K = CLASSES.length, D = KEYS.length;
  const W = Array.from({ length: K }, () => new Float64Array(D + 1));
  const count = CLASSES.map((_, k) => data.filter((r) => r.y === k).length || 1);
  for (let it = 0; it < 3000; it++) {
    const G = Array.from({ length: K }, () => new Float64Array(D + 1));
    for (const r of data) {
      const v = z(r.x);
      const s = W.map((w) => w[D] + v.reduce((a, b, j) => a + w[j] * b, 0));
      const m = Math.max(...s), e = s.map((q) => Math.exp(q - m)), t = e.reduce((a, b) => a + b, 0);
      const wt = 1 / count[r.y];
      for (let k = 0; k < K; k++) {
        const g = (e[k] / t - (k === r.y ? 1 : 0)) * wt;
        for (let j = 0; j < D; j++) G[k][j] += g * v[j];
        G[k][D] += g;
      }
    }
    for (let k = 0; k < K; k++) for (let j = 0; j <= D; j++) W[k][j] -= 0.5 * (G[k][j] / K + (j < D ? 0.01 * W[k][j] : 0));
  }
  return W;
}
const predict = (W, x) => { const v = z(x); let best = 0, bs = -Infinity; W.forEach((w, k) => { const s = w[KEYS.length] + v.reduce((a, b, j) => a + w[j] * b, 0); if (s > bs) { bs = s; best = k; } }); return best; };

// 5-fold cross-validation, all windows of a file in the same fold
const files = [...new Set(rows.map((r) => r.file))];
const foldOf = new Map(files.map((f, i) => [f, i % 5]));
const conf = CLASSES.map(() => CLASSES.map(() => 0));
for (let f = 0; f < 5; f++) {
  const W = fit(rows.filter((r) => foldOf.get(r.file) !== f));
  for (const r of rows.filter((q) => foldOf.get(q.file) === f)) conf[r.y][predict(W, r.x)]++;
}
console.log('cross-validated (rows: true class, columns: guessed)');
console.log('            ' + CLASSES.map((c) => c.padStart(10)).join(''));
CLASSES.forEach((c, k) => console.log(c.padEnd(12) + conf[k].map((v) => String(v).padStart(10)).join('') + `   ${((100 * conf[k][k]) / Math.max(1, conf[k].reduce((a, b) => a + b, 0))).toFixed(0)}%`));
for (const k of KEYS) {
  const j = KEYS.indexOf(k);
  console.log(k.padEnd(6), CLASSES.map((c, y) => { const v = rows.filter((r) => r.y === y).map((r) => r.x[j]).sort((a, b) => a - b); return `${c} ${v.length ? v[v.length >> 1].toFixed(1) : '-'}`; }).join('  '));
}

// rules in the features' own units
const W = fit(rows);
const rules = {};
CLASSES.forEach((c, k) => {
  const w = { b: W[k][KEYS.length] };
  KEYS.forEach((key, j) => { w[key] = W[k][j] / sd[j]; w.b -= (W[k][j] * mean[j]) / sd[j]; });
  for (const key of Object.keys(w)) w[key] = Number(w[key].toFixed(4));
  rules[c] = w;
});
// high gain among the distorted guitars: flatter spectrum and fewer peaks than the median one
const dist = rows.filter((r) => r.label === 'distorted');
if (dist.length) {
  const jf = KEYS.indexOf('flat'), jc = KEYS.indexOf('crest');
  const mf = dist.reduce((s, r) => s + r.x[jf], 0) / dist.length, mc = dist.reduce((s, r) => s + r.x[jc], 0) / dist.length;
  const sf = Math.sqrt(dist.reduce((s, r) => s + (r.x[jf] - mf) ** 2, 0) / dist.length) || 1;
  const sc = Math.sqrt(dist.reduce((s, r) => s + (r.x[jc] - mc) ** 2, 0) / dist.length) || 1;
  const score = (r) => (r.x[jf] - mf) / sf - (r.x[jc] - mc) / sc;
  const med = dist.map(score).sort((a, b) => a - b)[dist.length >> 1];
  rules.lead = { b: Number((-mf / sf + mc / sc - med).toFixed(4)), flat: Number((1 / sf).toFixed(4)), crest: Number((-1 / sc).toFixed(4)) };
}
console.log(`export const TONE_RULES = ${JSON.stringify(rules)};`);
