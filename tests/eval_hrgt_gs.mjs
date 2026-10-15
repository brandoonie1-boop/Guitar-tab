// Note accuracy of the guitar transcription model on GuitarSet (the 120 files that
// analyze_note_errors.mjs uses), run through the app's JS code. Activations are cached.
// usage: node tests/eval_hrgt_gs.mjs <models dir> [--hop 500] [--limit n] [--opts '{"onsetThreshold":0.3}'] [--clean]
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { parseChain, loadParts } from '../engine/demucs.js';
import { hrModel, hrActivations, hrNotes } from '../engine/hrgt.js';
import { cleanNotes } from '../engine/notes.js';

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const models = args[0];
const hop = Number(arg('--hop', 500));
const limit = Number(arg('--limit', 1e9));
const opts = JSON.parse(arg('--opts', '{}'));
const clean = args.includes('--clean');
const tag = arg('--tag', `hop${hop}`);
const ANN = JSON.parse(fs.readFileSync('/home/claude/work/data/gs/guitarset_annotations.json', 'utf8'));
const names = Object.keys(ANN).sort().filter((_, i) => i % 3 === 0).slice(0, limit);
const CACHE = `/home/claude/work/data/gs/hrjs_${tag}`;
fs.mkdirSync(CACHE, { recursive: true });

let ort = null, model = null;
async function activations(name) {
  const p = path.join(CACHE, name + '.f32');
  if (fs.existsSync(p)) {
    const all = new Float32Array(fs.readFileSync(p).buffer.slice(0));
    const T = all.length / (88 * 4);
    const sl = (i) => all.subarray(i * T * 88, (i + 1) * T * 88);
    return { T, onset: sl(0), offset: sl(1), frame: sl(2), velocity: sl(3) };
  }
  if (!model) {
    const ortMod = await import('/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js');
    ort = ortMod.default || ortMod;
    model = hrModel(await loadParts(ort, parseChain(fs.readFileSync(path.join(models, 'hrgt.chain'))), new Uint8Array(fs.readFileSync(path.join(models, 'hrgt.weights.bin'))), { intraOpNumThreads: 2 }));
  }
  const w = decodeWav(fs.readFileSync(`/home/claude/work/data/guitarset/audio/${name}_mic.wav`));
  const x = resample(w.channels[0], w.sr, 16000, 64, 0.958);
  const act = await hrActivations(ort, model, x, { hop });
  const all = new Float32Array(act.T * 88 * 4);
  ['onset', 'offset', 'frame', 'velocity'].forEach((k, i) => all.set(act[k], i * act.T * 88));
  fs.writeFileSync(p, Buffer.from(all.buffer));
  return act;
}

function match(est, ref, tol = 0.05) {
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

let TP = 0, NE = 0, NR = 0;
const t0 = Date.now();
const sub = { comp: [0, 0, 0], solo: [0, 0, 0] };
for (const [i, name] of names.entries()) {
  const act = await activations(name);
  let est = hrNotes(act, opts);
  if (clean) est = cleanNotes(est.map((n) => ({ ...n, bends: [] })), { idPrefix: 'x:', mergeRetrigger: false });
  const ref = ANN[name].notes.map(([t, d, v]) => ({ t0: t, t1: t + d, pitch: Math.round(v) }));
  const { ri } = match(est, ref);
  let tp = 0;
  for (let k = 0; k < ref.length; k++) if (ri[k] >= 0) tp++;
  TP += tp; NE += est.length; NR += ref.length;
  const s = sub[name.endsWith('solo') ? 'solo' : 'comp'];
  s[0] += tp; s[1] += est.length; s[2] += ref.length;
  if (i % 20 === 19) console.error(`${i + 1}/${names.length} ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
const f = (tp, ne, nr) => { const P = tp / ne, R = tp / nr; return `P ${(100 * P).toFixed(1)} R ${(100 * R).toFixed(1)} F1 ${(200 * P * R / (P + R)).toFixed(1)}`; };
console.log(`${tag} files ${names.length}: ${f(TP, NE, NR)}  refs ${NR} est ${NE}  | comp ${f(...sub.comp)} | solo ${f(...sub.solo)}`);
if (model) await model.release();
