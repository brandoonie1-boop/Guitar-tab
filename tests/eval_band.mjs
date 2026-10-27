// Notes found in a band: guitar recordings with the true tab (GuitarSet, EGSet12; clean or through
// a simulated amp) mixed into real backing tracks (MUSDB18 test songs: drums, bass and vocals;
// evaluation only) with the guitar 3-9 dB below the band, the guitar isolated by the app's own
// BS-RoFormer cut (tools/cut_bsr.mjs) through engine/pipeline.js separateStage as the app runs it
// ("Balanced": 10 % overlap), then the note model through the app's JS code. The isolated guitar
// is kept per take, so note models are compared on the very same separation.
//
// usage: node tests/eval_band.mjs <note models dir> <annotations.json> <audio dir> <backing dir> <bsr cut .bin>
//          [--suffix _mic] [--every k] [--offset o] [--limit n] [--fx amp:20] [--stems dir]
//          [--json out.json] [--hop 500] [--onset 0.4] [--per-file] [--tag name]
// backing dir: MUSDB18 test stems as "<song>.drums.mp3", ".bass.mp3", ".vocals.mp3" (decoded with ffmpeg)
// env: ORT = path of onnxruntime-node's dist/index.js
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { hrActivations } from '../engine/hrgt.js';
import { loadBsrChain } from '../engine/bsroformer.js';
import { separateStage } from '../engine/pipeline.js';
import { applyFx, rmsOf, loadNoteModel, readActs, writeActs, decodeNotes, matchNotes, f1 } from './evallib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--') && !['--per-file'].includes(args[i - 1])));
const [models, annPath, audioDir, backingDir, bsrPath] = pos;
const suffix = opt('--suffix', '');
const every = Number(opt('--every', 1)), offset = Number(opt('--offset', 0)), limit = Number(opt('--limit', 1e9));
const fx = opt('--fx', '');
const hop = Number(opt('--hop', 500));
const onset = opt('--onset') ? Number(opt('--onset')) : undefined;
const ann = JSON.parse(fs.readFileSync(annPath, 'utf8'));
const names = Object.keys(ann).sort().filter((_, i) => i % every === offset).slice(0, limit);
const set = path.basename(annPath, '.json');
const tag = opt('--tag', `${set}_band_${fx.replace(':', '') || 'clean'}`);
const stemsDir = opt('--stems', path.join(path.dirname(bsrPath), 'stems', tag));
const cacheDir = path.join(models, 'acts', `${tag}_h${hop}`);
fs.mkdirSync(stemsDir, { recursive: true });
fs.mkdirSync(cacheDir, { recursive: true });
const SR = 44100;

const ortMod = await import(process.env.ORT || 'onnxruntime-node');
const ort = ortMod.default || ortMod;

// backing: drums + bass + vocals of a MUSDB18 test song, decoded once
const songs = [...new Set(fs.readdirSync(backingDir).filter((f) => f.endsWith('.drums.mp3')).map((f) => f.slice(0, -'.drums.mp3'.length)))].sort();
const backCache = path.join(path.dirname(bsrPath), 'backing');
fs.mkdirSync(backCache, { recursive: true });
function backing(song) {
  const p = path.join(backCache, song.replace(/[^\w.-]+/g, '_') + '.f32');
  if (!fs.existsSync(p)) {
    let sum = null;
    for (const stem of ['drums', 'bass', 'vocals']) {
      const raw = execFileSync('ffmpeg', ['-v', 'error', '-i', path.join(backingDir, `${song}.${stem}.mp3`), '-f', 'f32le', '-ac', '2', '-ar', String(SR), '-'], { maxBuffer: 1 << 30 });
      const x = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength >> 2);
      if (!sum) sum = Float32Array.from(x);
      else for (let i = 0; i < Math.min(sum.length, x.length); i++) sum[i] += x[i];
    }
    fs.writeFileSync(p, Buffer.from(sum.buffer));
  }
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); // interleaved stereo
}

let session = null;
async function isolate(name, i) {
  const pL = path.join(stemsDir, name + '.L.f32'), pR = path.join(stemsDir, name + '.R.f32'), pj = path.join(stemsDir, name + '.json');
  if (fs.existsSync(pj)) {
    const rd = (p) => { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); };
    return { gL: rd(pL), gR: rd(pR), ...JSON.parse(fs.readFileSync(pj, 'utf8')) };
  }
  const w = decodeWav(fs.readFileSync(path.join(audioDir, `${name}${suffix}.wav`)));
  let g = w.channels.length > 1 ? Float32Array.from(w.channels[0], (v, k) => 0.5 * (v + w.channels[1][k])) : w.channels[0];
  g = applyFx(g, w.sr, fx);
  if (w.sr !== SR) g = resample(g, w.sr, SR, 64, 0.958);
  // the same backing song, place and level for a take whatever is being compared
  const song = songs[(i * 7) % songs.length];
  const bk = backing(song);
  const frames = bk.length / 2;
  const start = Math.floor(0.3 * Math.max(0, frames - g.length));
  const n = g.length;
  const bL = new Float32Array(n), bR = new Float32Array(n);
  for (let k = 0; k < n && start + k < frames; k++) { bL[k] = bk[2 * (start + k)]; bR[k] = bk[2 * (start + k) + 1]; }
  const gdb = -3 - 6 * ((i * 0.6180339887) % 1); // guitar 3-9 dB below the band
  const gb = Math.pow(10, gdb / 20) * Math.sqrt(0.5 * (rmsOf(bL) ** 2 + rmsOf(bR) ** 2)) / Math.max(1e-9, rmsOf(g));
  const L = new Float32Array(n), R = new Float32Array(n);
  let pk = 1e-9;
  for (let k = 0; k < n; k++) { L[k] = bL[k] + gb * g[k]; R[k] = bR[k] + gb * g[k]; pk = Math.max(pk, Math.abs(L[k]), Math.abs(R[k])); }
  if (pk > 0.98) for (let k = 0; k < n; k++) { L[k] *= 0.98 / pk; R[k] *= 0.98 / pk; }
  if (!session) {
    // memory arenas on: faster here (the app turns them off to save a phone's memory)
    session = await loadBsrChain(ort, new Uint8Array(fs.readFileSync(bsrPath)), { intraOpNumThreads: Number(process.env.THREADS || 4) });
    session.stems = ['guitar'];
  }
  const t = Date.now();
  const st = await separateStage({ left: L, right: R, sr: SR, name }, { isolate: 'guitar', split: 'off', overlap: 0.1 }, { ort, getDemucs: async () => session });
  const meta = { song, gdb: Math.round(gdb * 10) / 10, alone: !!st.alone, leftoverDb: st.leftoverDb, sepSeconds: (Date.now() - t) / 1000 };
  fs.writeFileSync(pL, Buffer.from(st.gL.buffer, st.gL.byteOffset, st.gL.byteLength));
  fs.writeFileSync(pR, Buffer.from(st.gR.buffer, st.gR.byteOffset, st.gR.byteLength));
  fs.writeFileSync(pj, JSON.stringify(meta));
  return { gL: st.gL, gR: st.gR, ...meta };
}

let model = null;
async function activations(name, i) {
  const p = path.join(cacheDir, name + '.f32');
  if (fs.existsSync(p)) return readActs(p);
  const st = await isolate(name, i);
  if (!model) model = await loadNoteModel(ort, models);
  const mono = Float32Array.from(st.gL, (v, k) => 0.5 * (v + st.gR[k]));
  const act = await hrActivations(ort, model, resample(mono, SR, 16000, 64, 0.958), { hop });
  writeActs(p, act);
  return act;
}

let TP = 0, NE = 0, NR = 0;
const perFile = {};
const t0 = Date.now();
for (const [i, name] of names.entries()) {
  const est = decodeNotes(await activations(name, i), { onset });
  const ref = ann[name].notes.map(([t, d, v]) => ({ t0: t, t1: t + d, pitch: Math.round(v) }));
  const { ri } = matchNotes(est, ref);
  let tp = 0;
  for (let k = 0; k < ref.length; k++) if (ri[k] >= 0) tp++;
  TP += tp; NE += est.length; NR += ref.length;
  perFile[name] = { tp, est: est.length, ref: ref.length, F: f1(tp, est.length, ref.length).F };
  if (args.includes('--per-file')) {
    const m = JSON.parse(fs.readFileSync(path.join(stemsDir, name + '.json'), 'utf8'));
    console.log(name.padEnd(28), (100 * perFile[name].F).toFixed(1), `guitar ${m.gdb} dB under "${m.song}"${m.alone ? ' (taken as a guitar on its own!)' : ''}`);
  }
  if (i % 10 === 9) console.error(`${i + 1}/${names.length} ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
const { P, R, F } = f1(TP, NE, NR);
console.log(`${tag} (${names.length} takes): notes found F1 ${(100 * F).toFixed(1)}  P ${(100 * P).toFixed(1)} R ${(100 * R).toFixed(1)}  refs ${NR} est ${NE}`);
if (opt('--json')) fs.writeFileSync(opt('--json'), JSON.stringify({ tag, takes: names.length, P, R, F, perFile }, null, 1));
if (model) await model.release();
if (session) await session.release();
