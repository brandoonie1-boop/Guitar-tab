// Full-pipeline evaluation on real guitar recordings (GuitarSet), optionally mixed with drums and bass.
// usage: node tests/eval_gs_pipeline.mjs <n files> [--mix] [--off] [--every k] [--offset o]
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { transcribe } from '../engine/pipeline.js';

const args = process.argv.slice(2);
const nFiles = Number(args[0] || 10);
const mix = args.includes('--mix');
const off = args.includes('--off');
const every = args.includes('--every') ? Number(args[args.indexOf('--every') + 1]) : 9;
const offsetArg = args.includes('--offset') ? Number(args[args.indexOf('--offset') + 1]) : 0;
const ANN = JSON.parse(fs.readFileSync('/home/claude/work/data/gs/guitarset_annotations.json', 'utf8'));
const AUD = '/home/claude/work/data/guitarset/audio';
const ortMod = await import('/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js');
const ort = ortMod.default || ortMod;
const MODELS = '/home/claude/guitar-tab/models';
let demucs = null, bp = null, bm = null;
const STD = [40, 45, 50, 55, 59, 64];
const extraOpts = process.env.OPTS ? JSON.parse(process.env.OPTS) : {};

import { addBand } from './gs_band.mjs';

const names = Object.keys(ANN).filter((_, i) => i % every === offsetArg).slice(0, nFiles);
let TP = 0, NE = 0, NR = 0, strOk = 0, setupOk = 0, tempoOk = 0;
const t0 = Date.now();
for (const name of names) {
  const wav = decodeWav(fs.readFileSync(path.join(AUD, `${name}_mic.wav`)));
  let x = wav.channels[0];
  if (mix) x = addBand(x, ANN[name].beats);
  const res = await transcribe({ left: x, right: null, sr: 44100, name }, { isolate: off ? 'off' : 'guitar', split: 'off', ...extraOpts }, {
    ort,
    // DEMUCS=<onnx file> SEG=<segment samples> to evaluate another export (e.g. the light model)
    getDemucs: async () => demucs || (demucs = Object.assign(await ort.InferenceSession.create(process.env.DEMUCS || path.join(MODELS, 'htdemucs_6s.f16.onnx'), { intraOpNumThreads: 2 }), { segment: Number(process.env.SEG || 343980) })),
    getBasicPitch: async () => bp || (bp = await ort.InferenceSession.create(path.join(MODELS, 'basic_pitch.onnx'))),
    getBeatModel: process.env.NO_BEAT_MODEL ? null : async () => bm || (bm = await ort.InferenceSession.create(path.join(MODELS, 'beat_this_small.f16.onnx'), { intraOpNumThreads: 2 })),
  });
  const p = res.parts[0];
  const est = p.notes.filter((n) => !n.dropped);
  const ref = ANN[name].notes.map(([t, d, v, s]) => ({ t0: t, pitch: Math.round(v), string: s, fret: Math.round(v) - STD[s] }));
  const used = new Set();
  let tp = 0, so = 0;
  for (const r of ref) {
    let bi = -1, bd = 1;
    est.forEach((e, i) => { if (!used.has(i) && e.pitch === r.pitch) { const d = Math.abs(e.t0 - r.t0); if (d <= 0.05 && d < bd) { bd = d; bi = i; } } });
    if (bi >= 0) { used.add(bi); tp++; const e = est[bi]; if (e.string === r.string && e.fret === r.fret && p.setup.capo === 0 && p.setup.tuning.join() === STD.join()) so++; }
  }
  const setup = `${p.setup.name} c${p.setup.capo}`;
  const sOk = p.setup.tuning.join() === STD.join() && p.setup.capo === 0;
  const tTrue = ANN[name].tempo;
  const ratio = res.rhythm.bpm / tTrue;
  const tOk = Math.abs(ratio - 1) < 0.04;
  TP += tp; NE += est.length; NR += ref.length; strOk += so; setupOk += sOk ? 1 : 0; tempoOk += tOk ? 1 : 0;
  console.log(`${name.padEnd(28)} P ${(tp / Math.max(1, est.length)).toFixed(2)} R ${(tp / ref.length).toFixed(2)} str ${(so / Math.max(1, tp)).toFixed(2)} setup ${sOk ? 'ok ' : 'BAD'} ${setup.padEnd(22)} bpm ${res.rhythm.bpm.toFixed(0)}/${tTrue}${tOk ? '' : ' x'}`);
}
const P = TP / NE, R = TP / NR;
console.log(`\nfiles ${names.length} (${((Date.now() - t0) / 1000).toFixed(0)} s)  notes P ${(P * 100).toFixed(1)} R ${(R * 100).toFixed(1)} F1 ${(200 * P * R / (P + R)).toFixed(1)}  string acc ${(100 * strOk / TP).toFixed(1)}%  setup ok ${setupOk}/${names.length}  tempo ok ${tempoOk}/${names.length}`);
