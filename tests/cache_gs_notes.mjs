// Transcribe GuitarSet files with the pipeline and cache detected notes + references, for fingering sweeps.
// usage: node tests/cache_gs_notes.mjs <out.json> <n> [--mix] [--every k] [--offset o]
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { transcribe } from '../engine/pipeline.js';

const args = process.argv.slice(2);
const out = args[0];
const n = Number(args[1] || 40);
const mix = args.includes('--mix');
const every = args.includes('--every') ? Number(args[args.indexOf('--every') + 1]) : 9;
const offset = args.includes('--offset') ? Number(args[args.indexOf('--offset') + 1]) : 0;
const ANN = JSON.parse(fs.readFileSync('/home/claude/work/data/gs/guitarset_annotations.json', 'utf8'));
const AUD = '/home/claude/work/data/guitarset/audio';
const ortMod = await import('/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js');
const ort = ortMod.default || ortMod;
let demucs = null, bp = null;
const names = Object.keys(ANN).filter((_, i) => i % every === offset).slice(0, n);
const res = {};
for (const name of names) {
  const wav = decodeWav(fs.readFileSync(path.join(AUD, `${name}_mic.wav`)));
  let x = wav.channels[0];
  if (mix) {
    const { addBand } = await import('./gs_band.mjs');
    x = addBand(x, ANN[name].beats);
  }
  const r = await transcribe({ left: x, right: null, sr: 44100, name }, { isolate: mix ? 'guitar' : 'off', split: 'off', tuning: [40, 45, 50, 55, 59, 64], capo: 0 }, {
    ort,
    getDemucs: async () => demucs || (demucs = await ort.InferenceSession.create('/home/claude/guitar-tab/models/htdemucs_6s.f16.onnx', { intraOpNumThreads: 2 })),
    getBasicPitch: async () => bp || (bp = await ort.InferenceSession.create('/home/claude/guitar-tab/models/basic_pitch.onnx')),
  });
  res[name] = {
    est: r.parts[0].notes.map((q) => ({ t0: q.t0, t1: q.t1, pitch: q.pitch, amp: q.amp, onsetProb: q.onsetProb })),
    ref: ANN[name].notes,
    bpm: r.rhythm.bpm,
  };
  process.stdout.write('.');
}
fs.writeFileSync(out, JSON.stringify(res));
console.log('\ncached', names.length);
