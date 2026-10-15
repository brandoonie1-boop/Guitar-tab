// Guitar isolation quality vs. model segment length, on GuitarSet guitar + synthetic band.
// The true guitar is known (the original recording), so SDR is measured directly.
// usage: node tests/eval_sep_segment.mjs [nFiles] [seconds]
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { separate } from '../engine/demucs.js';
import { addBand } from './gs_band.mjs';

const nFiles = Number(process.argv[2] || 6), secs = Number(process.argv[3] || 30);
const ANN = JSON.parse(fs.readFileSync('/home/claude/work/data/gs/guitarset_annotations.json', 'utf8'));
const ortMod = await import('/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js');
const ort = ortMod.default || ortMod;
const models = [
  [343980, 'models/htdemucs_6s.f16.onnx'],
  [171990, 'models/htdemucs_6s_seg171990.f16.onnx'],
  [114660, 'models/htdemucs_6s_seg114660.f16.onnx'],
  [85995, 'models/htdemucs_6s_seg85995.f16.onnx'],
];
const names = Object.keys(ANN).filter((n) => n.endsWith('comp')).filter((_, i) => i % 23 === 0).slice(0, nFiles);
const sdr = (est, ref) => { let e = 0, r = 0; for (let i = 0; i < ref.length; i++) { e += (est[i] - ref[i]) ** 2; r += ref[i] ** 2; } return 10 * Math.log10(r / e); };
const results = Object.fromEntries(models.map(([s]) => [s, []]));
const vsFull = Object.fromEntries(models.map(([s]) => [s, []]));
// prepare the mixes once
const items = names.map((name) => {
  const wav = decodeWav(fs.readFileSync(path.join('/home/claude/work/data/guitarset/audio', `${name}_mic.wav`)));
  const g = wav.channels[0].slice(0, secs * 44100);
  return { g, mix: addBand(g, ANN[name].beats) };
});
const fullEst = [];
// one model in memory at a time
for (const [seg, p] of models) {
  const session = await ort.InferenceSession.create(p, { intraOpNumThreads: 2, enableCpuMemArena: false });
  for (let k = 0; k < items.length; k++) {
    const { g, mix } = items[k];
    const out = await separate(ort, session, mix, mix, { keep: { guitar: 'mono' }, segment: seg });
    const est = out.guitar[0];
    results[seg].push(sdr(est, g));
    if (seg === 343980) fullEst[k] = est; else vsFull[seg].push(sdr(est, fullEst[k]));
    process.stdout.write('.');
  }
  if (session.release) await session.release();
}
console.log();
for (const [seg] of models) {
  const m = results[seg].reduce((a, b) => a + b, 0) / results[seg].length;
  const v = vsFull[seg].length ? (vsFull[seg].reduce((a, b) => a + b, 0) / vsFull[seg].length).toFixed(1) + ' dB' : '-';
  console.log(`segment ${(seg / 44100).toFixed(2)} s: guitar SDR ${m.toFixed(2)} dB (per file ${results[seg].map((x) => x.toFixed(1)).join(', ')}), similarity to 7.8 s output ${v}`);
}
