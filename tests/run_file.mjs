// Transcribe a WAV file from the command line (Node + onnxruntime-node), print the tab.
// usage: node tests/run_file.mjs file.wav [--off] [--range a b] [--out prefix]
// env: NO_HR=1 notes from basic-pitch only; NO_BEAT_MODEL=1 onset-envelope beat tracker
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { transcribe } from '../engine/pipeline.js';
import { parseChain, loadParts } from '../engine/demucs.js';
import { hrModel } from '../engine/hrgt.js';

const args = process.argv.slice(2);
const file = args[0];
const opts = {};
if (args.includes('--off')) opts.isolate = 'off';
if (args.includes('--nosplit')) opts.split = 'off';
const ri = args.indexOf('--range');
if (ri >= 0) opts.range = [Number(args[ri + 1]), Number(args[ri + 2])];
const oi = args.indexOf('--out');
const outPrefix = oi >= 0 ? args[oi + 1] : null;
const modelsDir = process.env.MODELS || '/home/claude/guitar-tab/models';
const ortPath = process.env.ORT || '/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js';
const ortMod = await import(ortPath);
const ort = ortMod.default || ortMod;

const wav = decodeWav(fs.readFileSync(file));
let L = wav.channels[0], R = wav.channels[1] || wav.channels[0];
if (wav.sr !== 44100) { L = resample(L, wav.sr, 44100); R = wav.channels[1] ? resample(R, wav.sr, 44100) : L; }
let demucs = null, bp = null, bm = null, hr = null;
const t0 = Date.now();
let lastLabel = '';
const res = await transcribe({ left: L, right: wav.channels[1] ? R : null, sr: 44100, name: path.basename(file) }, opts, {
  ort,
  getDemucs: async () => demucs || (demucs = await ort.InferenceSession.create(process.env.DEMUCS || path.join(modelsDir, 'htdemucs_6s.f16.onnx'), { intraOpNumThreads: 2 })),
  getBasicPitch: async () => bp || (bp = await ort.InferenceSession.create(path.join(modelsDir, 'basic_pitch.onnx'))),
  getBeatModel: process.env.NO_BEAT_MODEL ? null : async () => bm || (bm = await ort.InferenceSession.create(path.join(modelsDir, 'beat_this_small.f16.onnx'), { intraOpNumThreads: 2 })),
  getTranscriber: process.env.NO_HR ? null : async () => hr || (hr = hrModel(await loadParts(ort, parseChain(fs.readFileSync(path.join(modelsDir, 'hrgt.chain'))), new Uint8Array(fs.readFileSync(path.join(modelsDir, 'hrgt.weights.bin'))), { intraOpNumThreads: 2 }))),
  progress: (p) => { if (p.label !== lastLabel) { lastLabel = p.label; process.stderr.write(`\n[${((Date.now() - t0) / 1000).toFixed(1)}s] ${p.label}`); } },
});
process.stderr.write(`\n[${((Date.now() - t0) / 1000).toFixed(1)}s] done\n`);
console.log('BPM', res.rhythm.bpm.toFixed(1), 'meter', res.rhythm.meter, 'tuning offset', res.tuningCents, 'cents', 'split', JSON.stringify(res.splitInfo));
for (const p of res.parts) {
  console.log(`\n== ${p.name}: ${p.notes.length} notes; setup: ${p.setup.name} capo ${p.setup.capo}`);
  for (const r of p.ranking.slice(0, 4)) console.log(`   ${r.name} capo ${r.capo} ${(r.confidence * 100).toFixed(0)}%${r.equivalents.length ? ' (same sound: ' + r.equivalents.map((e) => e.name + ' capo ' + e.capo).join(', ') + ')' : ''}`);
}
console.log('\n' + res.ascii);
if (outPrefix) {
  fs.writeFileSync(outPrefix + '.atex', res.alphaTex);
  fs.writeFileSync(outPrefix + '.txt', res.ascii);
  fs.writeFileSync(outPrefix + '.notes.json', JSON.stringify(res.parts.map((p) => ({ name: p.name, setup: p.setup, notes: p.notes.map(({ bends, ...n }) => n) }))));
}
