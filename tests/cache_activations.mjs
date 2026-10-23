// Run the guitar note model over WAV files and keep its frame outputs (onset, offset, frame,
// velocity; 100 frames/s, 88 keys) as <out dir>/<name>.f32, for tests/eval_strings.mjs.
// usage: node tests/cache_activations.mjs <models dir with hrgt.chain + hrgt.weights.bin> <out dir> <file.wav>...
//        (GuitarSet file names lose their _mic / _mix suffix)
// env: ORT = path of onnxruntime-node's dist/index.js
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { parseChain, loadParts } from '../engine/demucs.js';
import { hrModel, hrActivations } from '../engine/hrgt.js';

const [models, outDir, ...wavs] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const ortMod = await import(process.env.ORT || 'onnxruntime-node');
const ort = ortMod.default || ortMod;
const model = hrModel(await loadParts(ort, parseChain(fs.readFileSync(path.join(models, 'hrgt.chain'))), new Uint8Array(fs.readFileSync(path.join(models, 'hrgt.weights.bin'))), { intraOpNumThreads: 4 }));
for (const file of wavs) {
  const name = path.basename(file).replace(/(_mic|_mix)?\.wav$/, '');
  const out = path.join(outDir, name + '.f32');
  if (fs.existsSync(out)) continue;
  const w = decodeWav(fs.readFileSync(file));
  const mono = w.channels.length > 1 ? Float32Array.from(w.channels[0], (v, i) => 0.5 * (v + w.channels[1][i])) : w.channels[0];
  const act = await hrActivations(ort, model, resample(mono, w.sr, 16000, 64, 0.958), { hop: 500 });
  const all = new Float32Array(act.T * 88 * 4);
  ['onset', 'offset', 'frame', 'velocity'].forEach((k, i) => all.set(act[k], i * act.T * 88));
  fs.writeFileSync(out, Buffer.from(all.buffer));
  console.log(name, `${(act.T / 100).toFixed(1)} s`);
}
await model.release();
