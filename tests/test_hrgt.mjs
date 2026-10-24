// JS front end + ONNX parts + note decoding of the guitar transcription model vs. the
// PyTorch original (reference data from tools/make_hrgt_ref.py).
// usage: node tests/test_hrgt.mjs <ref dir> <models dir with hrgt.chain + hrgt.weights.bin> [hop]
// env: ORT = path of onnxruntime-node's dist/index.js
import fs from 'node:fs';
import path from 'node:path';
import { parseChain, loadParts } from '../engine/demucs.js';
import { HR, hrMelBands, hrLogMel, hrModel, hrActivations, hrNotes } from '../engine/hrgt.js';

const [ref, models, hopArg] = process.argv.slice(2);
const f32 = (p) => new Float32Array(fs.readFileSync(path.join(ref, p)).buffer.slice(0));
const ortMod = await import(process.env.ORT || '/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js');
const ort = ortMod.default || ortMod;

// mel filters
const melW = f32('melW.f32'); // (1025, 229)
let fbErr = 0;
const bands = hrMelBands();
for (let m = 0; m < HR.nMels; m++) {
  const dense = new Float32Array(1025);
  dense.set(bands[m].w, bands[m].first);
  for (let i = 0; i < 1025; i++) fbErr = Math.max(fbErr, Math.abs(dense[i] - melW[i * HR.nMels + m]));
}
console.log('mel filters max diff', fbErr.toExponential(2));

const x = f32('audio16k.f32');
const t0 = Date.now();
const { mel, T } = hrLogMel(x);
const melRef = f32('mel.f32');
let melErr = 0;
for (let i = 0; i < mel.length; i++) melErr = Math.max(melErr, Math.abs(mel[i] - melRef[i]));
console.log(`log-mel T ${T} (ref ${melRef.length / HR.nMels}), max diff ${melErr.toFixed(4)} dB, ${Date.now() - t0} ms`);

const chain = parseChain(fs.readFileSync(path.join(models, 'hrgt.chain')));
const weights = new Uint8Array(fs.readFileSync(path.join(models, 'hrgt.weights.bin')));
const model = hrModel(await loadParts(ort, chain, weights, { intraOpNumThreads: 2 }));
const t1 = Date.now();
const act = await hrActivations(ort, model, x, { hop: Number(hopArg || 500) });
console.log(`model ${(Date.now() - t1) / 1000} s for ${(x.length / 16000).toFixed(1)} s of audio`);
const meta = JSON.parse(fs.readFileSync(path.join(ref, 'ref.json'), 'utf8'));
for (const [k, file] of [['onset', 'reg_onset_output'], ['offset', 'reg_offset_output'], ['frame', 'frame_output'], ['velocity', 'velocity_output']]) {
  const r = f32(file + '.f32');
  let e = 0, n = Math.min(r.length, act[k].length);
  for (let i = 0; i < n; i++) e = Math.max(e, Math.abs(r[i] - act[k][i]));
  console.log(`${k.padEnd(8)} max diff ${e.toFixed(4)} (frames ${act.T} vs ${r.length / 88})`);
}
// decode the reference outputs with the JS port: must give the Python notes exactly
const refAct = { T: meta.T, onset: f32('reg_onset_output.f32'), offset: f32('reg_offset_output.f32'), frame: f32('frame_output.f32'), velocity: f32('velocity_output.f32') };
const jsNotes = hrNotes(refAct);
const key = (n) => `${n[2]}@${n[0].toFixed(3)}-${n[1].toFixed(3)}`;
const a = new Set(meta.notes.map(key));
const b = new Set(jsNotes.map((n) => key([n.t0, n.t1, n.pitch])));
let same = 0;
for (const k of a) if (b.has(k)) same++;
console.log(`decoding: python ${a.size} notes, js ${b.size}, identical ${same}`);
const got = hrNotes(act);
let match = 0;
for (const n of got) if (meta.notes.some((r) => r[2] === n.pitch && Math.abs(r[0] - n.t0) < 0.02)) match++;
console.log(`end to end: ${got.length} notes, ${match} within 20 ms of a Python note (python ${meta.notes.length})`);
await model.release();
