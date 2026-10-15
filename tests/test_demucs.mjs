// Compare the JS Demucs port (STFT/iSTFT + ONNX core) against PyTorch apply_model output.
// usage: node tests/test_demucs.mjs <models dir with sep_in.f32 / sep_ref_*.f32> <onnx or .chain path> [node|web]
import fs from 'node:fs';
import path from 'node:path';
import { separate, parseChain, loadChain } from '../engine/demucs.js';

const [dir, onnxPath, runtime = 'node'] = process.argv.slice(2);
const ort = runtime === 'web'
  ? await import('/home/claude/work/jsenv/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs')
  : (await import('/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js')).default ?? await import('/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js');

function readF32(p) {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
const inp = readF32(path.join(dir, 'sep_in.f32'));
const n = inp.length / 2;
const left = inp.slice(0, n), right = inp.slice(n);
const opts = runtime === 'web' ? { executionProviders: ['wasm'], ...(process.env.SESS ? JSON.parse(process.env.SESS) : {}) } : { intraOpNumThreads: 2 };
if (runtime === 'web') { ort.env.wasm.numThreads = Number(process.env.THREADS || 1); }
let t0 = Date.now();
// a packed chain (tools/pack_demucs.py) needs the shared weights file next to it (web runtime only)
const session = onnxPath.endsWith('.chain')
  ? await loadChain(ort, parseChain(fs.readFileSync(onnxPath)), new Uint8Array(fs.readFileSync(path.join(path.dirname(onnxPath), 'demucs.weights.bin'))), opts)
  : await ort.InferenceSession.create(fs.readFileSync(onnxPath), opts);
console.log('session', Date.now() - t0, 'ms');
t0 = Date.now();
const out = await separate(ort, session, left, right, {
  keep: { guitar: 'stereo', drums: 'stereo', other: 'stereo', bass: 'mono' },
  onProgress: (f) => process.stdout.write(`\r${(f * 100).toFixed(0)}%   `),
});
console.log('\nseparate', Date.now() - t0, 'ms for', (n / 44100).toFixed(1), 's audio');
for (const s of ['guitar', 'drums', 'other']) {
  const ref = readF32(path.join(dir, `sep_ref_${s}.f32`));
  let maxErr = 0, maxRef = 0, errE = 0, refE = 0;
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < n; i++) {
      const e = Math.abs(out[s][c][i] - ref[c * n + i]);
      maxErr = Math.max(maxErr, e); maxRef = Math.max(maxRef, Math.abs(ref[c * n + i]));
      errE += e * e; refE += ref[c * n + i] ** 2;
    }
  }
  console.log(s, 'max err', maxErr.toExponential(2), 'ref peak', maxRef.toFixed(3), 'SNR vs torch', (10 * Math.log10(refE / errE)).toFixed(1), 'dB');
}
{
  const ref = readF32(path.join(dir, `sep_ref_bass.f32`));
  let errE = 0, refE = 0;
  for (let i = 0; i < n; i++) { const r = 0.5 * (ref[i] + ref[n + i]); errE += (out.bass[0][i] - r) ** 2; refE += r * r; }
  console.log('bass(mono) SNR vs torch', (10 * Math.log10(refE / errE)).toFixed(1), 'dB');
}
