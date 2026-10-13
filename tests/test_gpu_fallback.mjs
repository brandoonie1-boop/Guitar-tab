// The GPU path must fall back to the processor when the separation output is garbage.
import fs from 'node:fs';
import { decodeWav } from '../engine/wav.js';
import { transcribe } from '../engine/pipeline.js';
const ortMod = await import('/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js');
const ort = ortMod.default || ortMod;
const wav = decodeWav(fs.readFileSync(process.argv[2]));
const real = await ort.InferenceSession.create('/home/claude/guitar-tab/models/htdemucs_6s.f16.onnx', { intraOpNumThreads: 2 });
const broken = {
  inputNames: real.inputNames, outputNames: real.outputNames,
  async run(feeds) { // a "GPU" that returns zeros
    return { spec: new ort.Tensor('float32', new Float32Array(6 * 4 * 2048 * 336), [1, 6, 4, 2048, 336]), wave: new ort.Tensor('float32', new Float32Array(6 * 2 * 343980), [1, 6, 2, 343980]) };
  },
};
let fellBack = false, bp = null;
const env = {
  ort,
  checkSeparation: true,
  getDemucs: async () => broken,
  fallbackToCpu: async (err) => { fellBack = err.message; return { ort, session: real }; },
  getBasicPitch: async () => bp || (bp = await ort.InferenceSession.create('/home/claude/guitar-tab/models/basic_pitch.onnx')),
  progress: () => {},
};
const res = await transcribe({ left: wav.channels[0], right: wav.channels[1], sr: 44100, name: 'x.wav' }, {}, env);
console.log('fell back because:', fellBack, '| notes:', res.parts[0].notes.length, '| setup:', res.parts[0].setup.name, 'capo', res.parts[0].setup.capo);
