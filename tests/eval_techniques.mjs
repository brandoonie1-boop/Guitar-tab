// Playing techniques the app marks, on Guitar-TECHS's technique takes (every note of a take is
// played with the take's technique: bends, vibrato, palm mutes, harmonics) and on plain scales
// and chords (where any mark is a false one). Runs the whole engine (isolation off) and counts,
// per take, the detected notes and how many carry each mark.
// usage: node tests/eval_techniques.mjs <Guitar-TECHS dir> [--players P1,P2] [--kinds directinput,micamp] [--seconds 90] [--json out.json]
// env: MODELS = dir with basic_pitch.onnx, beat_this_small.f16.onnx, hrgt.chain, hrgt.weights.bin; ORT
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { transcribe } from '../engine/pipeline.js';
import { parseChain, loadParts } from '../engine/demucs.js';
import { hrModel } from '../engine/hrgt.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const dir = args[0];
const players = opt('--players', 'P1,P2').split(',');
const kinds = opt('--kinds', 'directinput').split(',');
const seconds = Number(opt('--seconds', 90));
const modelsDir = process.env.MODELS;
const ortMod = await import(process.env.ORT || 'onnxruntime-node');
const ort = ortMod.default || ortMod;
let bp = null, bm = null, hr = null;
const env = {
  ort,
  getBasicPitch: async () => bp || (bp = await ort.InferenceSession.create(path.join(modelsDir, 'basic_pitch.onnx'))),
  getBeatModel: async () => bm || (bm = await ort.InferenceSession.create(path.join(modelsDir, 'beat_this_small.f16.onnx'), { intraOpNumThreads: 2 })),
  getTranscriber: async () => hr || (hr = hrModel(await loadParts(ort, parseChain(fs.readFileSync(path.join(modelsDir, 'hrgt.chain'))), new Uint8Array(fs.readFileSync(path.join(modelsDir, 'hrgt.weights.bin'))), { intraOpNumThreads: 4 }))),
};

const TAKES = [
  ['techniques', 'Bendings', 'bend'], ['techniques', 'Vibrato', 'vibrato'], ['techniques', 'PalmMute', 'palmMute'],
  ['techniques', 'Harmonics', 'harmonic'], ['techniques', 'PinchHarmonics', 'harmonic'],
  ['scales', 'A', null], ['scales', 'E', null], ['chords', 'Drop3_Maj7', null], ['chords', 'Set1_7', null],
];
const MARKS = ['bend', 'vibrato', 'legato', 'slide', 'palmMute', 'harmonic'];
const rows = [];
for (const p of players) {
  for (const kind of kinds) {
    for (const [set, key, want] of TAKES) {
      const file = path.join(dir, `${p}_${set}`, 'audio', kind, `${kind}_${key}.wav`);
      if (!fs.existsSync(file)) continue;
      const w = decodeWav(fs.readFileSync(file));
      let x = w.channels.length > 1 ? Float32Array.from(w.channels[0], (v, i) => 0.5 * (v + w.channels[1][i])) : w.channels[0];
      if (w.sr !== 44100) x = resample(x, w.sr, 44100);
      x = x.subarray(0, Math.min(x.length, seconds * 44100));
      const res = await transcribe({ left: x, right: null, sr: 44100, name: path.basename(file) }, { isolate: 'off', split: 'off' }, env);
      const counts = Object.fromEntries(MARKS.map((m) => [m, 0]));
      let notes = 0;
      for (const part of res.parts) {
        const live = part.notes.filter((n) => !n.dropped && !n.deleted);
        notes += live.length;
        for (const n of live) {
          const t = part.tech && part.tech.get(n.id);
          for (const m of MARKS) if ((t && t[m]) || n[m]) counts[m]++;
        }
      }
      const row = { take: `${p} ${kind} ${set}/${key}`, want, notes, counts };
      rows.push(row);
      console.log(row.take.padEnd(42), `notes ${String(notes).padStart(4)}`, MARKS.map((m) => `${m} ${(100 * counts[m] / Math.max(1, notes)).toFixed(0).padStart(3)}%`).join('  '), want ? `<- ${want}` : '');
    }
  }
}
if (opt('--json')) fs.writeFileSync(opt('--json'), JSON.stringify(rows, null, 1));
