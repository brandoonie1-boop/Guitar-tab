// Tempo / beat evaluation on GuitarSet (+ synthetic band). usage: node tests/eval_tempo.mjs [n] [--clean]
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { analyzeRhythm } from '../engine/beats.js';
import { addBand } from './gs_band.mjs';

const n = Number(process.argv[2] || 40);
const clean = process.argv.includes('--clean');
const ANN = JSON.parse(fs.readFileSync('/home/claude/work/data/gs/guitarset_annotations.json', 'utf8'));
const names = Object.keys(ANN).filter((_, i) => i % 5 === 0).slice(0, n);
let ok = 0, oct = 0, beatF = 0;
for (const name of names) {
  const wav = decodeWav(fs.readFileSync(path.join('/home/claude/work/data/guitarset/audio', `${name}_mic.wav`)));
  let x = wav.channels[0];
  if (!clean) x = addBand(x, ANN[name].beats);
  const r = analyzeRhythm(x, 44100);
  const t = ANN[name].tempo;
  const ratio = r.bpm / t;
  const good = Math.abs(ratio - 1) < 0.04;
  const octave = Math.abs(ratio - 0.5) < 0.03 || Math.abs(ratio - 2) < 0.08;
  // beat F-measure (70 ms)
  const ref = ANN[name].beats.map((b) => b[0]);
  let hit = 0;
  const used = new Set();
  for (const b of r.beats) { const j = ref.findIndex((q, k) => !used.has(k) && Math.abs(q - b) < 0.07); if (j >= 0) { used.add(j); hit++; } }
  const f = (2 * hit) / (r.beats.length + ref.length);
  beatF += f;
  if (good) ok++; else if (octave) oct++;
  console.log(`${name.padEnd(28)} ${r.bpm.toFixed(1).padStart(6)} / ${t}  ${good ? 'ok' : octave ? 'octave' : 'WRONG'}  beatF ${f.toFixed(2)} meter ${r.meter}`);
}
console.log(`tempo ok ${ok}/${names.length}, octave errors ${oct}, beat F ${(beatF / names.length).toFixed(3)}`);
