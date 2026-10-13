import fs from 'node:fs';
import { analyzeRhythm } from '../engine/beats.js';

const readF32 = (p) => { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const inp = readF32(process.argv[2]);
const n = inp.length / 2;
const mono = new Float32Array(n);
for (let i = 0; i < n; i++) mono[i] = 0.5 * (inp[i] + inp[n + i]);
const t0 = Date.now();
const r = analyzeRhythm(mono, 44100);
console.log('time', Date.now() - t0, 'ms; bpm', r.bpm.toFixed(2), 'cands', r.bpmCandidates.map((b) => b.toFixed(1)).join(','), 'meter', r.meter, 'phase', r.downbeatPhase);
console.log('beats', r.beats.slice(0, 12).map((b) => b.toFixed(3)).join(' '));
