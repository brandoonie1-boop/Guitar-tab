// Beat / bar evaluation of the Beat This! path on GuitarSet + synthetic band (the files
// tests/eval_tempo.mjs uses), against the annotated beats and downbeats.
// usage: node tests/eval_beats.mjs [n] [--clean] [--variable]
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { beatActivations, beatsAndBars } from '../engine/beatthis.js';
import { addBand } from './gs_band.mjs';

const n = Number(process.argv[2] || 40);
const clean = process.argv.includes('--clean');
const variable = process.argv.includes('--variable');
const ANN = JSON.parse(fs.readFileSync('/home/claude/work/data/gs/guitarset_annotations.json', 'utf8'));
const names = Object.keys(ANN).filter((_, i) => i % 5 === 0).slice(0, n);
const ortMod = await import('/home/claude/work/jsenv/node_modules/onnxruntime-node/dist/index.js');
const ort = ortMod.default || ortMod;
const session = await ort.InferenceSession.create(path.join(path.dirname(new URL(import.meta.url).pathname), '../models/beat_this_small.f16.onnx'), { intraOpNumThreads: 2 });
const CACHE = `/home/claude/work/data/gs/btcache${clean ? '_clean' : ''}`;
fs.mkdirSync(CACHE, { recursive: true });

function fmeas(est, ref, tol = 0.07) {
  const used = new Set();
  let hit = 0;
  for (const b of est) {
    const j = ref.findIndex((q, k) => !used.has(k) && Math.abs(q - b) < tol);
    if (j >= 0) { used.add(j); hit++; }
  }
  return (2 * hit) / Math.max(1, est.length + ref.length);
}

let ok = 0, oct = 0, bF = 0, dF = 0, meterOk = 0;
for (const name of names) {
  const cp = path.join(CACHE, name + '.json');
  let act;
  if (fs.existsSync(cp)) {
    const c = JSON.parse(fs.readFileSync(cp, 'utf8'));
    act = { beat: Float32Array.from(c.beat), down: Float32Array.from(c.down), fps: c.fps, T: c.T };
  } else {
    let x = decodeWav(fs.readFileSync(path.join('/home/claude/work/data/guitarset/audio', `${name}_mic.wav`))).channels[0];
    if (!clean) x = addBand(x, ANN[name].beats);
    act = await beatActivations(ort, session, x, 44100);
    fs.writeFileSync(cp, JSON.stringify({ beat: Array.from(act.beat), down: Array.from(act.down), fps: act.fps, T: act.T }));
  }
  const r = beatsAndBars(act, { variable });
  const t = ANN[name].tempo;
  const ratio = r.bpm / t;
  const good = Math.abs(ratio - 1) < 0.04;
  const octave = Math.abs(ratio - 0.5) < 0.03 || Math.abs(ratio - 2) < 0.08;
  const ref = ANN[name].beats.map((b) => b[0]);
  const refD = ANN[name].beats.filter((b) => b[1] === 1).map((b) => b[0]);
  const trueMeter = Math.max(...ANN[name].beats.map((b) => b[1]));
  const f = fmeas(r.beats, ref), fd = fmeas(r.downbeats, refD);
  bF += f; dF += fd;
  if (good) ok++; else if (octave) oct++;
  if (r.meter === trueMeter) meterOk++;
  console.log(`${name.padEnd(28)} ${r.bpm.toFixed(1).padStart(6)} / ${t}  ${good ? 'ok' : octave ? 'octave' : 'WRONG'}  beatF ${f.toFixed(2)} downbeatF ${fd.toFixed(2)} meter ${r.meter} (true ${trueMeter}) bars ${r.bars.map((b) => b.len).join('')}`.slice(0, 160));
}
console.log(`tempo ok ${ok}/${names.length}, octave errors ${oct}, beat F ${(bF / names.length).toFixed(3)}, downbeat F ${(dF / names.length).toFixed(3)}, meter ok ${meterOk}/${names.length}`);
