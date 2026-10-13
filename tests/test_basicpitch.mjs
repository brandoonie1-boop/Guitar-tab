// Compare the JS basic-pitch runner + note creation against the Python implementation.
import fs from 'node:fs';
import { runModel, outputToNotes, addPitchBends } from '../engine/basicpitch.js';

const [prefix, onnxPath] = process.argv.slice(2);
const ort = await import('/home/claude/work/jsenv/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs');
ort.env.wasm.numThreads = 1;
const readF32 = (p) => { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const audio = readF32(prefix + '_audio22k.f32');
const session = await ort.InferenceSession.create(fs.readFileSync(onnxPath));
let t0 = Date.now();
const model = await runModel(ort, session, audio);
console.log('model', Date.now() - t0, 'ms; frames', model.nT);
for (const k of ['note', 'onset', 'contour']) {
  const ref = readF32(`${prefix}_${k}.f32`);
  const mine = model[k];
  let maxErr = 0;
  if (ref.length !== mine.length) console.log(k, 'LENGTH MISMATCH', ref.length, mine.length);
  for (let i = 0; i < Math.min(ref.length, mine.length); i++) maxErr = Math.max(maxErr, Math.abs(ref[i] - mine[i]));
  console.log(k, 'max abs diff', maxErr.toExponential(2));
}
// feed the python unwrapped outputs into the JS note creation for an exact comparison
const pyModel = { nT: model.nT, note: readF32(prefix + '_note.f32'), onset: readF32(prefix + '_onset.f32'), contour: readF32(prefix + '_contour.f32') };
t0 = Date.now();
const notes = addPitchBends(pyModel, outputToNotes(pyModel));
console.log('note creation', Date.now() - t0, 'ms');
const ref = JSON.parse(fs.readFileSync(prefix + '_notes.json', 'utf8'));
const key = (a) => `${a[0]}-${a[1]}-${a[2]}`;
const refSet = new Map(ref.map((r) => [key(r), r]));
let match = 0, bendMatch = 0, ampErr = 0;
for (const n of notes) {
  const r = refSet.get(key([n.s, n.e, n.pitch]));
  if (r) {
    match++;
    ampErr = Math.max(ampErr, Math.abs(r[3] - n.amp));
    if (JSON.stringify(r[4]) === JSON.stringify(n.bends)) bendMatch++;
  }
}
console.log('notes js', notes.length, 'py', ref.length, 'exact matches', match, 'bend matches', bendMatch, 'max amp err', ampErr.toExponential(2));
// also with JS model outputs
const notes2 = outputToNotes(model);
const m2 = notes2.filter((n) => refSet.has(key([n.s, n.e, n.pitch]))).length;
console.log('with JS model outputs: notes', notes2.length, 'matches', m2);
