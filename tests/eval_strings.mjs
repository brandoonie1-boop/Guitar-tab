// Tab accuracy after note detection: notes found, string/fret of the found notes, and notes
// that come out completely right (pitch, onset, string and fret), with and without the sound
// of each note weighing in (engine/strings.js). Notes come from cached note-model activations
// (tests/cache_activations.mjs), so this runs in seconds.
//
//   GuitarSet, each player judged by a string model that never heard them
//   (fold_<player>.json from: python tools/train_strings.py cvfit <folds dir> ...):
//     node tests/eval_strings.mjs guitarset <annotations.json> <audio dir> <activations dir> [folds dir] [--every k] [--detect]
//   EGSet12 (electric guitar, not used for training):
//     node tests/eval_strings.mjs egset <annotations.json> <audio dir> <activations dir> [model.json] [--detect]
//
// Without a folds dir / model the notes are fingered from the notes alone (the old way).
// --detect also works out tuning and capo (otherwise standard tuning, no capo, is given).
// --direct skips learning the guitar's strings from the recording (stiffnessEvidence).
// --true-notes fingers the annotated notes instead of detected ones.
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { hrNotes } from '../engine/hrgt.js';
import { cleanNotes } from '../engine/notes.js';
import { rankSetups, assignFingering, buildEvents } from '../engine/fretboard.js';
import { stringModel, attachStringEvidence, placeLogProbs } from '../engine/strings.js';
import { applySetup, DEFAULT_OPTS } from '../engine/pipeline.js';

const STD = [40, 45, 50, 55, 59, 64];
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1] === '--every'));
const [kind, annPath, audioDir, actDir, modelArg] = pos;
const every = Number(opt('--every', 1));
const ann = JSON.parse(fs.readFileSync(annPath, 'utf8'));
const names = Object.keys(ann).sort().filter((n, i) => i % every === 0 && (args.includes('--true-notes') || fs.existsSync(path.join(actDir, n + '.f32'))));

const models = {};
function modelFor(name) {
  if (!modelArg) return null;
  const file = kind === 'guitarset' ? path.join(modelArg, `fold_${name.slice(0, 2)}.json`) : modelArg;
  return (models[file] ||= stringModel(JSON.parse(fs.readFileSync(file, 'utf8'))));
}

function activations(name) {
  const all = new Float32Array(fs.readFileSync(path.join(actDir, name + '.f32')).buffer.slice(0));
  const T = all.length / (88 * 4);
  const sl = (i) => all.subarray(i * T * 88, (i + 1) * T * 88);
  return { T, onset: sl(0), offset: sl(1), frame: sl(2), velocity: sl(3) };
}

// mir_eval-style onset matching: same pitch, onsets within 50 ms, maximum matching
function matchNotes(est, ref, tol = 0.05) {
  const ri = new Int32Array(ref.length).fill(-1), ej = new Int32Array(est.length).fill(-1);
  const byPitch = new Map();
  ref.forEach((r, i) => { if (!byPitch.has(r.pitch)) byPitch.set(r.pitch, { r: [], e: [] }); byPitch.get(r.pitch).r.push(i); });
  est.forEach((e, j) => { const g = byPitch.get(e.pitch); if (g) g.e.push(j); });
  for (const { r, e } of byPitch.values()) {
    const adj = r.map((i) => e.filter((j) => Math.abs(est[j].t0 - ref[i].t0) <= tol));
    const aug = (k, seen) => {
      for (const j of adj[k]) {
        if (seen.has(j)) continue;
        seen.add(j);
        if (ej[j] < 0 || aug(r.indexOf(ej[j]), seen)) { ej[j] = r[k]; ri[r[k]] = j; return true; }
      }
      return false;
    };
    for (let k = 0; k < r.length; k++) aug(k, new Set());
  }
  return { ri, ej };
}

let TP = 0, NE = 0, NR = 0, strOk = 0, tabOk = 0, setupOk = 0;
for (const name of names) {
  const model = modelFor(name);
  // --true-notes: finger the annotated notes themselves (the string choice on its own)
  const notes = args.includes('--true-notes')
    ? ann[name].notes.map(([t, d, v], i) => ({ id: 'x:' + i, t0: t, t1: t + d, pitch: Math.round(v), amp: 0.5 }))
    : cleanNotes(hrNotes(activations(name), { onsetThreshold: DEFAULT_OPTS.hrOnset, offsetThreshold: DEFAULT_OPTS.hrOffset, frameThreshold: DEFAULT_OPTS.hrFrame })
      .map((n) => ({ ...n, bends: [], hr: true, conf: n.onsetProb, onsetProb: 0.5, support: 1 })), { idPrefix: 'x:', mergeFragments: false, mergeRetrigger: false });
  if (model) {
    const w = decodeWav(fs.readFileSync(path.join(audioDir, kind === 'guitarset' ? `${name}_mic.wav` : `${name}.wav`)));
    const mono = w.channels.length > 1 ? Float32Array.from(w.channels[0], (v, i) => 0.5 * (v + w.channels[1][i])) : w.channels[0];
    attachStringEvidence(notes, w.sr === 44100 ? mono : resample(mono, w.sr, 44100), 0, model);
  }
  const setup = args.includes('--detect') ? rankSetups([notes], { stringModel: model }).ranked[0].setup : { tuning: STD, capo: 0 };
  if (args.includes('--direct')) {
    const fa = assignFingering(notes, buildEvents(notes), setup, { beam: 48, audio: model ? placeLogProbs(model, notes, setup) : null });
    notes.forEach((n, i) => { const a = fa.assign.get(i); n.string = a ? a.string : null; n.fret = a ? a.fret : null; });
  } else applySetup({ notes }, setup, model);
  const sOk = setup.capo === 0 && setup.tuning.join() === STD.join();
  const ref = ann[name].notes.map(([t, d, v, s]) => ({ t0: t, pitch: Math.round(v), string: s, fret: Math.round(v) - STD[s] }));
  const { ri } = matchNotes(notes, ref);
  ref.forEach((r, i) => {
    if (ri[i] < 0) return;
    TP++;
    const n = notes[ri[i]];
    if (n.string === r.string) { strOk++; if (sOk && n.fret === r.fret) tabOk++; }
  });
  NE += notes.length; NR += ref.length; setupOk += sOk ? 1 : 0;
}
const f1 = (tp) => (200 * tp) / (NE + NR);
console.log(`${kind}: ${names.length} takes, ${modelArg ? 'with' : 'without'} the sound of each note${args.includes('--direct') ? ' (no adaptation)' : ''}`);
console.log(`  notes found: P ${(100 * TP / NE).toFixed(1)} R ${(100 * TP / NR).toFixed(1)} F1 ${f1(TP).toFixed(1)}`);
console.log(`  right string and fret (of the notes found): ${(100 * strOk / TP).toFixed(1)}%`);
console.log(`  notes completely right (F1): ${f1(tabOk).toFixed(1)}${args.includes('--detect') ? `   tuning and capo right: ${setupOk}/${names.length}` : ''}`);
