// Evaluate the fingering model on GuitarSet ground-truth notes (real players' string choices).
// usage: node tests/eval_fingering_gs.mjs <guitarset_annotations.json> [maxFiles] [detect]
import fs from 'node:fs';
import { assignFingering, buildEvents, rankSetups, DEFAULT_COSTS } from '../engine/fretboard.js';

const [path, maxFilesArg, mode] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(path, 'utf8'));
const STD = [40, 45, 50, 55, 59, 64];
const names = Object.keys(data).slice(0, Number(maxFilesArg || 1e9));
const costOverride = process.env.COSTS ? JSON.parse(process.env.COSTS) : {};

let tot = 0, ok = 0, okString = 0;
const byStyle = {};
let detOk = 0, detEq = 0, detN = 0;
const t0 = Date.now();
for (const name of names) {
  const d = data[name];
  const notes = d.notes.map(([t, dur, v, s]) => ({ t0: t, t1: t + dur, pitch: Math.round(v), amp: 0.5, trueString: s }));
  const events = buildEvents(notes);
  const res = assignFingering(notes, events, { tuning: STD, capo: 0 }, { costs: costOverride, beam: 48 });
  let fOk = 0;
  notes.forEach((n, i) => {
    const a = res.assign.get(i);
    if (a && a.string === n.trueString) fOk++;
  });
  tot += notes.length; ok += fOk;
  const style = name.split('_')[1].replace(/\d.*/, '') + '_' + name.split('_').pop();
  byStyle[style] = byStyle[style] || [0, 0];
  byStyle[style][0] += fOk; byStyle[style][1] += notes.length;
  if (mode === 'detect') {
    const r = rankSetups(notes);
    const b = r.ranked[0].setup;
    detN++;
    if (b.capo === 0 && b.tuning.join() === STD.join()) detOk++;
    else if (b.tuning.map((m) => m + b.capo).join() === STD.join()) detEq++;
    else console.log('  detect miss', name, b.name, 'capo', b.capo, (r.ranked[0].confidence * 100).toFixed(0) + '%');
  }
}
console.log(`files ${names.length}, notes ${tot}, string accuracy ${(100 * ok / tot).toFixed(2)}%  (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
for (const [k, [a, b]] of Object.entries(byStyle).sort()) console.log(`  ${k.padEnd(10)} ${(100 * a / b).toFixed(1)}%`);
if (mode === 'detect') console.log(`detection: exact ${detOk}/${detN}, equivalent ${detEq}/${detN}`);
