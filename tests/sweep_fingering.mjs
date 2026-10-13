// Coordinate-descent sweep of fingering cost weights on a GuitarSet subset.
import fs from 'node:fs';
import { assignFingering, buildEvents, DEFAULT_COSTS } from '../engine/fretboard.js';

const [path, everyArg = '4', offsetArg = '0'] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(path, 'utf8'));
const STD = [40, 45, 50, 55, 59, 64];
const names = Object.keys(data).filter((_, i) => i % Number(everyArg) === Number(offsetArg));
const sets = names.map((name) => {
  const notes = data[name].notes.map(([t, dur, v, s]) => ({ t0: t, t1: t + dur, pitch: Math.round(v), amp: 0.5, trueString: s }));
  return { notes, events: buildEvents(notes) };
});

function evaluate(costs) {
  let tot = 0, ok = 0;
  for (const { notes, events } of sets) {
    const res = assignFingering(notes, events, { tuning: STD, capo: 0 }, { costs, beam: 24, maxAssign: 24 });
    notes.forEach((n, i) => { const a = res.assign.get(i); if (a && a.string === n.trueString) ok++; });
    tot += notes.length;
  }
  return ok / tot;
}

let best = { ...DEFAULT_COSTS, ...(process.env.START ? JSON.parse(process.env.START) : {}) };
let bestAcc = evaluate(best);
console.log('start', (bestAcc * 100).toFixed(2));
const grid = {
  fretted: [0, 0.04, 0.08, 0.15, 0.25],
  fretHeight: [0, 0.01, 0.02, 0.04, 0.07],
  move: [0.1, 0.2, 0.3, 0.45, 0.7, 1.0],
  moveBase: [0, 0.1, 0.2, 0.4, 0.8],
  reach: [3, 4],
  stretch: [0.2, 0.5, 1.0],
  cut: [0.5, 1, 2, 4, 8],
  cutTime: [0.15, 0.3, 0.6],
  stringGap: [0, 0.15, 0.35, 0.7],
  fingers: [1, 2.5, 5],
  highFret: [0.05, 0.15, 0.3],
};
for (let round = 0; round < 2; round++) {
  for (const [k, vals] of Object.entries(grid)) {
    for (const v of vals) {
      if (best[k] === v) continue;
      const c = { ...best, [k]: v };
      const acc = evaluate(c);
      if (acc > bestAcc + 1e-4) { best = c; bestAcc = acc; console.log(`  ${k}=${v} -> ${(acc * 100).toFixed(2)}`); }
    }
  }
  console.log('round', round, (bestAcc * 100).toFixed(2), JSON.stringify(best));
}
