// Tune the setup-detection cost profile.
// Objective: detect the right tuning+capo on synthetic cases, and never invent a capo /
// alternate tuning for GuitarSet takes (all standard tuning, no capo).
import fs from 'node:fs';
import { rankSetups, assignFingering, buildEvents, DETECT_COSTS, FINGER_COSTS } from '../engine/fretboard.js';
import { cases } from './synthetic_cases.mjs';

const [gsPath, every = '6'] = process.argv.slice(2);
const STD = [40, 45, 50, 55, 59, 64];
const gs = JSON.parse(fs.readFileSync(gsPath, 'utf8'));
const gsNames = Object.keys(gs).filter((_, i) => i % Number(every) === 0);
const gsSets = gsNames.map((n) => gs[n].notes.map(([t, dur, v]) => ({ t0: t, t1: t + dur, pitch: Math.round(v), amp: 0.5 })));
const syn = cases();
const openKey = (t, c) => t.map((m) => m + c).join(',');

function score(dc, verbose = false) {
  let synScore = 0;
  for (const c of syn) {
    const r = rankSetups(c.notes, { detectCosts: dc });
    const b = r.ranked[0].setup;
    const exact = b.tuning.join() === c.tuning.join() && b.capo === c.capo;
    const eq = openKey(b.tuning, b.capo) === openKey(c.tuning, c.capo);
    synScore += exact ? 1 : eq ? 0.6 : 0;
    if (verbose) console.log(`  ${c.name.padEnd(32)} ${exact ? 'EXACT' : eq ? 'equiv' : 'WRONG'}  got ${b.name} capo ${b.capo} (${(r.ranked[0].confidence * 100).toFixed(0)}%)`);
  }
  let gsOk = 0;
  for (const notes of gsSets) {
    const r = rankSetups(notes, { detectCosts: dc });
    const b = r.ranked[0].setup;
    if (b.capo === 0 && b.tuning.join() === STD.join()) gsOk++;
  }
  const gsRate = gsOk / gsSets.length;
  return { total: synScore / syn.length + gsRate, syn: synScore, gs: gsRate };
}

let best = { ...DETECT_COSTS, ...(process.env.START ? JSON.parse(process.env.START) : {}) };
let bs = score(best, true);
console.log('start', JSON.stringify(bs));
if (process.argv.includes('--eval')) process.exit(0);
const grid = {
  fretted: [0, 0.04, 0.08, 0.15, 0.25, 0.4],
  fretHeight: [0, 0.01, 0.02, 0.04, 0.07, 0.1],
  move: [0.1, 0.2, 0.3, 0.5, 0.8],
  moveBase: [0, 0.15, 0.3, 0.6],
  cut: [1, 2, 4, 8],
  stringGap: [0.1, 0.35, 0.7],
};
for (let round = 0; round < 2; round++) {
  for (const [k, vals] of Object.entries(grid)) {
    for (const v of vals) {
      if (best[k] === v) continue;
      const c = { ...best, [k]: v };
      const s = score(c);
      if (s.total > bs.total + 1e-6) { best = c; bs = s; console.log(`  ${k}=${v} -> ${JSON.stringify(s)}`); }
    }
  }
  console.log('round', round, JSON.stringify(bs), JSON.stringify(best));
}
score(best, true);
