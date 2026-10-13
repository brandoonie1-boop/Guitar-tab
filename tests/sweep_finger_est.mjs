// Fingering sweep on TRANSCRIBED notes (cache from cache_gs_notes.mjs): string accuracy on matched notes.
import fs from 'node:fs';
import { assignFingering, buildEvents, FINGER_COSTS } from '../engine/fretboard.js';

const [cachePath] = process.argv.slice(2);
const cache = JSON.parse(fs.readFileSync(cachePath, 'utf8'));
const STD = [40, 45, 50, 55, 59, 64];
const sets = Object.entries(cache).map(([name, d]) => {
  const ref = d.ref.map(([t, dur, v, s]) => ({ t0: t, pitch: Math.round(v), string: s }));
  const est = d.est;
  // match once
  const used = new Set();
  const match = new Map(); // est index -> ref string
  for (const r of ref) {
    let bi = -1, bd = 1;
    est.forEach((e, i) => { if (!used.has(i) && e.pitch === r.pitch) { const dd = Math.abs(e.t0 - r.t0); if (dd <= 0.05 && dd < bd) { bd = dd; bi = i; } } });
    if (bi >= 0) { used.add(bi); match.set(bi, r.string); }
  }
  return { name, est, match };
});

function evaluate(costs, ringScale = 1, verbose = false) {
  let ok = 0, tot = 0, okSolo = 0, totSolo = 0;
  for (const s of sets) {
    const notes = s.est.map((n) => ({ ...n, t1: n.t0 + (n.t1 - n.t0) * ringScale }));
    const events = buildEvents(notes);
    const r = assignFingering(notes, events, { tuning: STD, capo: 0 }, { costs, beam: 24, maxAssign: 24 });
    for (const [i, str] of s.match) {
      const a = r.assign.get(i);
      const good = a && a.string === str;
      tot++; if (good) ok++;
      if (s.name.includes('solo')) { totSolo++; if (good) okSolo++; }
    }
  }
  return { acc: ok / tot, solo: okSolo / Math.max(1, totSolo), comp: (ok - okSolo) / Math.max(1, tot - totSolo) };
}

let best = { ...FINGER_COSTS, ...(process.env.START ? JSON.parse(process.env.START) : {}) };
let bestScale = 1;
let bs = evaluate(best, bestScale);
console.log('start', JSON.stringify(bs));
const grid = {
  fretted: [0, 0.02, 0.04, 0.08, 0.15],
  fretHeight: [0, 0.005, 0.01, 0.02, 0.04],
  move: [0.05, 0.1, 0.2, 0.3, 0.5],
  moveBase: [0, 0.1, 0.2, 0.4],
  stretch: [0.1, 0.2, 0.5, 1],
  cut: [0.5, 1, 2, 4, 8],
  cutTime: [0.15, 0.3, 0.6],
  stringGap: [0, 0.15, 0.35, 0.7],
  highFret: [0.05, 0.15, 0.3],
};
for (let round = 0; round < 2; round++) {
  for (const sc of [0.7, 1, 1.4, 2]) {
    const s = evaluate(best, sc);
    if (s.acc > bs.acc + 1e-4) { bs = s; bestScale = sc; console.log(`  ringScale=${sc} -> ${JSON.stringify(s)}`); }
  }
  for (const [k, vals] of Object.entries(grid)) {
    for (const v of vals) {
      if (best[k] === v) continue;
      const c = { ...best, [k]: v };
      const s = evaluate(c, bestScale);
      if (s.acc > bs.acc + 1e-4) { best = c; bs = s; console.log(`  ${k}=${v} -> ${JSON.stringify(s)}`); }
    }
  }
  console.log('round', round, JSON.stringify(bs), 'ringScale', bestScale, JSON.stringify(best));
}
