// Precompute per-setup features for synthetic cases (known tuning/capo) and GuitarSet takes
// (standard, no capo), then search the detection weights.
import fs from 'node:fs';
import { buildEvents, candidateGroups, groupFeatures, scoreGroup, DETECT_WEIGHTS } from '../engine/fretboard.js';
import { cases } from './synthetic_cases.mjs';

const [gsPath, every = '4'] = process.argv.slice(2);
const STD = [40, 45, 50, 55, 59, 64];
const openKey = (t, c) => t.map((m) => m + c).join(',');

function prep(notes, trueTuning, trueCapo, name) {
  const events = buildEvents(notes);
  const groups = candidateGroups(notes);
  const nEv = events.length;
  const ranges = nEv > 300 ? [[0, 75], [Math.floor(nEv / 3), Math.floor(nEv / 3) + 75], [Math.floor(2 * nEv / 3), Math.floor(2 * nEv / 3) + 75], [nEv - 75, nEv]] : [[0, nEv]];
  for (const g of groups) g.f = groupFeatures(notes, events, g, { ranges });
  return { name, groups, trueKey: openKey(trueTuning, trueCapo), trueExact: trueTuning.join() + '|' + trueCapo };
}

const t0 = Date.now();
const sets = [];
for (const c of cases()) sets.push({ ...prep(c.notes, c.tuning, c.capo, c.name), kind: 'syn' });
const gs = JSON.parse(fs.readFileSync(gsPath, 'utf8'));
const names = Object.keys(gs).filter((_, i) => i % Number(every) === 0);
for (const n of names) {
  const notes = gs[n].notes.map(([t, dur, v]) => ({ t0: t, t1: t + dur, pitch: Math.round(v), amp: 0.5 }));
  sets.push({ ...prep(notes, STD, 0, n), kind: 'gs' });
}
console.log('features computed in', ((Date.now() - t0) / 1000).toFixed(1), 's for', sets.length, 'sets');

function evaluate(w, verbose = false) {
  let syn = 0, nSyn = 0, gsOk = 0, nGs = 0;
  for (const s of sets) {
    let best = null;
    for (const g of s.groups) { const sc = scoreGroup(g.f, g.prior, w); if (!best || sc < best.sc) best = { g, sc }; }
    const m = best.g.members[0];
    const exact = m.tuning.join() + '|' + m.capo === s.trueExact;
    const eq = best.g.key === s.trueKey;
    if (s.kind === 'syn') { nSyn++; syn += exact ? 1 : eq ? 0.6 : 0; if (verbose) console.log(`  ${s.name.padEnd(32)} ${exact ? 'EXACT' : eq ? 'equiv' : 'WRONG'} -> ${m.tuning.join(',')} capo ${m.capo}`); }
    else { nGs++; if (exact) gsOk++; else if (verbose && nGs < 400) console.log(`  gs miss ${s.name} -> ${m.tuning.join(',')} capo ${m.capo}`); }
  }
  return { total: syn / nSyn + gsOk / nGs, syn: syn / nSyn, gs: gsOk / nGs };
}

if (process.argv.includes('--grid')) {
  const res = [];
  for (const cost of [1, 2, 4, 8, 16])
    for (const prior of [0.35, 0.5, 0.7, 1.0, 1.4])
      for (const open of [0, 1, 2])
        for (const openTime of [0, 1, 2])
          for (const excess of [0, 2, 4, 6, 8, 12, 16])
            for (const theta of [0.25, 0.35, 0.45, 0.55]) {
              const w = { cost, prior, open, openTime, drop: 4, excess, theta };
              res.push({ w, ...evaluate(w) });
            }
  // pareto front on (syn, gs)
  res.sort((a, b) => b.syn - a.syn || b.gs - a.gs);
  let bestGs = -1;
  for (const r of res) if (r.gs > bestGs) { bestGs = r.gs; console.log(`syn ${r.syn.toFixed(2)} gs ${r.gs.toFixed(3)} ${JSON.stringify(r.w)}`); }
  process.exit(0);
}
let best = { ...DETECT_WEIGHTS };
let bs = evaluate(best);
console.log('start', JSON.stringify(bs));
const grid = {
  cost: [0.5, 1, 2, 4, 6, 8, 12, 16, 24],
  prior: [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4],
  open: [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6],
  openTime: [0, 0.5, 1, 2, 3],
  drop: [2, 4, 8, 16, 32],
};
for (let round = 0; round < 6; round++) {
  for (const [k, vals] of Object.entries(grid)) {
    for (const v of vals) {
      const c = { ...best, [k]: v };
      const s = evaluate(c);
      if (s.total > bs.total + 1e-9) { best = c; bs = s; console.log(`  ${k}=${v} -> ${JSON.stringify(s)}`); }
    }
  }
}
console.log('best', JSON.stringify(best), JSON.stringify(bs));
evaluate(best, true);
