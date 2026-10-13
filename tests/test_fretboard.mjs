// Synthetic tests: build notes from a known tab, then check that tuning/capo detection
// and string assignment recover it.
import { rankSetups, assignFingering, buildEvents } from '../engine/fretboard.js';
import { tuningName } from '../engine/tunings.js';

// events: [[time, dur, [[stringNumber(6..1), fret], ...]], ...]  frets relative to capo
function notesFromTab(tuning, capo, events) {
  const notes = [];
  for (const [t, dur, frets] of events) {
    for (const [sn, f] of frets) {
      const s = 6 - sn;
      notes.push({ t0: t, t1: t + dur, pitch: tuning[s] + capo + f, amp: 0.6, trueString: s, trueFret: f });
    }
  }
  // a string can only ring one note: re-plucking a string cuts the previous note
  notes.sort((a, b) => a.t0 - b.t0);
  for (let i = 0; i < notes.length; i++) {
    for (let j = i + 1; j < notes.length; j++) {
      if (notes[j].trueString === notes[i].trueString && notes[j].t0 > notes[i].t0) { notes[i].t1 = Math.min(notes[i].t1, notes[j].t0); break; }
    }
  }
  return notes;
}

function repeatPattern(pattern, times, period) {
  const out = [];
  for (let k = 0; k < times; k++) for (const [t, d, f] of pattern) out.push([t + k * period, d, f]);
  return out;
}

const STD = [40, 45, 50, 55, 59, 64];
const DAEABE = [38, 45, 52, 57, 59, 64];
const DROPD = [38, 45, 50, 55, 59, 64];
const FACGCE = [41, 45, 48, 55, 60, 64];

const tests = [];
// 1. Brave Bird style riff: DAEABE capo 7, low string 7 -> 5 -> 3 (relative to capo) under ringing open strings
{
  const bar = (low) => [
    [0, 1.9, [[6, low]]], [0.25, 1.6, [[4, 0]]], [0.5, 1.5, [[3, 0]]], [0.75, 1.2, [[2, 0]]],
    [1.0, 1.0, [[1, 0]]], [1.25, 0.7, [[2, 0]]], [1.5, 0.5, [[3, 0]]], [1.75, 0.25, [[5, 0]]],
  ];
  const ev = [];
  [7, 5, 3, 5].forEach((low, i) => bar(low).forEach(([t, d, f]) => ev.push([t + i * 2, d, f])));
  tests.push({ name: 'DAEABE capo 7 riff', tuning: DAEABE, capo: 7, notes: notesFromTab(DAEABE, 7, repeatPattern(ev, 4, 8)) });
}
// 2. Standard open chords strummed C - G - Am - F + melody
{
  const C = [[5, 3], [4, 2], [3, 0], [2, 1], [1, 0]];
  const G = [[6, 3], [5, 2], [4, 0], [3, 0], [2, 0], [1, 3]];
  const Am = [[5, 0], [4, 2], [3, 2], [2, 1], [1, 0]];
  const F = [[6, 1], [5, 3], [4, 3], [3, 2], [2, 1], [1, 1]];
  const ev = [];
  [C, G, Am, F].forEach((ch, i) => { for (let k = 0; k < 4; k++) ev.push([i * 2 + k * 0.5, 0.45, ch]); });
  const mel = [[1, 0], [1, 1], [1, 3], [2, 3], [2, 1], [3, 2], [3, 0], [2, 1]];
  mel.forEach((n, i) => ev.push([8 + i * 0.25, 0.24, [n]]));
  tests.push({ name: 'Standard open chords', tuning: STD, capo: 0, notes: notesFromTab(STD, 0, repeatPattern(ev, 3, 10)) });
}
// 3. Standard capo 3 fingerpicking (G shape / C shape)
{
  const pat = [[0, 1.0, [[6, 3]]], [0.25, 0.7, [[3, 0]]], [0.5, 0.5, [[2, 0]]], [0.75, 0.25, [[1, 3]]],
    [1, 1.0, [[5, 3]]], [1.25, 0.7, [[3, 0]]], [1.5, 0.5, [[2, 1]]], [1.75, 0.25, [[1, 0]]],
    [2, 1.0, [[4, 0]]], [2.25, 0.7, [[3, 2]]], [2.5, 0.5, [[2, 3]]], [2.75, 0.25, [[1, 2]]]];
  tests.push({ name: 'Standard capo 3 fingerpicking', tuning: STD, capo: 3, notes: notesFromTab(STD, 3, repeatPattern(pat, 6, 3)) });
}
// 4. Drop D power chord riff
{
  const pc = (f) => [[6, f], [5, f], [4, f]];
  const pat = [[0, 0.2, pc(0)], [0.25, 0.2, pc(0)], [0.5, 0.2, pc(3)], [0.75, 0.2, pc(5)], [1.0, 0.45, pc(0)], [1.5, 0.45, pc(7)]];
  tests.push({ name: 'Drop D power chords', tuning: DROPD, capo: 0, notes: notesFromTab(DROPD, 0, repeatPattern(pat, 8, 2)) });
}
// 5. FACGCE twinkle (American Football style) capo 0
{
  const pat = [[0, 1.2, [[6, 0]]], [0.2, 1.0, [[3, 0]]], [0.4, 0.9, [[2, 0]]], [0.6, 0.8, [[1, 0]]], [0.8, 0.6, [[2, 3]]], [1.0, 0.5, [[3, 2]]],
    [1.2, 1.2, [[5, 0]]], [1.4, 1.0, [[3, 0]]], [1.6, 0.9, [[2, 2]]], [1.8, 0.8, [[1, 0]]], [2.0, 0.6, [[2, 0]]], [2.2, 0.5, [[4, 2]]]];
  tests.push({ name: 'FACGCE twinkle', tuning: FACGCE, capo: 0, notes: notesFromTab(FACGCE, 0, repeatPattern(pat, 6, 2.4)) });
}

for (const t of tests) {
  const t0 = Date.now();
  const res = rankSetups(t.notes);
  const ms = Date.now() - t0;
  const top = res.ranked.slice(0, 4).map((r) => `${tuningName(r.setup.tuning)} capo ${r.setup.capo} (${(r.confidence * 100).toFixed(0)}%${r.equivalents.length ? ', = ' + r.equivalents.map((e) => tuningName(e.tuning) + ' capo ' + e.capo).join(' / ') : ''})`);
  const best = res.ranked[0].setup;
  const trueKey = t.tuning.map((m) => m + t.capo).join(',');
  const gotKey = best.tuning.map((m) => m + best.capo).join(',');
  const exact = best.tuning.join(',') === t.tuning.join(',') && best.capo === t.capo;
  // string accuracy with the TRUE setup (isolates the fingering model)
  const events = buildEvents(t.notes);
  const fa = assignFingering(t.notes, events, { tuning: t.tuning, capo: t.capo });
  let ok = 0;
  t.notes.forEach((n, i) => { const a = fa.assign.get(i); if (a && a.string === n.trueString && a.fret === n.trueFret) ok++; });
  console.log(`\n${t.name}: ${ms} ms, ${t.notes.length} notes`);
  console.log('  detected:', exact ? 'EXACT' : gotKey === trueKey ? 'EQUIVALENT-SOUNDING' : 'WRONG', '|', top.join('  ||  '));
  console.log(`  string/fret accuracy with true setup: ${(100 * ok / t.notes.length).toFixed(1)}%`);
}
