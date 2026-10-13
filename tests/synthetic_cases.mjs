// Synthetic tab excerpts in different tunings / capo positions, for testing tuning+capo
// detection and fingering. Each case lists its true setup; notes carry their true string/fret.

export const STD = [40, 45, 50, 55, 59, 64];
export const DAEABE = [38, 45, 52, 57, 59, 64];
export const DROPD = [38, 45, 50, 55, 59, 64];
export const FACGCE = [41, 45, 48, 55, 60, 64];
export const OPEND = [38, 45, 50, 54, 57, 62];
export const DADGAD = [38, 45, 50, 55, 57, 62];
export const EB = [39, 44, 49, 54, 58, 63];

// deterministic pseudo random
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

// events: [[time, dur, [[stringNumber(6..1), fret], ...]], ...]  frets relative to capo
export function notesFromTab(tuning, capo, events, seed = 1, strum = 0) {
  const r = rng(seed);
  const notes = [];
  for (const [t, dur, frets] of events) {
    // strummed chords: low to high with a small spread
    const sorted = [...frets].sort((a, b) => b[0] - a[0]);
    sorted.forEach(([sn, f], k) => {
      const s = 6 - sn;
      const jitter = (r() - 0.5) * 0.012 + (strum ? k * strum : 0);
      notes.push({ t0: t + jitter, t1: t + jitter + dur * (0.85 + 0.3 * r()), pitch: tuning[s] + capo + f, amp: 0.35 + 0.4 * r(), trueString: s, trueFret: f });
    });
  }
  notes.sort((a, b) => a.t0 - b.t0);
  for (let i = 0; i < notes.length; i++) {
    for (let j = i + 1; j < notes.length; j++) {
      if (notes[j].trueString === notes[i].trueString && notes[j].t0 > notes[i].t0) { notes[i].t1 = Math.min(notes[i].t1, notes[j].t0); break; }
    }
  }
  return notes;
}

function rep(pattern, times, period) {
  const out = [];
  for (let k = 0; k < times; k++) for (const [t, d, f] of pattern) out.push([t + k * period, d, f]);
  return out;
}

export function cases() {
  const out = [];
  // 1. DAEABE capo 7: low string 7 -> 5 -> 3 under ringing open strings (Brave Bird style)
  {
    const bar = (low) => [
      [0, 1.9, [[6, low]]], [0.25, 1.6, [[4, 0]]], [0.5, 1.5, [[3, 0]]], [0.75, 1.2, [[2, 0]]],
      [1.0, 1.0, [[1, 0]]], [1.25, 0.7, [[2, 0]]], [1.5, 0.5, [[3, 0]]], [1.75, 0.25, [[5, 0]]],
    ];
    const ev = [];
    [7, 5, 3, 5].forEach((low, i) => bar(low).forEach(([t, d, f]) => ev.push([t + i * 2, d, f])));
    out.push({ name: 'DAEABE capo 7 riff', tuning: DAEABE, capo: 7, notes: notesFromTab(DAEABE, 7, rep(ev, 3, 8), 11) });
  }
  // 2. Standard open chords strummed + melody
  {
    const C = [[5, 3], [4, 2], [3, 0], [2, 1], [1, 0]];
    const G = [[6, 3], [5, 2], [4, 0], [3, 0], [2, 0], [1, 3]];
    const Am = [[5, 0], [4, 2], [3, 2], [2, 1], [1, 0]];
    const F = [[6, 1], [5, 3], [4, 3], [3, 2], [2, 1], [1, 1]];
    const ev = [];
    [C, G, Am, F].forEach((ch, i) => { for (let k = 0; k < 4; k++) ev.push([i * 2 + k * 0.5, 0.45, ch]); });
    const mel = [[1, 0], [1, 1], [1, 3], [2, 3], [2, 1], [3, 2], [3, 0], [2, 1]];
    mel.forEach((n, i) => ev.push([8 + i * 0.25, 0.24, [n]]));
    out.push({ name: 'Standard open chords', tuning: STD, capo: 0, notes: notesFromTab(STD, 0, rep(ev, 2, 10), 2, 0.008) });
  }
  // 3. Standard capo 3 travis picking (G / C / D shapes)
  {
    const pat = [[0, 1.0, [[6, 3]]], [0.25, 0.7, [[3, 0]]], [0.5, 0.9, [[4, 0]]], [0.75, 0.5, [[2, 0]]], [1, 0.8, [[6, 3]]], [1.25, 0.5, [[1, 3]]], [1.5, 0.9, [[4, 0]]], [1.75, 0.4, [[2, 0]]],
      [2, 1.0, [[5, 3]]], [2.25, 0.7, [[3, 0]]], [2.5, 0.9, [[4, 2]]], [2.75, 0.5, [[2, 1]]], [3, 0.8, [[5, 3]]], [3.25, 0.5, [[1, 0]]], [3.5, 0.9, [[4, 2]]], [3.75, 0.4, [[2, 1]]],
      [4, 1.0, [[4, 0]]], [4.25, 0.7, [[3, 2]]], [4.5, 0.9, [[5, 0]]], [4.75, 0.5, [[2, 3]]], [5, 0.8, [[4, 0]]], [5.25, 0.5, [[1, 2]]], [5.5, 0.9, [[5, 0]]], [5.75, 0.4, [[2, 3]]]];
    out.push({ name: 'Standard capo 3 travis picking', tuning: STD, capo: 3, notes: notesFromTab(STD, 3, rep(pat, 3, 6), 3) });
  }
  // 4. Drop D power chord riff
  {
    const pc = (f) => [[6, f], [5, f], [4, f]];
    const pat = [[0, 0.2, pc(0)], [0.25, 0.2, pc(0)], [0.5, 0.2, pc(3)], [0.75, 0.2, pc(5)], [1.0, 0.45, pc(0)], [1.5, 0.45, pc(7)]];
    out.push({ name: 'Drop D power chords', tuning: DROPD, capo: 0, notes: notesFromTab(DROPD, 0, rep(pat, 6, 2), 4, 0.006) });
  }
  // 5. FACGCE twinkle, clusters of open strings with a few fretted notes
  {
    const pat = [[0, 1.6, [[6, 0]]], [0.2, 1.4, [[3, 0]]], [0.4, 1.2, [[2, 0]]], [0.6, 1.0, [[1, 0]]], [0.8, 0.8, [[4, 0]]], [1.0, 0.6, [[2, 2]]],
      [1.2, 1.6, [[5, 0]]], [1.4, 1.4, [[3, 2]]], [1.6, 1.2, [[2, 0]]], [1.8, 1.0, [[1, 0]]], [2.0, 0.8, [[4, 0]]], [2.2, 0.6, [[2, 3]]],
      [2.4, 1.6, [[6, 5]]], [2.6, 1.4, [[3, 0]]], [2.8, 1.2, [[2, 0]]], [3.0, 1.0, [[1, 0]]], [3.2, 0.8, [[4, 5]]], [3.4, 0.6, [[1, 3]]]];
    out.push({ name: 'FACGCE twinkle', tuning: FACGCE, capo: 0, notes: notesFromTab(FACGCE, 0, rep(pat, 3, 3.6), 5) });
  }
  // 6. Open D strumming: D / G (barre 5) / A (barre 7) with open drones + melody on top strings
  {
    const bar = (f) => [[6, f], [5, f], [4, f], [3, f], [2, f], [1, f]];
    const ev = [];
    [0, 5, 7, 5].forEach((f, i) => { ev.push([i * 2, 0.9, bar(f)]); ev.push([i * 2 + 1, 0.9, [[6, 0], [5, 0], [4, f], [3, f], [2, f], [1, f]]]); });
    [[1, 2], [1, 4], [1, 5], [1, 4], [1, 2], [1, 0], [2, 3], [2, 0]].forEach((n, i) => ev.push([8 + i * 0.25, 0.5, [n, [6, 0]]]));
    out.push({ name: 'Open D strumming', tuning: OPEND, capo: 0, notes: notesFromTab(OPEND, 0, rep(ev, 2, 10), 6, 0.01) });
  }
  // 7. Standard tuning twinkly lead: high frets on strings 1-3 against open B/E
  {
    const pat = [[0, 0.6, [[1, 12]]], [0.15, 0.6, [[2, 0]]], [0.3, 0.6, [[1, 14]]], [0.45, 0.6, [[2, 0]]], [0.6, 0.6, [[1, 15]]], [0.75, 0.6, [[1, 0]]], [0.9, 0.6, [[2, 12]]], [1.05, 0.6, [[1, 0]]],
      [1.2, 0.6, [[3, 14]]], [1.35, 0.6, [[2, 0]]], [1.5, 0.6, [[3, 12]]], [1.65, 0.6, [[1, 0]]], [1.8, 0.6, [[2, 14]]], [1.95, 0.6, [[1, 0]]], [2.1, 0.6, [[2, 15]]], [2.25, 0.6, [[1, 0]]]];
    out.push({ name: 'Standard twinkly lead', tuning: STD, capo: 0, notes: notesFromTab(STD, 0, rep(pat, 4, 2.4), 7) });
  }
  // 8. Standard capo 2 open-shape strumming G C D Em
  {
    const G = [[6, 3], [5, 2], [4, 0], [3, 0], [2, 3], [1, 3]];
    const C = [[5, 3], [4, 2], [3, 0], [2, 1], [1, 0]];
    const D = [[4, 0], [3, 2], [2, 3], [1, 2]];
    const Em = [[6, 0], [5, 2], [4, 2], [3, 0], [2, 0], [1, 0]];
    const ev = [];
    [G, C, D, Em].forEach((ch, i) => { [0, 0.5, 0.75, 1.25, 1.5].forEach((o) => ev.push([i * 2 + o, 0.3, ch])); });
    out.push({ name: 'Standard capo 2 strumming', tuning: STD, capo: 2, notes: notesFromTab(STD, 2, rep(ev, 2, 8), 8, 0.007) });
  }
  // 9. DADGAD capo 2 fingerstyle with drones
  {
    const pat = [[0, 1.5, [[6, 0]]], [0.25, 1.2, [[3, 0]]], [0.5, 1.0, [[2, 0]]], [0.75, 0.8, [[1, 0]]], [1.0, 0.6, [[2, 2]]], [1.25, 0.5, [[1, 2]]], [1.5, 0.5, [[2, 0]]], [1.75, 0.5, [[3, 2]]],
      [2, 1.5, [[5, 0]]], [2.25, 1.2, [[4, 0]]], [2.5, 1.0, [[3, 4]]], [2.75, 0.8, [[2, 0]]], [3.0, 0.6, [[1, 0]]], [3.25, 0.5, [[2, 3]]], [3.5, 0.5, [[3, 0]]], [3.75, 0.5, [[4, 2]]]];
    out.push({ name: 'DADGAD capo 2 fingerstyle', tuning: DADGAD, capo: 2, notes: notesFromTab(DADGAD, 2, rep(pat, 3, 4), 9) });
  }
  // 10. Eb standard riff with low open strings
  {
    const pat = [[0, 0.2, [[6, 0]]], [0.25, 0.2, [[6, 0]]], [0.5, 0.2, [[6, 3]]], [0.75, 0.2, [[5, 0]]], [1.0, 0.2, [[5, 2]]], [1.25, 0.2, [[6, 0]]], [1.5, 0.2, [[4, 0]]], [1.75, 0.2, [[4, 2]]],
      [2, 0.4, [[6, 0], [5, 2], [4, 2]]], [2.5, 0.4, [[6, 3], [5, 5], [4, 5]]], [3, 0.4, [[6, 5], [5, 7], [4, 7]]], [3.5, 0.4, [[5, 0], [4, 2], [3, 2]]]];
    out.push({ name: 'Eb standard riff', tuning: EB, capo: 0, notes: notesFromTab(EB, 0, rep(pat, 4, 4), 10, 0.006) });
  }
  return out;
}
