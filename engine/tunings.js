// Guitar tunings library. Pitches are MIDI numbers, listed low string (6) -> high string (1).

import { midiName } from './dsp.js';

export const STANDARD = [40, 45, 50, 55, 59, 64];

// prior: rough popularity weight (higher = more common). Used only to break near-ties.
export const TUNINGS = [
  { id: 'std', name: 'Standard', notes: [40, 45, 50, 55, 59, 64], prior: 1.0 },
  { id: 'dropd', name: 'Drop D', notes: [38, 45, 50, 55, 59, 64], prior: 0.55 },
  { id: 'eb', name: 'Half step down (Eb)', notes: [39, 44, 49, 54, 58, 63], prior: 0.35 },
  { id: 'dstd', name: 'Whole step down (D standard)', notes: [38, 43, 48, 53, 57, 62], prior: 0.2 },
  { id: 'dropcs', name: 'Drop C#', notes: [37, 44, 49, 54, 58, 63], prior: 0.12 },
  { id: 'dropc', name: 'Drop C', notes: [36, 43, 48, 53, 57, 62], prior: 0.15 },
  { id: 'doubledropd', name: 'Double drop D (DADGBD)', notes: [38, 45, 50, 55, 59, 62], prior: 0.12 },
  { id: 'dadgad', name: 'DADGAD', notes: [38, 45, 50, 55, 57, 62], prior: 0.15 },
  { id: 'opend', name: 'Open D (DADF#AD)', notes: [38, 45, 50, 54, 57, 62], prior: 0.2 },
  { id: 'opendm', name: 'Open D minor (DADFAD)', notes: [38, 45, 50, 53, 57, 62], prior: 0.06 },
  { id: 'openg', name: 'Open G (DGDGBD)', notes: [38, 43, 50, 55, 59, 62], prior: 0.15 },
  { id: 'openc', name: 'Open C (CGCGCE)', notes: [36, 43, 48, 55, 60, 64], prior: 0.08 },
  { id: 'opene', name: 'Open E (EBEG#BE)', notes: [40, 47, 52, 56, 59, 64], prior: 0.08 },
  { id: 'opena', name: 'Open A (EAEAC#E)', notes: [40, 45, 52, 57, 61, 64], prior: 0.06 },
  { id: 'facgce', name: 'FACGCE', notes: [41, 45, 48, 55, 60, 64], prior: 0.18 },
  { id: 'daeabe', name: 'DAEABE', notes: [38, 45, 52, 57, 59, 64], prior: 0.16 },
  { id: 'eaeabe', name: 'EAEABE', notes: [40, 45, 52, 57, 59, 64], prior: 0.06 },
  { id: 'cgdgbe', name: 'CGDGBE', notes: [36, 43, 50, 55, 59, 64], prior: 0.07 },
  { id: 'csus2', name: 'Open Csus2 (CGDGCD)', notes: [36, 43, 50, 55, 60, 62], prior: 0.06 },
  { id: 'dgdgcd', name: 'DGDGCD', notes: [38, 43, 50, 55, 60, 62], prior: 0.04 },
  { id: 'c6', name: 'Open C6 (CACGCE)', notes: [36, 45, 48, 55, 60, 64], prior: 0.06 },
  { id: 'dadead', name: 'DADEAD', notes: [38, 45, 50, 52, 57, 62], prior: 0.04 },
  { id: 'cgdgad', name: 'CGDGAD', notes: [36, 43, 50, 55, 57, 62], prior: 0.05 },
  { id: 'eadeae', name: 'EADEAE', notes: [40, 45, 50, 52, 57, 64], prior: 0.03 },
  { id: 'dadfsad_low', name: 'Open C# (C#G#C#FG#C#)', notes: [37, 44, 49, 53, 56, 61], prior: 0.03 },
  { id: 'nashville', name: 'B standard', notes: [35, 40, 45, 50, 54, 59], prior: 0.05 },
  { id: 'dropb', name: 'Drop B', notes: [35, 42, 47, 52, 56, 61], prior: 0.05 },
];

export function tuningLabel(notes) {
  // e.g. "D A E A B E"
  return notes.map((m) => midiName(m, false)).join(' ');
}

export function tuningCompact(notes) {
  return notes.map((m) => midiName(m, false)).join('');
}

export function tuningFull(notes) {
  return notes.map((m) => midiName(m, true)).join(' ');
}

export function findTuning(notes) {
  const key = notes.join(',');
  return TUNINGS.find((t) => t.notes.join(',') === key) || null;
}

export function tuningName(notes) {
  const t = findTuning(notes);
  if (t) return t.name;
  // transposed versions of library shapes
  for (const lib of TUNINGS) {
    const d = notes[0] - lib.notes[0];
    if (d !== 0 && notes.every((m, i) => m - lib.notes[i] === d)) {
      const dir = d < 0 ? 'down' : 'up';
      return `${lib.name}, ${Math.abs(d)} semitone${Math.abs(d) > 1 ? 's' : ''} ${dir}`;
    }
  }
  return tuningCompact(notes);
}

// How far each string is from standard; tunings outside a realistic range are rejected.
export function isPlausible(notes) {
  for (let i = 0; i < 6; i++) {
    const d = notes[i] - STANDARD[i];
    if (d < -7 || d > 2) return false;
  }
  for (let i = 1; i < 6; i++) if (notes[i] <= notes[i - 1] - 3) return false; // no big crossings
  return true;
}

// Candidate (tuning, capo) setups to test. Includes transposed library shapes.
export function candidateSetups(opts = {}) {
  const maxCapo = opts.maxCapo ?? 10;
  const out = [];
  const seen = new Set();
  const capos = opts.capo != null ? [opts.capo] : Array.from({ length: maxCapo + 1 }, (_, i) => i);
  const shapes = opts.tuning ? [{ id: 'custom', name: tuningName(opts.tuning), notes: opts.tuning, prior: 1 }] : TUNINGS;
  for (const t of shapes) {
    const transposes = opts.tuning ? [0] : [-2, -1, 0, 1];
    for (const tr of transposes) {
      const notes = t.notes.map((m) => m + tr);
      if (!isPlausible(notes)) continue;
      // a transposed shape that is itself a library tuning gets added as that entry instead
      if (tr !== 0 && !opts.tuning && findTuning(notes)) continue;
      const key = notes.join(',');
      if (seen.has(key)) continue;
      seen.add(key);
      const prior = t.prior * (tr === 0 ? 1 : 0.18 / (1 + Math.abs(tr)));
      for (const capo of capos) {
        out.push({ tuning: notes, capo, tuningId: tr === 0 ? t.id : `${t.id}${tr > 0 ? '+' : ''}${tr}`, name: tuningName(notes), prior: prior * capoPrior(capo) });
      }
    }
  }
  return out;
}

export function capoPrior(c) {
  const p = [1.0, 0.25, 0.4, 0.38, 0.32, 0.3, 0.2, 0.22, 0.12, 0.1, 0.06, 0.04, 0.03];
  return p[Math.min(c, p.length - 1)];
}

// Sounding pitches of the open strings with a capo (what actually rings)
export function soundingOpen(tuning, capo) {
  return tuning.map((m) => m + capo);
}
