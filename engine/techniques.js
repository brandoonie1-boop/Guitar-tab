// Guitar techniques: bends, vibrato, hammer-ons / pull-offs, slides, palm mutes, harmonics.
// How each note began - picked, or hammered on, pulled off or slid into from the note before it -
// is heard when the notes are found (attachOnsetEvidence: a small network on the attack and the
// pitch across the move, engine/techfeat.js onsetFeatures, trained on real recordings of each
// technique). The fingering then keeps a hammer-on, pull-off or slide on the string of the note it
// came from (engine/fretboard.js), and the marks follow from both. How each note itself sounds -
// bent, shaken, palm-muted, a harmonic - comes from its precise pitch contour, decay and spectrum
// (attachTechEvidence, engine/techfeat.js techFeatures, another small network).
// Conservative on purpose: a missing "h" is less confusing than a wrong one.

import { onsetFeatures, onsetSources, techFeatures } from './techfeat.js';
import { ONSET_MODEL } from './onset-model.js';
import { TECH_MODEL } from './tech-model.js';

// how sure the network must be before the tab shows h / p or a slide, and before a note heard as
// the rest of a bend is joined back into the note it came from (tuned on validation recordings:
// tests/eval_egipt.mjs)
export const MARK_AT = { legato: 0.6, slide: 0.6, join: 0.7, bend: 0.6, vibrato: 0.6, harmonic: 0.7, palmMute: 0.7 };

/** A network from tools/train_onset.py or tools/train_tech.py 'fit' output. */
export function onsetNet(w) {
  return { classes: w.classes, mean: w.mean, scale: w.scale, layers: w.layers.map((l) => ({ W: l.W, b: l.b })) };
}
export const techNet = onsetNet;

let DEFAULT, DEFAULT_TECH;
/** The networks shipped with the app (engine/onset-model.js, engine/tech-model.js), or null. */
export function defaultOnsetNet() {
  if (DEFAULT === undefined) DEFAULT = ONSET_MODEL ? onsetNet(ONSET_MODEL) : null;
  return DEFAULT;
}
export function defaultTechNet() {
  if (DEFAULT_TECH === undefined) DEFAULT_TECH = TECH_MODEL ? techNet(TECH_MODEL) : null;
  return DEFAULT_TECH;
}

/** Class probabilities ({pick, legato, slide, bend}, or {plain, bend, vibrato, ...}) from one feature vector. */
export function onsetProbs(net, f) {
  let h = f.map((v, i) => (v - net.mean[i]) / net.scale[i]);
  net.layers.forEach((l, k) => {
    const out = l.b.slice();
    for (let i = 0; i < h.length; i++) {
      const v = h[i];
      if (!v) continue;
      const row = l.W[i];
      for (let j = 0; j < out.length; j++) out[j] += v * row[j];
    }
    h = k < net.layers.length - 1 ? out.map((v) => (v > 0 ? v : 0)) : out;
  });
  const mx = Math.max(...h);
  const e = h.map((v) => Math.exp(v - mx));
  const sum = e.reduce((s, v) => s + v, 0);
  return Object.fromEntries(net.classes.map((c, i) => [c, e[i] / sum]));
}

/**
 * How each note began, from the guitar audio (mono, rate sr): every note that may have come from
 * the note before it without a new pick attack (engine/techfeat.js onsetSources) gets n.from (that
 * note's id) and n.how = {legato, slide, bend} probabilities (the rest is a pick). cents: how far
 * the guitar is tuned from A440.
 * A note heard as the rest of a bend or vibrato that note detection split off is joined back into
 * the note it came from, which then lasts until it ended and keeps the pitches it went through
 * (n.path, from its own); a bend that starts bent (released) moves the note to the fret it was
 * played at. Returns the notes left.
 */
export function attachOnsetEvidence(notes, audio, sr, cents = 0, net = defaultOnsetNet()) {
  for (const n of notes) { delete n.from; delete n.how; delete n.path; }
  if (!net || !notes.length) return notes;
  const r2 = (v) => Math.round(v * 100) / 100;
  const src = onsetSources(notes);
  for (const [b, a] of src) {
    const others = notes.filter((o) => o !== a && o !== b && o.t0 < b.t0 + 0.03 && o.t1 > b.t0);
    const p = onsetProbs(net, onsetFeatures(audio, sr, a, b, others, cents));
    b.from = a.id;
    b.how = { legato: r2(p.legato), slide: r2(p.slide), bend: r2(p.bend) };
  }
  // in time order, so a chain (bend, release, bend again) all ends up in its first note
  const into = new Map();
  for (const b of notes.slice().sort((p, q) => p.t0 - q.t0)) {
    if (!b.how || b.how.bend < MARK_AT.join || b.how.bend < Math.max(b.how.legato, b.how.slide)) continue;
    let a = src.get(b);
    while (into.has(a)) a = into.get(a);
    if (!a.path) a.path = [a.pitch];
    a.path.push(b.pitch);
    a.t1 = Math.max(a.t1, b.t1);
    into.set(b, a);
  }
  if (!into.size) return notes;
  for (const n of notes) {
    // started bent and only came down: played at the lowest pitch it reached
    if (n.path && n.path.every((p) => p <= n.path[0])) n.pitch = Math.min(...n.path);
  }
  return notes.filter((n) => !into.has(n));
}

/**
 * How each note sounds, from the guitar audio: n.tp = {bend, vibrato, harmonic, palmMute}
 * probabilities and rise (cents its pitch rose, for how far a bend went). Notes too short to tell
 * get none.
 */
export function attachTechEvidence(notes, audio, sr, cents = 0, net = defaultTechNet()) {
  for (const n of notes) delete n.tp;
  if (!net) return;
  const r2 = (v) => Math.round(v * 100) / 100;
  for (const n of notes) {
    if (n.t1 - n.t0 < 0.08) continue;
    const others = notes.filter((o) => o !== n && o.t0 < n.t1 && o.t1 > n.t0);
    const f = techFeatures(audio, sr, n, others, undefined, cents);
    const p = onsetProbs(net, f);
    n.tp = { bend: r2(p.bend), vibrato: r2(p.vibrato), harmonic: r2(p.harmonic), palmMute: r2(p.palmMute), rise: Math.round(f[0]) };
  }
}

/** What a joined note's pitches (n.path) say: {vibrato} back and forth, {bend: semitones up}, or {release: semitones down}. */
function pathMarks(path, pitch) {
  let turns = 0;
  for (let i = 2; i < path.length; i++) if (Math.sign(path[i] - path[i - 1]) === -Math.sign(path[i - 1] - path[i - 2]) && path[i] !== path[i - 1]) turns++;
  if (turns >= 2) return { vibrato: true };
  const up = Math.max(...path) - pitch;
  if (up > 0 && path[0] === pitch) return { bend: Math.min(3, up) };
  if (up > 0) return { release: Math.min(3, up) };
  return {};
}

export function detectTechniques(notes, opts = {}) {
  const tech = new Map();
  const get = (id) => { if (!tech.has(id)) tech.set(id, {}); return tech.get(id); };
  // the pitch track follows the strongest pitch within several semitones of a note, so a
  // neighbouring note that rings at the same time can look like a bend or vibrato
  const byStart = notes.slice().sort((a, b) => a.t0 - b.t0);
  const neighbour = (n, lo, hi) => byStart.some((m) => m !== n && m.t0 < n.t1 - 0.05 && m.t1 > n.t0 + 0.05 && m.pitch !== n.pitch && m.pitch >= n.pitch + lo && m.pitch <= n.pitch + hi);
  for (const n of notes) {
    if (n.path && n.fret !== 0) { // joined from notes the detection split
      const m = pathMarks(n.path, n.pitch);
      if (Object.keys(m).length) { Object.assign(get(n.id), m); continue; }
    }
    if (n.tp) { // heard from the note's own sound (attachTechEvidence)
      const t = {};
      if (n.tp.palmMute >= MARK_AT.palmMute) t.palmMute = true;
      if (n.tp.harmonic >= MARK_AT.harmonic) t.harmonic = HARM.includes(n.fret) ? 'natural' : 'pinch';
      if (n.fret !== 0 && !t.harmonic) { // an open string can't be bent or shaken
        if (n.tp.bend >= MARK_AT.bend && n.tp.rise >= 50) t.bend = Math.min(3, Math.max(0.5, Math.round(n.tp.rise / 50) / 2));
        else if (n.tp.vibrato >= MARK_AT.vibrato) t.vibrato = true;
      }
      if (Object.keys(t).length) Object.assign(get(n.id), t);
      continue;
    }
    // bends / vibrato from the pitch-bend track (1/3 semitone units per frame)
    const b = n.bends;
    if (!b || b.length < 6) continue;
    if (n.fret === 0) continue; // an open string can't be bent or shaken
    const start = median(b.slice(0, Math.min(3, b.length)));
    let mx = -99, mxi = 0;
    for (let i = 0; i < b.length; i++) if (b[i] > mx) { mx = b[i]; mxi = i; }
    const rise = mx - start;
    // a bend: stable start, then a rise of at least a semitone that is held for a while
    const head = b.slice(0, Math.min(4, b.length));
    const stableStart = Math.max(...head) - Math.min(...head) <= 1;
    let held = 0;
    for (let i = mxi; i < b.length && b[i] >= start + rise - 1; i++) held++;
    if (stableStart && rise >= 3 && held >= 4 && mxi >= 3 && n.t1 - n.t0 >= 0.18) {
      const semis = Math.round((rise / 3) * 2) / 2; // half-step resolution
      if (semis >= 0.5 && !neighbour(n, 1, Math.ceil(semis) + 1)) get(n.id).bend = Math.min(3, semis);
      continue;
    }
    // vibrato: several swings of at least 1/3 semitone around the centre
    const c = median(b);
    let swings = 0, sign = 0;
    for (const v of b) {
      const d = v - c;
      if (Math.abs(d) >= 1) {
        const s = Math.sign(d);
        if (s !== sign) { swings++; sign = s; }
      }
    }
    if (swings >= 4 && n.t1 - n.t0 > 0.35 && !neighbour(n, -2, 2)) get(n.id).vibrato = true;
  }
  if (notes.some((n) => n.how)) {
    // hammer-ons, pull-offs and slides as heard (attachOnsetEvidence), where the fingering put
    // both notes on one string
    const byId = new Map(notes.map((n) => [n.id, n]));
    for (const b of notes) {
      const a = b.how && byId.get(b.from);
      if (!a || a.string == null || a.string !== b.string || a.fret === b.fret) continue;
      const t = get(a.id);
      if (b.how.slide >= MARK_AT.slide && b.how.slide > b.how.legato) { t.slide = true; t.slideUp = b.pitch > a.pitch; }
      else if (b.how.legato >= MARK_AT.legato) { t.legato = true; t.legatoUp = b.pitch > a.pitch; }
    }
    return tech;
  }
  // without that (a project saved before it existed): consecutive notes on one string, the
  // second one barely re-attacked
  const byString = new Map();
  for (const n of notes) {
    if (n.string == null) continue;
    if (!byString.has(n.string)) byString.set(n.string, []);
    byString.get(n.string).push(n);
  }
  const weak = opts.weakOnset ?? 0.32;
  for (const list of byString.values()) {
    list.sort((a, b) => a.t0 - b.t0);
    for (let i = 0; i + 1 < list.length; i++) {
      const a = list[i], b = list[i + 1];
      const gap = b.t0 - a.t1;
      if (gap > 0.04 || gap < -0.12) continue;
      if (b.t0 - a.t0 > 0.6) continue;
      if ((b.onsetProb ?? 1) > weak) continue;
      const df = b.fret - a.fret;
      if (df === 0 || b.fret === 0 && a.fret === 0) continue;
      // a glide at the end of the first note toward the second one means slide
      const glide = a.bends && a.bends.length > 3 ? a.bends[a.bends.length - 1] - a.bends[Math.max(0, a.bends.length - 4)] : 0;
      const t = get(a.id);
      if (Math.abs(df) >= 2 && Math.sign(glide) === Math.sign(df) && Math.abs(glide) >= 2) {
        t.slide = true; t.slideUp = df > 0;
      } else if (Math.abs(df) <= 5) {
        t.legato = true; t.legatoUp = df > 0;
      }
    }
  }
  return tech;
}

// frets where natural harmonics are played
const HARM = [3, 4, 5, 7, 9, 12, 16, 19, 24];

function median(a) {
  const s = a.slice().sort((x, y) => x - y);
  return s.length ? s[s.length >> 1] : 0;
}
