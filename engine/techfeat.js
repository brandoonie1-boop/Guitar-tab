// Features of one note for telling playing techniques apart (bend, vibrato, palm mute, natural
// harmonic, plain): the shape of its precise pitch contour (engine/pitchtrack.js), how its level
// decays, how bright it is, how pure its spectrum is, and its context. And features of how a note
// began (onsetFeatures): picked, or hammered on / pulled off, slid into, or the rest of a bend.

import { noteContour } from './pitchtrack.js';

export const TECH_FEATURES = ['rise', 'fall', 'range', 'endDelta', 'riseTime', 'held', 'wobble', 'wobbleRate', 'wobbleDepth', 'measured',
  'decay', 'drop150', 'bright', 'brightLate', 'between', 'dur', 'pitch', 'openPitch', 'poly', 'harmPos'];

const pct = (a, q) => {
  const s = a.filter((v) => !Number.isNaN(v)).sort((x, y) => x - y);
  return s.length ? s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))] : 0;
};
const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);

// natural harmonic positions above an open string: 12th fret (octave), 7th (octave + fifth),
// 5th (two octaves), 4th (two octaves + major third)
const HARM = [12, 19, 24, 28];
const STD = [40, 45, 50, 55, 59, 64];

/**
 * Feature vector (TECH_FEATURES order) of note n, from the guitar audio x (mono, rate sr); others:
 * the notes sounding with it. Computed before strings are assigned: openPitch says whether some
 * open string has the note's pitch and harmPos whether some string has a natural harmonic there
 * (standard tuning, or the tuning given). cents: how far the guitar is tuned from A440.
 */
export function techFeatures(x, sr, n, others, tuning = STD, cents0 = 0) {
  const c = noteContour(x, sr, n, others);
  const cents = Array.from(c.cents, (v) => v - cents0), lv = Array.from(c.level);
  const nf = cents.length, dt = c.dt;
  const ok = cents.filter((v) => !Number.isNaN(v));
  const head = cents.slice(0, Math.max(2, Math.round(0.06 / dt))).filter((v) => !Number.isNaN(v));
  const base = head.length ? pct(head, 0.5) : pct(cents, 0.5);
  const hi = pct(cents, 0.9), lo = pct(cents, 0.1);
  const last = cents.slice(Math.max(0, nf - Math.round(0.06 / dt))).filter((v) => !Number.isNaN(v));
  // how long until the contour first gets most of the way up
  let riseTime = 0;
  if (hi - base > 40) for (let i = 0; i < nf; i++) if (cents[i] >= base + 0.7 * (hi - base)) { riseTime = i * dt; break; }
  let held = 0;
  for (const v of cents) if (!Number.isNaN(v) && v >= base + 0.7 * (hi - base) && hi - base > 40) held += dt;
  // wobble: the contour minus its 250 ms running mean
  const k = Math.max(1, Math.round(0.125 / dt));
  const det = [];
  for (let i = 0; i < nf; i++) {
    if (Number.isNaN(cents[i])) continue;
    let s = 0, m = 0;
    for (let j = Math.max(0, i - k); j <= Math.min(nf - 1, i + k); j++) if (!Number.isNaN(cents[j])) { s += cents[j]; m++; }
    det.push(cents[i] - s / m);
  }
  let zc = 0;
  for (let i = 1; i < det.length; i++) if ((det[i] > 0) !== (det[i - 1] > 0)) zc++;
  const wob = Math.sqrt(mean(det.map((v) => v * v)));
  // level: decay over the first 300 ms after the loudest frame, and the drop at 150 ms
  let pk = 0;
  for (let i = 0; i < Math.min(nf, Math.round(0.1 / dt)); i++) if (lv[i] > lv[pk]) pk = i;
  const i150 = Math.min(nf - 1, pk + Math.round(0.15 / dt)), i300 = Math.min(nf - 1, pk + Math.round(0.3 / dt));
  const decay = i300 > pk ? (lv[pk] - lv[i300]) / ((i300 - pk) * dt) : 0;
  const early = Array.from(c.bright).slice(0, Math.max(1, Math.round(0.1 / dt)));
  const late = Array.from(c.bright).slice(Math.round(0.1 / dt), Math.round(0.3 / dt));
  const harmPos = tuning.some((o) => HARM.includes(n.pitch - o)) ? 1 : 0;
  return [
    hi - base, base - lo, hi - lo, (last.length ? mean(last) : base) - base, riseTime, held,
    wob, det.length > 1 ? zc / (2 * det.length * dt) : 0, 2 * Math.SQRT2 * wob, ok.length / Math.max(1, nf),
    decay, i150 > pk ? lv[pk] - lv[i150] : 0, mean(early), late.length ? mean(late) : mean(early),
    pct(Array.from(c.between), 0.5), n.t1 - n.t0, n.pitch, tuning.includes(n.pitch) ? 1 : 0, others.length, harmPos,
  ];
}

export const TRANS_FEATURES = ['gap', 'interval', 'absInterval', 'aEndToward', 'bStartFrom', 'levelJump', 'hfJump', 'aDur', 'bDur', 'aMeasured', 'bMeasured'];

/** Features of the move from note a to note b (the next note on the same string): a slide glides, a hammer-on or pull-off steps without a new pick attack, a picked note steps with one. */
export function transFeatures(x, sr, a, b, others) {
  const iv = b.pitch - a.pitch, sgn = Math.sign(iv) || 1;
  const ca = noteContour(x, sr, a, others.filter((o) => o !== b), { down: 450, up: 450 });
  const cb = noteContour(x, sr, b, others.filter((o) => o !== a), { down: 450, up: 450 });
  const valid = (arr) => Array.from(arr).filter((v) => !Number.isNaN(v));
  const nEnd = Math.max(2, Math.round(0.04 / ca.dt));
  const aAll = valid(ca.cents), aEnd = valid(ca.cents.slice(-nEnd));
  const bAll = valid(cb.cents), bStart = valid(cb.cents.slice(0, nEnd));
  const med = (v) => { const s = v.slice().sort((p, q) => p - q); return s.length ? s[s.length >> 1] : 0; };
  const aEndToward = aEnd.length && aAll.length ? sgn * (med(aEnd) - med(aAll)) : 0;
  const bStartFrom = bStart.length && bAll.length ? -sgn * (med(bStart) - med(bAll)) : 0;
  const la = Array.from(ca.level), lb = Array.from(cb.level);
  const aLate = la.slice(-nEnd), bEarly = lb.slice(0, nEnd);
  const levelJump = (bEarly.length ? Math.max(...bEarly) : 0) - (aLate.length ? mean(aLate) : 0);
  // broadband energy above 2 kHz just after b starts against just before (a pick's click)
  const e = (from, to) => {
    let s = 0, m = 0;
    for (let i = Math.max(1, from); i < Math.min(x.length, to); i++) { const d = x[i] - x[i - 1]; s += d * d; m++; }
    return 10 * Math.log10(s / Math.max(1, m) + 1e-12);
  };
  const ib = Math.round(b.t0 * sr), w = Math.round(0.02 * sr);
  const hfJump = e(ib, ib + w) - e(ib - 2 * w, ib - w);
  return [b.t0 - a.t1, iv, Math.abs(iv), aEndToward, bStartFrom, levelJump, hfJump, a.t1 - a.t0, b.t1 - b.t0,
    aAll.length / Math.max(1, ca.cents.length), bAll.length / Math.max(1, cb.cents.length)];
}

export const ONSET_FEATURES = ['interval', 'absInterval', 'gap', 'aDur', 'bDur', 'sinceA', 'glide', 'between', 'measured',
  'levelJump', 'hfJump', 'hfBurst', 'sharp', 'attackTime', 'conf', 'velocity', 'velocityA', 'others'];

/**
 * How note b began, given note a, the note before it that ends about where b starts: a pick
 * attack (a click of high-frequency noise and a jump in level), no new attack (a hammer-on or
 * pull-off: the string was ringing already), a slide (the pitch passes through the frets between)
 * or the rest of a bend or vibrato that note detection split in two (the pitch glides). Feature
 * vector in ONSET_FEATURES order; x: mono audio at rate sr, others: the other notes sounding then,
 * cents: how far the guitar is tuned from A440.
 */
export function onsetFeatures(x, sr, a, b, others = [], cents = 0) {
  const iv = b.pitch - a.pitch, D = 100 * iv;
  // the pitch across the move: one contour from a's pitch that reaches b's (only the partials
  // whose search bands don't run into the next partial up)
  const lo = Math.min(0, D) - 80, hi = Math.max(0, D) + 80;
  const partials = Math.max(1, Math.min(4, Math.floor(1 / (Math.pow(2, (hi - lo) / 1200) - 1))));
  const tS = Math.max(a.t0 + 0.02, b.t0 - 0.12), tE = Math.min(Math.max(b.t1, b.t0 + 0.03), b.t0 + 0.12);
  const c = noteContour(x, sr, { pitch: a.pitch, t0: tS - 0.015, t1: tE }, [], { down: -lo, up: hi, partials });
  const pitch = Array.from(c.cents, (v) => v - cents);
  const near = (v, to) => !Number.isNaN(v) && Math.abs(v - to) < 50;
  // arrival: the first of two frames in a row at b's pitch; departure: the last frame at a's before it
  let arrive = -1, leave = -1, between = 0;
  for (let i = 0; i + 1 < pitch.length; i++) if (near(pitch[i], D) && near(pitch[i + 1], D)) { arrive = i; break; }
  if (arrive >= 0) {
    for (let i = arrive - 1; i >= 0; i--) if (near(pitch[i], 0)) { leave = i; break; }
    const [p, q] = D > 0 ? [50, D - 50] : [D + 50, -50];
    for (let i = Math.max(0, leave); i < arrive; i++) if (pitch[i] > p && pitch[i] < q) between++;
  }
  const glide = arrive >= 0 && leave >= 0 ? (arrive - leave) * c.dt : arrive >= 0 ? arrive * c.dt : 0.25;
  const measured = pitch.filter((v) => !Number.isNaN(v)).length / Math.max(1, pitch.length);
  // levels around b's start, 1.45 ms apart: broadband, and of the second difference (the top
  // end, where a pick's click is)
  const hop = Math.max(1, Math.round(0.00145 * sr)), win = 2 * hop, origin = b.t0 - 0.08;
  const nH = Math.round((0.2 * sr) / hop);
  const E = new Float32Array(nH), H = new Float32Array(nH);
  for (let k = 0; k < nH; k++) {
    let e = 0, h = 0;
    const s0 = Math.round(origin * sr) + k * hop;
    for (let i = Math.max(2, s0); i < Math.min(x.length, s0 + win); i++) {
      const d = x[i] - 2 * x[i - 1] + x[i - 2];
      e += x[i] * x[i]; h += d * d;
    }
    E[k] = 10 * Math.log10(e / win + 1e-12); H[k] = 10 * Math.log10(h / win + 1e-12);
  }
  const at = (t) => Math.max(0, Math.min(nH - 1, Math.round(((t - origin) * sr) / hop)));
  // the attack: the steepest rise of the top end within 30 ms of where b was detected
  let kOn = at(b.t0), sharp = -Infinity;
  for (let k = at(b.t0 - 0.03); k <= at(b.t0 + 0.03); k++) {
    const r = H[Math.min(nH - 1, k + 2)] - H[Math.max(0, k - 2)];
    if (r > sharp) { sharp = r; kOn = k; }
  }
  const tOn = origin + (kOn * hop) / sr;
  const meanOf = (A, t1, t2) => { let s = 0, m = 0; for (let k = at(t1); k <= at(t2); k++) { s += A[k]; m++; } return m ? s / m : 0; };
  const maxOf = (A, t1, t2) => { let s = -Infinity; for (let k = at(t1); k <= at(t2); k++) s = Math.max(s, A[k]); return s; };
  const preE = meanOf(E, tOn - 0.03, tOn - 0.005), preH = meanOf(H, tOn - 0.03, tOn - 0.005);
  const topH = maxOf(H, tOn, tOn + 0.015);
  let kPk = at(tOn);
  for (let k = at(tOn); k <= at(tOn + 0.06); k++) if (E[k] > E[kPk]) kPk = k;
  return [
    iv, Math.abs(iv), b.t0 - a.t1, a.t1 - a.t0, b.t1 - b.t0, b.t0 - a.t0, glide, between * c.dt, measured,
    maxOf(E, tOn, tOn + 0.04) - preE, topH - preH, topH - meanOf(H, tOn + 0.04, tOn + 0.07), sharp, ((kPk - at(tOn)) * hop) / sr,
    b.conf ?? 0.5, b.velocity ?? 0.5, a.velocity ?? 0.5, others.length,
  ];
}

/**
 * The note each note may have come from without a new pick attack: the note that ends nearest to
 * where it starts (from 120 ms before to 50 ms after), started before it, and is a different pitch
 * no more than an octave away. Returns a Map note -> that note.
 */
export function onsetSources(notes) {
  const out = new Map();
  const byEnd = notes.slice().sort((p, q) => p.t1 - q.t1);
  for (const b of notes) {
    let best = null, bestD = Infinity;
    for (const a of byEnd) {
      if (a.t1 < b.t0 - 0.12) continue;
      if (a.t1 > b.t0 + 0.05) break;
      if (a === b || a.t0 > b.t0 - 0.03 || a.pitch === b.pitch || Math.abs(a.pitch - b.pitch) > 12) continue;
      const d = Math.abs(a.t1 - b.t0) + 0.01 * Math.abs(a.pitch - b.pitch);
      if (d < bestD) { bestD = d; best = a; }
    }
    if (best) out.set(b, best);
  }
  return out;
}
