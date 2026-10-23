// Which string a note was played on, from how it sounds.
//
// The same pitch played on a thicker string higher up the neck sounds different: a shorter,
// stiffer vibrating length makes the overtones run sharp (inharmonicity grows with the
// string's thickness and with 1/length^2), and the pick position along the string shapes
// which overtones are strong or weak. For every note this measures its first overtones in
// three short snapshots after the attack (level, how sharp each one runs, and a fitted
// inharmonicity coefficient), skipping overtones that collide with other notes sounding at
// the same time. A small network (trained on GuitarSet, tools/train_strings.py) turns that
// into a score for each place on the neck the note could have been played.

import { RealFFT, hannPeriodic } from './fft.js';
import { MODEL } from './strings-model.js';

export const SF = {
  sr: 44100,
  win: 4096, // 93 ms analysis window
  nfft: 8192, // zero-padded for finer peak positions
  frames: [0.01, 0.06, 0.14], // window starts after the onset (s)
  K: 24, // overtones per snapshot
  fmax: 9000, // ignore overtones above this
  collide: 35, // cents: another note's overtone this close spoils a measurement
};
// features per note: K x 3 per snapshot (level, deviation, found) + 3 per snapshot (B, fit ok,
// snapshot ok) + 4 global
export const featureCount = (cfg = SF) => cfg.frames.length * (cfg.K * 3 + 3) + 4;
export const NFEAT = featureCount(SF);

const midiHz = (m) => 440 * Math.pow(2, (m - 69) / 12);

/**
 * Overtone features for notes of a mono 44.1 kHz guitar signal.
 * @param x Float32Array audio (44.1 kHz)
 * @param notes [{t0, t1, pitch}] (all the notes of the part: they are also used to find colliding overtones)
 * @param opts {cents: tuning offset of the recording, jitter: seconds added to every onset (training)}
 * @returns Float32Array(notes.length * NFEAT)
 */
export function stringFeatures(x, notes, opts = {}) {
  const cfg = opts.cfg ? { ...SF, ...opts.cfg } : SF;
  const { win, nfft, frames, K, fmax } = cfg;
  const NF = featureCount(cfg);
  const sr = SF.sr;
  const cents = opts.cents || 0;
  const jit = opts.jitter || null;
  const W = hannPeriodic(win);
  const rfft = RealFFT.get(nfft);
  const bins = nfft / 2 + 1;
  const binHz = sr / nfft;
  const buf = new Float64Array(nfft), re = new Float64Array(bins), im = new Float64Array(bins), mag = new Float64Array(bins);
  const out = new Float32Array(notes.length * NF);
  // notes sorted by start, to find the ones sounding during a snapshot quickly
  const order = notes.map((_, i) => i).sort((a, b) => notes[a].t0 - notes[b].t0);
  let maxDur = 0;
  for (const n of notes) maxDur = Math.max(maxDur, n.t1 - n.t0);
  const lvl = new Float64Array(K), fk = new Float64Array(K), ok = new Uint8Array(K), bad = new Uint8Array(K);
  for (let ni = 0; ni < notes.length; ni++) {
    const n = notes[ni];
    const t0 = n.t0 + (jit ? jit[ni] : 0);
    const f0 = midiHz(n.pitch + cents / 100);
    const o = ni * NF;
    let poly = 0, ref = null;
    for (let fi = 0; fi < frames.length; fi++) {
      const fo = o + fi * (K * 3 + 3);
      const a = Math.round((t0 + frames[fi]) * sr);
      const ta = a / sr, tb = (a + win) / sr;
      if (a < 0 || a + win > x.length || ta > n.t1 + 0.05) continue; // snapshot unusable: all zeros
      // overtones of other notes sounding in this window
      bad.fill(0);
      let lo = 0, hi = order.length;
      while (lo < hi) { const m = (lo + hi) >> 1; if (notes[order[m]].t0 < ta - maxDur) lo = m + 1; else hi = m; }
      for (let q = lo; q < order.length; q++) {
        const m = notes[order[q]];
        if (m.t0 >= tb) break;
        if (order[q] === ni || m.t1 <= ta) continue;
        if (fi === 0) poly++;
        const g0 = midiHz(m.pitch + cents / 100);
        for (let h = 1; h * g0 < fmax * 1.05; h++) {
          const r = (h * g0) / f0; // in units of this note's f0
          const k = Math.round(r);
          if (k < 1 || k > K) continue;
          if (Math.abs(1200 * Math.log2(r / k)) < cfg.collide + 20 * (k > 8 ? 1 : 0)) bad[k - 1] = 1;
        }
      }
      // spectrum of the snapshot
      for (let i = 0; i < win; i++) buf[i] = x[a + i] * W[i];
      buf.fill(0, win);
      rfft.forward(buf, re, im);
      for (let b = 0; b < bins; b++) mag[b] = Math.sqrt(re[b] * re[b] + im[b] * im[b]) + 1e-9;
      // follow the overtones up, predicting each from the inharmonicity fitted so far
      let B = 0, f0m = f0, sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, nfit = 0, topMag = 0;
      ok.fill(0);
      for (let k = 1; k <= K; k++) {
        const exp = k * f0m * Math.sqrt(1 + Math.max(0, B) * k * k);
        if (exp > fmax) break;
        const wc = k <= 2 ? 45 : 30;
        let flo = exp * Math.pow(2, -wc / 1200), fhi = exp * Math.pow(2, wc / 1200);
        flo = Math.max(flo, exp - 0.4 * f0); fhi = Math.min(fhi, exp + 0.4 * f0);
        let b0 = Math.max(2, Math.floor(flo / binHz)), b1 = Math.min(bins - 3, Math.ceil(fhi / binHz));
        if (b1 - b0 < 2) { b0 = Math.max(2, Math.round(exp / binHz) - 1); b1 = b0 + 2; }
        let bi = b0;
        for (let b = b0 + 1; b <= b1; b++) if (mag[b] > mag[bi]) bi = b;
        const isPeak = mag[bi] >= mag[bi - 1] && mag[bi] >= mag[bi + 1] && bi > b0 && bi < b1;
        const la = Math.log(mag[bi - 1]), lb = Math.log(mag[bi]), lc = Math.log(mag[bi + 1]);
        const den = la - 2 * lb + lc;
        const d = den < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (la - lc)) / den)) : 0;
        const f = (bi + d) * binHz;
        lvl[k - 1] = (20 / Math.LN10) * (lb - 0.25 * (la - lc) * d);
        fk[k - 1] = f;
        ok[k - 1] = isPeak ? 1 : 0;
        if (isPeak && !bad[k - 1]) {
          // weighted least squares of (f_k / k)^2 = f0^2 (1 + B k^2)
          topMag = Math.max(topMag, mag[bi]);
          const w = Math.max(0.05, Math.min(1, mag[bi] / topMag)) * (k <= 2 ? 0.5 : 1);
          const xk = k * k, yk = (f / k) * (f / k);
          sw += w; sx += w * xk; sy += w * yk; sxx += w * xk * xk; sxy += w * xk * yk; nfit++;
          if (nfit >= 3) {
            const vx = sxx - (sx * sx) / sw;
            if (vx > 1e-9) {
              const slope = (sxy - (sx * sy) / sw) / vx, icpt = (sy - slope * sx) / sw;
              if (icpt > 0) { B = slope / icpt; f0m = Math.sqrt(icpt); }
            }
          } else if (k <= 3) f0m = f / k;
        }
      }
      // levels relative to the loudest overtone of the first usable snapshot (so later
      // snapshots show how fast each overtone dies away)
      let top = -1e9;
      for (let k = 0; k < K; k++) if (ok[k] && !bad[k] && lvl[k] > top) top = lvl[k];
      if (ref === null && top > -1e8) ref = top;
      const r0 = ref === null ? top : ref;
      for (let k = 0; k < K; k++) {
        const use = ok[k] && !bad[k] && (k + 1) * f0m < fmax;
        const dev = use ? 1200 * Math.log2(fk[k] / ((k + 1) * f0m)) : 0;
        out[fo + 3 * k] = use ? Math.max(-3, Math.min(1, (lvl[k] - r0) / 20)) + 1.5 : 0;
        out[fo + 3 * k + 1] = use ? Math.max(-2, Math.min(4, dev / 25)) : 0;
        out[fo + 3 * k + 2] = use ? 1 : 0;
      }
      const fitOk = nfit >= 4 && B > -2e-4 && B < 5e-3;
      out[fo + 3 * K] = fitOk ? (Math.log10(Math.max(1e-6, B)) + 4) / 2 : 0;
      out[fo + 3 * K + 1] = fitOk ? 1 : 0;
      out[fo + 3 * K + 2] = 1;
    }
    const g = o + frames.length * (K * 3 + 3);
    out[g] = (n.pitch - 60) / 24;
    out[g + 1] = Math.log(Math.max(0.02, n.t1 - n.t0)) / 2;
    out[g + 2] = Math.min(6, poly) / 6;
    out[g + 3] = cents / 50;
  }
  return out;
}

/** Places a note can be played: [{s, fret (from the nut, capo included)}] */
export function candidates(pitch, tuning, capo = 0, maxFret = 22) {
  const out = [];
  for (let s = 0; s < tuning.length; s++) {
    const f = pitch - tuning[s];
    if (f >= capo && f <= maxFret) out.push({ s, fret: f });
  }
  return out;
}

/** Input vector describing one candidate place for the scoring network */
export function candFeatures(s, fret) {
  const c = new Float32Array(8);
  c[s] = 1;
  c[6] = fret / 12;
  c[7] = fret === 0 ? 1 : 0;
  return c;
}

/**
 * The scoring network from its weights (tools/train_strings.py 'fit' output, or the packed
 * float16 form in engine/strings-model.js).
 */
export function stringModel(w) {
  const [H1, H2, H3, NC] = w.h;
  const f32 = (a) => (a instanceof Float32Array ? a : Float32Array.from(a.flat ? a.flat(2) : a));
  return { D: w.nfeat, H1, H2, H3, NC, W1: f32(w.weights.W1), b1: f32(w.weights.b1), W2: f32(w.weights.W2), b2: f32(w.weights.b2), W3h: f32(w.weights.W3h), W3c: f32(w.weights.W3c), b3: f32(w.weights.b3), w4: f32(w.weights.w4), b4: f32(w.weights.b4)[0] };
}

/**
 * The part of the network that depends only on the sound: per note, H3 numbers that
 * placeLogProbs() combines with each possible string and fret.
 * @returns Float32Array(nNotes * H3)
 */
export function noteEmbeddings(m, feats, n) {
  const { D, H1, H2, H3 } = m;
  const out = new Float32Array(n * H3);
  const h1 = new Float32Array(H1), h2 = new Float32Array(H2);
  for (let i = 0; i < n; i++) {
    const x = feats.subarray(i * D, (i + 1) * D);
    h1.set(m.b1);
    for (let d = 0; d < D; d++) {
      const v = x[d];
      if (v === 0) continue;
      const row = d * H1;
      for (let j = 0; j < H1; j++) h1[j] += v * m.W1[row + j];
    }
    h2.set(m.b2);
    for (let a = 0; a < H1; a++) {
      const v = h1[a];
      if (v <= 0) continue;
      const row = a * H2;
      for (let j = 0; j < H2; j++) h2[j] += v * m.W2[row + j];
    }
    const o = i * H3;
    for (let j = 0; j < H3; j++) out[o + j] = m.b3[j];
    for (let a = 0; a < H2; a++) {
      const v = h2[a];
      if (v <= 0) continue;
      const row = a * H3;
      for (let j = 0; j < H3; j++) out[o + j] += v * m.W3h[row + j];
    }
  }
  return out;
}

/**
 * Log-probability of each string for every note under a setup (tuning + capo), from the
 * notes' sound embeddings (n.sv, from noteEmbeddings). Strings the note can't be played on
 * get -Infinity; notes without an embedding get null.
 * @returns array (per note) of Float32Array(6) | null
 */
export function placeLogProbs(m, notes, setup, maxFret = 22) {
  const { H3, W3c, w4, b4 } = m;
  const out = new Array(notes.length);
  const z = new Float64Array(6);
  for (let i = 0; i < notes.length; i++) {
    const n = notes[i], e = n.sv;
    if (!e) { out[i] = null; continue; }
    let best = -Infinity;
    for (let s = 0; s < 6; s++) {
      const f = n.pitch - setup.tuning[s];
      const rel = f - (setup.capo || 0);
      if (rel < 0 || f > maxFret) { z[s] = -Infinity; continue; }
      const open = rel === 0 ? 1 : 0;
      let v = b4;
      for (let j = 0; j < H3; j++) {
        const a = e[j] + W3c[s * H3 + j] + (f / 12) * W3c[6 * H3 + j] + open * W3c[7 * H3 + j];
        if (a > 0) v += a * w4[j];
      }
      z[s] = v;
      if (v > best) best = v;
    }
    const lp = new Float32Array(6);
    if (best === -Infinity) { lp.fill(-Infinity); out[i] = lp; continue; }
    let sum = 0;
    for (let s = 0; s < 6; s++) if (z[s] > -Infinity) sum += Math.exp(z[s] - best);
    const lse = best + Math.log(sum);
    for (let s = 0; s < 6; s++) lp[s] = z[s] - lse;
    out[i] = lp;
  }
  return out;
}

let DEFAULT;
/** The trained network shipped with the app (engine/strings-model.js), or null without one. */
export function defaultStringModel() {
  if (DEFAULT === undefined) DEFAULT = MODEL ? stringModel(unpackModel(MODEL)) : null;
  return DEFAULT;
}

/** {nfeat, h, shapes, data: base64 of float16 weights} -> the stringModel() input form */
export function unpackModel(p) {
  const bin = typeof atob === 'function' ? Uint8Array.from(atob(p.data), (c) => c.charCodeAt(0)) : Uint8Array.from(Buffer.from(p.data, 'base64'));
  const u16 = new Uint16Array(bin.buffer, bin.byteOffset, bin.byteLength >> 1);
  const weights = {};
  let o = 0;
  for (const [name, n] of p.shapes) {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = halfToFloat(u16[o + i]);
    weights[name] = a;
    o += n;
  }
  return { nfeat: p.nfeat, h: p.h, weights };
}

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

/**
 * Measure every note's overtones in the part's audio and keep the sound embedding on the
 * note (n.sv), so any later re-fingering (another tuning or capo) can use it.
 */
export function attachStringEvidence(notes, audio, cents, model) {
  if (!model || !notes.length) return;
  const feats = stringFeatures(audio, notes, { cents });
  const emb = noteEmbeddings(model, feats, notes.length);
  const H = model.H3;
  notes.forEach((n, i) => {
    n.sv = emb.slice(i * H, (i + 1) * H);
    const b = noteStiffness(feats, i);
    if (b != null) n.sb = b; else delete n.sb;
  });
}

/** median fitted inharmonicity (log10 B) over a note's snapshots, or null */
function noteStiffness(feats, i) {
  const { K, frames } = SF;
  const v = [];
  for (let fi = 0; fi < frames.length; fi++) {
    const o = i * NFEAT + fi * (K * 3 + 3);
    if (feats[o + 3 * K + 1] > 0) v.push(feats[o + 3 * K] * 2 - 4);
  }
  if (!v.length) return null;
  v.sort((a, b) => a - b);
  return v[v.length >> 1];
}

// log10 B grows by log10(2)/6 per fret (B ~ 1/length^2, length halves every 12 frets)
const STIFF_PER_FRET = Math.log10(2) / 6;

/**
 * This guitar's strings, measured from the recording itself: given a fingering, fit each
 * string's stiffness (log10 B at the nut) from the notes placed on it, and score every place
 * each note could be played by how well its measured stiffness matches that string there.
 * Electric and acoustic string sets differ a lot here (a plain or a wound G string, lighter
 * or heavier gauges), which a fixed model can't know. Adds the score to the sound evidence.
 * (Tuned on EGSet12 and Guitar-TECHS player 3, both electric, and GuitarSet: tests/eval_strings.mjs.)
 * @param base per-note log-probabilities (placeLogProbs) for this setup
 * @param assign Map note index -> {string, fret} (fret from the capo)
 * @returns new per-note evidence arrays (base where nothing is known)
 */
export function stiffnessEvidence(notes, setup, base, assign, weight = 8) {
  const zs = Array.from({ length: 6 }, () => []);
  const capo = setup.capo || 0;
  notes.forEach((n, i) => {
    const a = assign.get(i);
    if (a && n.sb != null) zs[a.string].push(n.sb - STIFF_PER_FRET * (a.fret + capo));
  });
  const all = zs.flat();
  if (all.length < 20) return base;
  const med = (a) => { const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1]; };
  const spread = (a, m) => Math.max(0.15, 1.4826 * med(a.map((x) => Math.abs(x - m))));
  const gm = med(all), gsd = spread(all, gm);
  const mu = zs.map((a) => (a.length >= 8 ? med(a) : null));
  // only when the G string sounds stiffer than the D, as a plain G (electric sets) does: the
  // model already knows wound G strings, and refitting there only adds noise
  if (mu[3] == null || mu[2] == null || mu[3] - mu[2] < 0.05) return base;
  const sd = zs.map((a, s) => (a.length >= 8 ? spread(a, mu[s]) : gsd));
  return notes.map((n, i) => {
    const b = base[i];
    if (!b || n.sb == null) return b;
    const out = new Float32Array(6);
    for (let s = 0; s < 6; s++) {
      if (!(b[s] > -Infinity) || mu[s] == null) { out[s] = b[s]; continue; }
      const z = (n.sb - STIFF_PER_FRET * (n.pitch - setup.tuning[s]) - mu[s]) / sd[s];
      out[s] = b[s] + weight * (-0.5 * Math.min(9, z * z) - Math.log(sd[s]));
    }
    return out;
  });
}
