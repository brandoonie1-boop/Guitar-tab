// Precise pitch contours of single notes, for bends, vibrato and slides. The basic-pitch contour
// moves in thirds of a semitone; a guitar's vibrato is often a quarter of a semitone. Here each
// frame measures the instantaneous frequencies (phase vocoder) of the note's first partials, near
// where they should be, and combines them into one pitch, in cents from the note's nominal pitch.
// Partials that another sounding note also has are left out.

import { RealFFT } from './fft.js';

const N = 2048, HOP = 256;
let WIN = null;

function window() {
  if (!WIN) {
    WIN = new Float64Array(N);
    for (let i = 0; i < N; i++) WIN[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
  }
  return WIN;
}

const hz = (m) => 440 * Math.pow(2, (m - 69) / 12);

/**
 * Contour of one note: { dt (s between frames), t0 (time of the first frame), cents (relative to
 * the note's pitch, NaN where it can't be measured), level (dB) }.
 * x: mono audio, sr: its rate; others: notes sounding at the same time (their partials are skipped).
 * Each partial is looked for from 250 cents below where it should be to 350 above (bends, a
 * released bend), or within opts.near cents of it (default 90) where another note's partial is in
 * the way, never as far as the note's own next partial. opts.down / opts.up: fixed bounds instead
 * (a slide's both ways).
 */
export function noteContour(x, sr, note, others = [], opts = {}) {
  const near = opts.near ?? 90, maxH = opts.partials ?? 6;
  const f0 = hz(note.pitch);
  const fft = RealFFT.get(N), w = window();
  const re = new Float64Array(N / 2 + 1), im = new Float64Array(N / 2 + 1);
  const pre = new Float64Array(N / 2 + 1), pim = new Float64Array(N / 2 + 1);
  const buf = new Float64Array(N);
  const start = Math.round((note.t0 + 0.015) * sr), end = Math.round(Math.min(note.t1, note.t0 + 3) * sr);
  const nFrames = Math.max(0, Math.floor((end - start) / HOP));
  const cents = new Float32Array(nFrames).fill(NaN), level = new Float32Array(nFrames);
  const bright = new Float32Array(nFrames), between = new Float32Array(nFrames);
  if (nFrames < 2) return { dt: HOP / sr, t0: start / sr, cents, level, bright, between };
  // the partials whose search band no other note's partial falls into
  const clashes = (lo, hi) => {
    for (const o of others) {
      if (o === note) continue;
      const g = hz(o.pitch);
      for (let k = 1; k * g < hi * 1.01; k++) if (k * g > lo * 0.99) return true;
    }
    return false;
  };
  const bands = [];
  for (let h = 1; h <= maxH; h++) {
    const cents = (c) => h * f0 * Math.pow(2, c / 1200);
    let lo, hi;
    if (opts.down != null) {
      lo = cents(-opts.down); hi = cents(opts.up);
    } else {
      const upMax = 0.9 * 1200 * Math.log2((h + 1) / h), downMax = h > 1 ? 0.9 * 1200 * Math.log2(h / (h - 1)) : 1200;
      lo = cents(-Math.min(250, downMax)); hi = cents(Math.min(350, upMax));
      if (clashes(lo, hi)) { // in the way: just near the partial
        lo = cents(-near); hi = cents(near);
        if (clashes(lo, hi)) continue;
      }
    }
    if (hi >= sr / 2 - 200) break;
    bands.push({ h, k0: Math.max(1, Math.floor((lo * N) / sr)), k1: Math.min(N / 2 - 1, Math.ceil((hi * N) / sr)) });
  }
  if (!bands.length) return { dt: HOP / sr, t0: start / sr, cents, level, bright, between };
  const kOf = (f) => Math.max(1, Math.min(N / 2 - 1, Math.round((f * N) / sr)));
  const kLow = kOf(f0 * 0.8), kMid = kOf(f0 * 3.5), kTop = kOf(Math.min(sr / 2 - 500, 10000));
  const frame = (pos, outRe, outIm) => {
    for (let i = 0; i < N; i++) {
      const j = pos - N / 2 + i;
      buf[i] = j >= 0 && j < x.length ? x[j] * w[i] : 0;
    }
    fft.forward(buf, outRe, outIm);
  };
  frame(start - HOP, pre, pim);
  const est = [], wts = [];
  for (let m = 0; m < nFrames; m++) {
    const pos = start + m * HOP;
    frame(pos, re, im);
    est.length = 0; wts.length = 0;
    let e = 0;
    for (const b of bands) {
      let best = -1, kb = -1;
      for (let k = b.k0; k <= b.k1; k++) {
        const p = re[k] * re[k] + im[k] * im[k];
        if (p > best) { best = p; kb = k; }
      }
      e += best;
      if (kb <= b.k0 || kb >= b.k1 || best <= 0) continue; // the band's edge: no peak inside
      // instantaneous frequency from the phase advance over one hop
      const ph = Math.atan2(im[kb], re[kb]), pph = Math.atan2(pim[kb], pre[kb]);
      let d = ph - pph - (2 * Math.PI * kb * HOP) / N;
      d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      const f = ((kb + (d * N) / (2 * Math.PI * HOP)) * sr) / N;
      est.push(1200 * Math.log2(f / (b.h * f0)));
      wts.push(Math.sqrt(best));
    }
    level[m] = 10 * Math.log10(e / N + 1e-12);
    // brightness: energy above the 3rd partial against energy up to it; "between": energy at the
    // half-harmonics (1.5 f0, 2.5 f0) against the 2nd and 3rd partials (a natural harmonic has
    // a pure spectrum; a fretted note's own partials are all there)
    let lo = 0, hiE = 0;
    for (let k = kLow; k < kMid; k++) lo += re[k] * re[k] + im[k] * im[k];
    for (let k = kMid; k < kTop; k++) hiE += re[k] * re[k] + im[k] * im[k];
    bright[m] = 10 * Math.log10((hiE + 1e-12) / (lo + 1e-12));
    let half = 0, whole = 0;
    for (const [mult, add] of [[1.5, 0], [2.5, 0], [2, 1], [3, 1]]) {
      const k = kOf(mult * f0);
      const v = re[k] * re[k] + im[k] * im[k] + re[k - 1] * re[k - 1] + im[k - 1] * im[k - 1] + re[k + 1] * re[k + 1] + im[k + 1] * im[k + 1];
      if (add) whole += v; else half += v;
    }
    between[m] = 10 * Math.log10((half + 1e-12) / (whole + 1e-12));
    if (est.length) {
      // weighted median of the partials' estimates
      const idx = est.map((_, i) => i).sort((a, b) => est[a] - est[b]);
      const tot = wts.reduce((s, v) => s + v, 0);
      let acc = 0;
      for (const i of idx) { acc += wts[i]; if (acc >= tot / 2) { cents[m] = est[i]; break; } }
    }
    pre.set(re); pim.set(im);
  }
  return { dt: HOP / sr, t0: start / sr, cents, level, bright, between };
}
