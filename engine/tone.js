// What kind of guitar sound a recording has, so the tab player starts with a matching sound:
// 'acoustic' | 'clean' | 'crunch' | 'lead' (distortion). Read from the isolated guitar.
import { RealFFT, hannPeriodic } from './fft.js';

/**
 * Features of a mono guitar signal (44.1 kHz), over its louder stretches:
 *  hf     energy above 4 kHz relative to 100 Hz - 4 kHz (dB): acoustic guitars keep their
 *         top; an amp's speaker cuts it
 *  flat   spectral flatness 500 Hz - 4 kHz (dB): distortion fills the gaps between harmonics
 *  crest  peak over RMS in 100 ms windows (dB): distortion squashes the peaks
 *  decay  how fast notes fade after they start (dB per second): distortion sustains
 */
export function toneFeatures(x, sr = 44100) {
  const N = 2048, HOP = 1024;
  const win = hannPeriodic(N);
  const fft = RealFFT.get(N);
  const re = new Float64Array(N / 2 + 1), im = new Float64Array(N / 2 + 1), fr = new Float64Array(N);
  const nF = Math.max(0, Math.floor((x.length - N) / HOP));
  const binHz = sr / N;
  const k = (hz) => Math.min(N / 2, Math.round(hz / binHz));
  const k100 = k(100), k500 = k(500), k4k = k(4000), k12k = k(12000);
  const frames = [];
  let maxE = 0;
  for (let f = 0; f < nF; f++) {
    const o = f * HOP;
    for (let i = 0; i < N; i++) fr[i] = x[o + i] * win[i];
    fft.forward(fr, re, im);
    let lo = 0, hi = 0, logSum = 0, lin = 0, nb = 0;
    for (let b = k100; b < k4k; b++) lo += re[b] * re[b] + im[b] * im[b];
    for (let b = k4k; b < k12k; b++) hi += re[b] * re[b] + im[b] * im[b];
    for (let b = k500; b < k4k; b++) { const p = re[b] * re[b] + im[b] * im[b] + 1e-14; logSum += Math.log(p); lin += p; nb++; }
    const e = lo + hi;
    maxE = Math.max(maxE, e);
    frames.push({ e, hf: 10 * Math.log10((hi + 1e-14) / (lo + 1e-14)), flat: 10 * Math.log10(Math.exp(logSum / nb) / (lin / nb)) });
  }
  const loud = frames.filter((f) => f.e > maxE * 1e-3);
  const med = (a) => { if (!a.length) return 0; const s = a.slice().sort((p, q) => p - q); return s[s.length >> 1]; };
  // crest factor in 100 ms windows
  const W = Math.round(0.1 * sr);
  const crests = [];
  let maxR = 0;
  const rmsW = [];
  for (let o = 0; o + W <= x.length; o += W) {
    let pk = 0, s2 = 0;
    for (let i = o; i < o + W; i++) { const v = Math.abs(x[i]); if (v > pk) pk = v; s2 += v * v; }
    const r = Math.sqrt(s2 / W);
    rmsW.push(r);
    maxR = Math.max(maxR, r);
    if (r > 0) crests.push([r, 20 * Math.log10(pk / r)]);
  }
  const crest = med(crests.filter(([r]) => r > maxR * 0.05).map(([, c]) => c));
  // decay: energy slope over the 0.5 s after each clear rise in energy
  const slopes = [];
  const dbs = frames.map((f) => 10 * Math.log10(f.e + 1e-14));
  const per = HOP / sr;
  const span = Math.round(0.5 / per);
  for (let f = 2; f + span < frames.length; f++) {
    if (dbs[f] - dbs[f - 2] > 6 && frames[f].e > maxE * 1e-2) {
      let a = f;
      while (a + 1 < frames.length && dbs[a + 1] > dbs[a]) a++;
      if (a + span < frames.length) slopes.push((dbs[a] - dbs[a + span]) / (span * per));
      f = a + 2;
    }
  }
  return { hf: med(loud.map((f) => f.hf)), flat: med(loud.map((f) => f.flat)), crest, decay: med(slopes), frames: loud.length };
}

/**
 * The tab player's sound for a recording. Fitted by tools/fit_tone.mjs on guitars isolated by
 * BS-RoFormer (which takes away much of the top end, so the rules hold for its output only): a
 * score per kind of guitar (acoustic, clean electric, distorted electric) from the features, and
 * among distorted guitars, a line between crunch and high gain ('lead'). Data: 12 tracks of each
 * kind from MUSDB25 multitracks mixed with MUSDB18 drums, bass and vocals, plus GuitarSet
 * (acoustic) and EGSet12 (clean electric) band mixes; cross-validated: acoustic 71%, clean 74%,
 * distorted 91%.
 */
export const TONE_RULES = {
  acoustic: { b: 4.0628, hf: 0.0392, flat: -0.0745, crest: -0.4053, decay: 0.0727 },
  clean: { b: -9.49, hf: -0.0513, flat: -0.0745, crest: 0.6026, decay: 0.0089 },
  distorted: { b: 5.4272, hf: 0.0121, flat: 0.149, crest: -0.1973, decay: -0.0817 },
  lead: { b: 22.9011, flat: 0.3685, crest: -1.9244 },
};

export function guessTone(x, sr = 44100, rules = TONE_RULES) {
  const f = toneFeatures(x, sr);
  if (!rules || f.frames < 20) return { tone: 'clean', features: f };
  let best = 'clean', bestS = -Infinity;
  for (const kind of ['acoustic', 'clean', 'distorted']) {
    const w = rules[kind];
    const s = w.b + w.hf * f.hf + w.flat * f.flat + w.crest * f.crest + w.decay * f.decay;
    if (s > bestS) { bestS = s; best = kind; }
  }
  if (best === 'distorted') {
    const l = rules.lead;
    best = l && l.b + l.flat * f.flat + l.crest * f.crest > 0 ? 'lead' : 'crunch';
  }
  return { tone: best, features: f };
}
