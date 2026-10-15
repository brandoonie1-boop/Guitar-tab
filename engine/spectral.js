// Spectral helpers: magnitude STFT, mel filterbank, chroma.
import { RealFFT, hannSymmetric, hannPeriodic } from './fft.js';

/** Number of frames stftFrames()/stftMag() produce for n samples. */
export function frameCount(n, nfft, hop, center = true) {
  const pad = center ? nfft / 2 : 0;
  return Math.max(1, Math.floor((n + 2 * pad - nfft) / hop) + 1);
}

/**
 * Walk the STFT frame by frame without storing it (periodic Hann, reflect-padded, centered,
 * identical to stftMag/stft). fn(t, re, im) gets Float64Arrays it must not keep.
 */
export function stftFrames(x, nfft, hop, fn, center = true) {
  const win = hannPeriodic(nfft);
  const rfft = RealFFT.get(nfft);
  const bins = nfft / 2 + 1;
  const pad = center ? nfft / 2 : 0;
  const n = x.length;
  const frames = frameCount(n, nfft, hop, center);
  const buf = new Float64Array(nfft);
  const re = new Float64Array(bins), im = new Float64Array(bins);
  for (let t = 0; t < frames; t++) {
    const start = t * hop - pad;
    for (let k = 0; k < nfft; k++) {
      let j = start + k;
      if (j < 0) j = -j;
      if (j >= n) j = 2 * n - 2 - j;
      buf[k] = (j >= 0 && j < n ? x[j] : 0) * win[k];
    }
    rfft.forward(buf, re, im);
    fn(t, re, im);
  }
  return frames;
}

/** Magnitude spectrogram. Returns {mag: Float32Array (frames*bins), frames, bins, hop, sr, nfft}. */
export function stftMag(x, sr, nfft = 2048, hop = 512, center = true) {
  const win = hannPeriodic(nfft);
  const rfft = RealFFT.get(nfft);
  const bins = nfft / 2 + 1;
  const pad = center ? nfft / 2 : 0;
  const n = x.length;
  const frames = Math.max(1, Math.floor((n + 2 * pad - nfft) / hop) + 1);
  const mag = new Float32Array(frames * bins);
  const buf = new Float64Array(nfft);
  const re = new Float64Array(bins), im = new Float64Array(bins);
  for (let t = 0; t < frames; t++) {
    const start = t * hop - pad;
    for (let k = 0; k < nfft; k++) {
      let j = start + k;
      // reflect padding
      if (j < 0) j = -j;
      if (j >= n) j = 2 * n - 2 - j;
      buf[k] = (j >= 0 && j < n ? x[j] : 0) * win[k];
    }
    rfft.forward(buf, re, im);
    const o = t * bins;
    for (let b = 0; b < bins; b++) mag[o + b] = Math.hypot(re[b], im[b]);
  }
  return { mag, frames, bins, hop, sr, nfft };
}

function hzToMel(f) { return 2595 * Math.log10(1 + f / 700); }
function melToHz(m) { return 700 * (Math.pow(10, m / 2595) - 1); }

/** Triangular mel filterbank: array of {start, weights} per band. */
export function melFilterbank(sr, nfft, nMels = 64, fmin = 30, fmax = 8000) {
  const bins = nfft / 2 + 1;
  fmax = Math.min(fmax, sr / 2);
  const mMin = hzToMel(fmin), mMax = hzToMel(fmax);
  const pts = [];
  for (let i = 0; i < nMels + 2; i++) pts.push(melToHz(mMin + ((mMax - mMin) * i) / (nMels + 1)));
  const binHz = sr / nfft;
  const bands = [];
  for (let m = 0; m < nMels; m++) {
    const f0 = pts[m], f1 = pts[m + 1], f2 = pts[m + 2];
    const b0 = Math.max(0, Math.floor(f0 / binHz)), b2 = Math.min(bins - 1, Math.ceil(f2 / binHz));
    const w = [];
    for (let b = b0; b <= b2; b++) {
      const f = b * binHz;
      let v = 0;
      if (f >= f0 && f <= f1) v = (f - f0) / Math.max(1e-9, f1 - f0);
      else if (f > f1 && f <= f2) v = (f2 - f) / Math.max(1e-9, f2 - f1);
      w.push(v);
    }
    bands.push({ start: b0, weights: Float32Array.from(w), center: f1 });
  }
  return bands;
}

/**
 * Mel power spectrogram computed frame by frame (same result as
 * applyFilterbank(stftMag(x, ...), bands) without holding the full spectrogram).
 */
export function melPower(x, sr, nfft, hop, bands) {
  const nB = bands.length;
  const frames = frameCount(x.length, nfft, hop);
  const out = new Float32Array(frames * nB);
  stftFrames(x, nfft, hop, (t, re, im) => {
    for (let m = 0; m < nB; m++) {
      const { start, weights } = bands[m];
      let s = 0;
      for (let k = 0; k < weights.length; k++) {
        // same rounding as the stored path: magnitude kept as float32, then squared
        const v = Math.fround(Math.hypot(re[start + k], im[start + k]));
        s += weights[k] * v * v;
      }
      out[t * nB + m] = s;
    }
  });
  return { mel: out, frames, nB };
}

export function applyFilterbank(spec, bands) {
  const { mag, frames, bins } = spec;
  const nB = bands.length;
  const out = new Float32Array(frames * nB);
  for (let t = 0; t < frames; t++) {
    const o = t * bins;
    for (let m = 0; m < nB; m++) {
      const { start, weights } = bands[m];
      let s = 0;
      for (let k = 0; k < weights.length; k++) { const v = mag[o + start + k]; s += weights[k] * v * v; }
      out[t * nB + m] = s;
    }
  }
  return out;
}

/** Complex STFT (periodic Hann, centered). Returns {re, im, frames, bins, hop, nfft, n}. */
export function stft(x, nfft = 2048, hop = 512) {
  const win = hannPeriodic(nfft);
  const rfft = RealFFT.get(nfft);
  const bins = nfft / 2 + 1;
  const pad = nfft / 2;
  const n = x.length;
  const frames = Math.floor((n + 2 * pad - nfft) / hop) + 1;
  const re = new Float32Array(frames * bins), im = new Float32Array(frames * bins);
  const buf = new Float64Array(nfft), r = new Float64Array(bins), i2 = new Float64Array(bins);
  for (let t = 0; t < frames; t++) {
    const start = t * hop - pad;
    for (let k = 0; k < nfft; k++) {
      let j = start + k;
      if (j < 0) j = -j;
      if (j >= n) j = 2 * n - 2 - j;
      buf[k] = (j >= 0 && j < n ? x[j] : 0) * win[k];
    }
    rfft.forward(buf, r, i2);
    re.set(r, t * bins); im.set(i2, t * bins);
  }
  return { re, im, frames, bins, hop, nfft, n };
}

/** Inverse of stft() with window-squared normalisation. */
export function istft(spec, n) {
  const { re, im, frames, bins, hop, nfft } = spec;
  const win = hannPeriodic(nfft);
  const rfft = RealFFT.get(nfft);
  const pad = nfft / 2;
  const olen = (frames - 1) * hop + nfft;
  const y = new Float64Array(olen), env = new Float64Array(olen);
  const r = new Float64Array(bins), i2 = new Float64Array(bins), buf = new Float64Array(nfft);
  for (let t = 0; t < frames; t++) {
    for (let b = 0; b < bins; b++) { r[b] = re[t * bins + b]; i2[b] = im[t * bins + b]; }
    i2[0] = 0; i2[bins - 1] = 0;
    rfft.inverse(r, i2, buf);
    const o = t * hop;
    for (let k = 0; k < nfft; k++) { y[o + k] += buf[k] * win[k]; env[o + k] += win[k] * win[k]; }
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) { const j = i + pad; out[i] = env[j] > 1e-8 ? y[j] / env[j] : 0; }
  return out;
}
