// Estimate how far the recording's tuning is from A = 440 Hz (in cents), from spectral peaks.
import { stftMag } from './spectral.js';

export function estimateTuningCents(x, sr, opts = {}) {
  const nfft = opts.nfft || 8192, hop = opts.hop || 4096;
  const fmin = opts.fmin || 75, fmax = opts.fmax || 1600;
  const spec = stftMag(x, sr, nfft, hop);
  const { mag, frames, bins } = spec;
  const binHz = sr / nfft;
  const hist = new Float64Array(100); // 1-cent bins, -50..49
  const b0 = Math.max(2, Math.floor(fmin / binHz)), b1 = Math.min(bins - 2, Math.ceil(fmax / binHz));
  let used = 0;
  for (let t = 0; t < frames; t++) {
    const o = t * bins;
    let mx = 0;
    for (let b = b0; b <= b1; b++) if (mag[o + b] > mx) mx = mag[o + b];
    if (mx < 1e-4) continue;
    for (let b = b0; b <= b1; b++) {
      const v = mag[o + b];
      if (v < 0.2 * mx || v <= mag[o + b - 1] || v < mag[o + b + 1]) continue;
      // quadratic interpolation on log magnitude
      const a = Math.log(mag[o + b - 1] + 1e-12), c = Math.log(mag[o + b + 1] + 1e-12), m = Math.log(v + 1e-12);
      const den = a - 2 * m + c;
      const d = den < 0 ? (0.5 * (a - c)) / den : 0;
      const f = (b + d) * binHz;
      const midi = 69 + 12 * Math.log2(f / 440);
      let cents = Math.round(100 * (midi - Math.round(midi)));
      if (cents >= 50) cents -= 100;
      if (cents < -50) cents += 100;
      hist[cents + 50] += v / mx;
      used++;
    }
  }
  if (used < 20) return { cents: 0, confidence: 0 };
  // circular smoothing
  const sm = new Float64Array(100);
  for (let i = 0; i < 100; i++) for (let k = -4; k <= 4; k++) sm[i] += hist[(i + k + 100) % 100] * Math.exp(-0.5 * (k / 2.5) ** 2);
  let bi = 0;
  for (let i = 1; i < 100; i++) if (sm[i] > sm[bi]) bi = i;
  const mean = sm.reduce((s, v) => s + v, 0) / 100;
  return { cents: bi - 50, confidence: Math.min(1, (sm[bi] / (mean + 1e-9) - 1) / 3) };
}
