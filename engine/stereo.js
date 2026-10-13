// Split a (separated) guitar stem by stereo position. Many recordings pan two guitars
// hard left and right; this pulls them apart so each gets its own tab.
import { stft, istft } from './spectral.js';

function smooth(x, lo, hi) {
  if (x <= lo) return 0;
  if (x >= hi) return 1;
  const t = (x - lo) / (hi - lo);
  return t * t * (3 - 2 * t);
}

export function splitStereo(L, R, opts = {}) {
  const nfft = opts.nfft || 2048, hop = opts.hop || 512;
  const n = L.length;
  const SL = stft(L, nfft, hop), SR = stft(R, nfft, hop);
  const { frames, bins } = SL;
  const lRe = new Float32Array(SL.re.length), lIm = new Float32Array(SL.re.length);
  const rRe = new Float32Array(SL.re.length), rIm = new Float32Array(SL.re.length);
  const cRe = new Float32Array(SL.re.length), cIm = new Float32Array(SL.re.length);
  let eL = 0, eR = 0, eC = 0;
  // per-frame energy profiles for similarity check
  const profL = new Float32Array(frames), profR = new Float32Array(frames);
  for (let i = 0; i < SL.re.length; i++) {
    const a = Math.hypot(SL.re[i], SL.im[i]), b = Math.hypot(SR.re[i], SR.im[i]);
    const pan = (b - a) / (a + b + 1e-9); // -1 left .. +1 right
    const mL = 1 - smooth(pan, -0.55, -0.2);
    const mR = smooth(pan, 0.2, 0.55);
    const mC = Math.max(0, 1 - mL - mR);
    lRe[i] = mL * SL.re[i]; lIm[i] = mL * SL.im[i];
    rRe[i] = mR * SR.re[i]; rIm[i] = mR * SR.im[i];
    cRe[i] = mC * 0.5 * (SL.re[i] + SR.re[i]); cIm[i] = mC * 0.5 * (SL.im[i] + SR.im[i]);
    const pl = mL * a * mL * a, pr = mR * b * mR * b, pc = mC * mC * 0.25 * ((SL.re[i] + SR.re[i]) ** 2 + (SL.im[i] + SR.im[i]) ** 2);
    eL += pl; eR += pr; eC += pc;
    const t = Math.floor(i / bins);
    profL[t] += pl; profR[t] += pr;
  }
  const tot = eL + eR + eC + 1e-12;
  // correlation of the left/right energy envelopes: high = probably the same part double-tracked
  let ml = 0, mr = 0;
  for (let t = 0; t < frames; t++) { profL[t] = Math.log1p(1e4 * profL[t]); profR[t] = Math.log1p(1e4 * profR[t]); ml += profL[t]; mr += profR[t]; }
  ml /= frames; mr /= frames;
  let sxy = 0, sxx = 0, syy = 0;
  for (let t = 0; t < frames; t++) { const x = profL[t] - ml, y = profR[t] - mr; sxy += x * y; sxx += x * x; syy += y * y; }
  const corr = sxy / Math.sqrt(sxx * syy + 1e-12);
  const out = {
    stats: { left: eL / tot, right: eR / tot, center: eC / tot, corr },
    left: () => istft({ ...SL, re: lRe, im: lIm }, n),
    right: () => istft({ ...SR, re: rRe, im: rIm }, n),
    center: () => istft({ ...SL, re: cRe, im: cIm }, n),
  };
  return out;
}

/** Decide which parts to transcribe from the split statistics. */
export function chooseParts(stats) {
  const { left, right, center, corr } = stats;
  const parts = [];
  const sides = left > 0.18 && right > 0.18;
  if (sides) {
    parts.push('left', 'right');
    if (center > 0.3) parts.push('center');
    return { mode: corr > 0.85 ? 'doubled' : 'split', parts };
  }
  return { mode: 'single', parts: ['all'] };
}
