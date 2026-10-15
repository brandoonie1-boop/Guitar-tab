// Split a (separated) guitar stem by stereo position. Many recordings pan two guitars
// hard left and right; this pulls them apart so each gets its own tab.
// Works frame by frame in two passes (statistics, then only the parts needed), so it never
// holds a whole-song spectrogram: a 4-minute song would otherwise need ~850 MB here.
import { RealFFT, hannPeriodic } from './fft.js';
import { frameCount } from './spectral.js';

function smooth(x, lo, hi) {
  if (x <= lo) return 0;
  if (x >= hi) return 1;
  const t = (x - lo) / (hi - lo);
  return t * t * (3 - 2 * t);
}

// both channels' STFT frames (periodic Hann, centered, reflect padding), one frame at a time
function framesLR(L, R, nfft, hop, fn) {
  const win = hannPeriodic(nfft);
  const rfft = RealFFT.get(nfft);
  const bins = nfft / 2 + 1;
  const pad = nfft / 2;
  const n = L.length;
  const frames = frameCount(n, nfft, hop);
  const bufL = new Float64Array(nfft), bufR = new Float64Array(nfft);
  const lr = new Float64Array(bins), li = new Float64Array(bins), rr = new Float64Array(bins), ri = new Float64Array(bins);
  for (let t = 0; t < frames; t++) {
    const start = t * hop - pad;
    for (let k = 0; k < nfft; k++) {
      let j = start + k;
      if (j < 0) j = -j;
      if (j >= n) j = 2 * n - 2 - j;
      const ok = j >= 0 && j < n;
      bufL[k] = (ok ? L[j] : 0) * win[k];
      bufR[k] = (ok ? R[j] : 0) * win[k];
    }
    rfft.forward(bufL, lr, li);
    rfft.forward(bufR, rr, ri);
    fn(t, lr, li, rr, ri);
  }
  return frames;
}

// Left / right / center masks for one bin from the two channel magnitudes, written into
// out[o..o+2]. (No array per bin: a whole song is ~50 million bins, and allocating there made
// Safari's memory balloon.)
function masks(a, b, out, o) {
  const pan = (b - a) / (a + b + 1e-9); // -1 left .. +1 right
  const mL = 1 - smooth(pan, -0.55, -0.2);
  const mR = smooth(pan, 0.2, 0.55);
  out[o] = mL; out[o + 1] = mR; out[o + 2] = Math.max(0, 1 - mL - mR);
}

export function splitStereo(L, R, opts = {}) {
  const nfft = opts.nfft || 2048, hop = opts.hop || 512;
  const bins = nfft / 2 + 1;
  const n = L.length;
  const frames = frameCount(n, nfft, hop);
  // pass 1: how much energy sits left / right / center, and how alike the two sides move
  let eL = 0, eR = 0, eC = 0;
  const profL = new Float32Array(frames), profR = new Float32Array(frames);
  const m3 = new Float64Array(3);
  framesLR(L, R, nfft, hop, (t, lr, li, rr, ri) => {
    let fl = 0, fr = 0;
    for (let b = 0; b < bins; b++) {
      const a = Math.hypot(lr[b], li[b]), c = Math.hypot(rr[b], ri[b]);
      masks(a, c, m3, 0);
      const mL = m3[0], mR = m3[1], mC = m3[2];
      const pl = mL * a * mL * a, pr = mR * c * mR * c;
      const sr = lr[b] + rr[b], si = li[b] + ri[b];
      eL += pl; eR += pr; eC += mC * mC * 0.25 * (sr * sr + si * si);
      fl += pl; fr += pr;
    }
    profL[t] = fl; profR[t] = fr;
  });
  const tot = eL + eR + eC + 1e-12;
  // correlation of the left/right energy envelopes: high = probably the same part double-tracked
  let ml = 0, mr = 0;
  for (let t = 0; t < frames; t++) { profL[t] = Math.log1p(1e4 * profL[t]); profR[t] = Math.log1p(1e4 * profR[t]); ml += profL[t]; mr += profR[t]; }
  ml /= frames; mr /= frames;
  let sxy = 0, sxx = 0, syy = 0;
  for (let t = 0; t < frames; t++) { const x = profL[t] - ml, y = profR[t] - mr; sxy += x * y; sxx += x * x; syy += y * y; }
  const corr = sxy / Math.sqrt(sxx * syy + 1e-12);

  // pass 2: rebuild only the requested parts (mono each) by masked overlap-add
  const render = (names) => {
    const want = new Set(names);
    const win = hannPeriodic(nfft);
    const rfft = RealFFT.get(nfft);
    const pad = nfft / 2;
    const olen = (frames - 1) * hop + nfft;
    const ys = {};
    for (const k of want) ys[k] = new Float32Array(olen);
    const re = new Float64Array(bins), im = new Float64Array(bins), buf = new Float64Array(nfft);
    const mk = new Float64Array(3 * bins);
    const addFrame = (y, t) => {
      im[0] = 0; im[bins - 1] = 0;
      rfft.inverse(re, im, buf);
      const o = t * hop;
      for (let k = 0; k < nfft; k++) y[o + k] += buf[k] * win[k];
    };
    framesLR(L, R, nfft, hop, (t, lr, li, rr, ri) => {
      // masks once per frame, reused for every part
      for (let b = 0; b < bins; b++) masks(Math.hypot(lr[b], li[b]), Math.hypot(rr[b], ri[b]), mk, 3 * b);
      if (ys.left) { for (let b = 0; b < bins; b++) { re[b] = mk[3 * b] * lr[b]; im[b] = mk[3 * b] * li[b]; } addFrame(ys.left, t); }
      if (ys.right) { for (let b = 0; b < bins; b++) { re[b] = mk[3 * b + 1] * rr[b]; im[b] = mk[3 * b + 1] * ri[b]; } addFrame(ys.right, t); }
      if (ys.center) { for (let b = 0; b < bins; b++) { const c = mk[3 * b + 2] * 0.5; re[b] = c * (lr[b] + rr[b]); im[b] = c * (li[b] + ri[b]); } addFrame(ys.center, t); }
    });
    // window-squared normalisation (same as istft), then drop the padding
    const w2 = Float64Array.from(win, (v) => v * v);
    const out = {};
    for (const [k, y] of Object.entries(ys)) {
      const o = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const j = i + pad;
        const tMin = Math.max(0, Math.ceil((j - nfft + 1) / hop)), tMax = Math.min(frames - 1, Math.floor(j / hop));
        let env = 0;
        for (let t = tMin; t <= tMax; t++) env += w2[j - t * hop];
        o[i] = env > 1e-8 ? y[j] / env : 0;
      }
      out[k] = o;
    }
    return out;
  };
  return {
    stats: { left: eL / tot, right: eR / tot, center: eC / tot, corr },
    render,
    left: () => render(['left']).left,
    right: () => render(['right']).right,
    center: () => render(['center']).center,
  };
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
