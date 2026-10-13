// Source separation with Demucs v4 (htdemucs_6s), running the exported ONNX "core"
// (encoder/transformer/decoder) and doing the STFT / iSTFT here in JS, exactly as
// demucs.htdemucs does in PyTorch. Validated against the PyTorch model.

import { RealFFT, hannPeriodic } from './fft.js';

export const DEMUCS = {
  sr: 44100,
  segment: 343980, // 7.8 s training segment
  nfft: 4096,
  hop: 1024,
  bins: 2048,
  frames: 336,
  sources: ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
};

const NFFT = DEMUCS.nfft, HOP = DEMUCS.hop, BINS = DEMUCS.bins, FR = DEMUCS.frames, SEG = DEMUCS.segment;
const PAD = (HOP / 2) * 3; // 1536
const P1LEN = SEG + PAD + (PAD + FR * HOP - SEG); // length after first reflect pad (347136)
const NFR = FR + 4; // frames produced by torch.stft(center=True) on the padded signal
let WIN = null, WIN2SUM = null;

function reflect(i, n) {
  // numpy / torch 'reflect' padding (edge sample not repeated)
  if (n === 1) return 0;
  const period = 2 * n - 2;
  i = ((i % period) + period) % period;
  return i < n ? i : period - i;
}

function initTables() {
  if (WIN) return;
  WIN = hannPeriodic(NFFT);
  // overlap-added squared window over all NFR frames (torch.istft window envelope)
  const olen = (NFR - 1) * HOP + NFFT;
  WIN2SUM = new Float64Array(olen);
  for (let f = 0; f < NFR; f++) for (let k = 0; k < NFFT; k++) WIN2SUM[f * HOP + k] += WIN[k] * WIN[k];
}

// x: Float32Array(SEG). Writes complex-as-channels into mag at channel offsets (cRe, cIm).
function specChannel(x, mag, cRe, cIm) {
  initTables();
  const rfft = RealFFT.get(NFFT);
  const frame = new Float64Array(NFFT);
  const re = new Float64Array(NFFT / 2 + 1), im = new Float64Array(NFFT / 2 + 1);
  const scale = 1 / Math.sqrt(NFFT);
  const baseRe = cRe * BINS * FR, baseIm = cIm * BINS * FR;
  for (let t = 2; t < 2 + FR; t++) {
    const start = t * HOP;
    for (let k = 0; k < NFFT; k++) {
      // index in the center-padded signal -> first padded signal -> original
      const j1 = reflect(start + k - NFFT / 2, P1LEN);
      const j0 = reflect(j1 - PAD, SEG);
      frame[k] = x[j0] * WIN[k];
    }
    rfft.forward(frame, re, im);
    const tt = t - 2;
    for (let f = 0; f < BINS; f++) {
      mag[baseRe + f * FR + tt] = re[f] * scale;
      mag[baseIm + f * FR + tt] = im[f] * scale;
    }
  }
}

// Inverse: spec arrays (Float32Array views) laid out [f][t] for re and im, weight w.
// Adds w * istft(...) into out (Float32Array SEG) .
function ispecAdd(specRe, specIm, offRe, offIm, out, w = 1) {
  initTables();
  const rfft = RealFFT.get(NFFT);
  const re = new Float64Array(NFFT / 2 + 1), im = new Float64Array(NFFT / 2 + 1);
  const buf = new Float64Array(NFFT);
  const sq = Math.sqrt(NFFT);
  const startOut = NFFT / 2 + PAD; // 3584: offset of output sample 0 in OLA coordinates
  for (let f = 2; f < 2 + FR; f++) {
    const tt = f - 2;
    for (let b = 0; b < BINS; b++) {
      re[b] = specRe[offRe + b * FR + tt] * sq;
      im[b] = specIm[offIm + b * FR + tt] * sq;
    }
    re[BINS] = 0; im[BINS] = 0;
    im[0] = 0; // irfft ignores the imaginary part of DC
    rfft.inverse(re, im, buf);
    const pos = f * HOP;
    for (let k = 0; k < NFFT; k++) {
      const o = pos + k - startOut;
      if (o < 0 || o >= SEG) continue;
      out[o] += (w * buf[k] * WIN[k]) / WIN2SUM[pos + k];
    }
  }
}

function addMonoOrStereo(acc, chunkL, chunkR, offset, len, delta, weight, mono) {
  // acc: [L] or [L, R] Float32Array accumulators (full length)
  const d0 = delta >> 1;
  if (mono) {
    const a = acc[0];
    for (let i = 0; i < len; i++) a[offset + i] += weight[i] * 0.5 * (chunkL[d0 + i] + chunkR[d0 + i]);
  } else {
    const a = acc[0], b = acc[1];
    for (let i = 0; i < len; i++) {
      a[offset + i] += weight[i] * chunkL[d0 + i];
      b[offset + i] += weight[i] * chunkR[d0 + i];
    }
  }
}

/**
 * Separate a stereo track.
 * @param {object} ort onnxruntime module (web or node)
 * @param {object} session InferenceSession of the htdemucs core
 * @param {Float32Array} left 44.1 kHz
 * @param {Float32Array} right 44.1 kHz
 * @param {object} opts { keep: {guitar:'stereo', drums:'mono', ...}, overlap, onProgress(frac), shouldCancel() }
 * @returns {Promise<object>} map source -> [Float32Array] (1 or 2 channels)
 */
export async function separate(ort, session, left, right, opts = {}) {
  const keep = opts.keep || { guitar: 'stereo', other: 'stereo', drums: 'mono' };
  const overlap = opts.overlap ?? 0.25;
  const length = left.length;
  // normalisation over the mono reference (demucs api.separate_tensor)
  let m = 0;
  for (let i = 0; i < length; i++) m += 0.5 * (left[i] + right[i]);
  m /= Math.max(1, length);
  let v = 0;
  for (let i = 0; i < length; i++) { const d = 0.5 * (left[i] + right[i]) - m; v += d * d; }
  const std = Math.sqrt(v / Math.max(1, length - 1)) + 1e-8;

  const out = {};
  for (const [name, mode] of Object.entries(keep)) {
    out[name] = mode === 'stereo' ? [new Float32Array(length), new Float32Array(length)] : [new Float32Array(length)];
  }
  const sumW = new Float32Array(length);
  const stride = Math.floor((1 - overlap) * SEG);
  const weightFull = new Float32Array(SEG);
  const halfSeg = Math.floor(SEG / 2);
  for (let i = 0; i < halfSeg; i++) weightFull[i] = i + 1;
  for (let i = halfSeg; i < SEG; i++) weightFull[i] = SEG - i;
  let wmax = 0;
  for (let i = 0; i < SEG; i++) wmax = Math.max(wmax, weightFull[i]);
  for (let i = 0; i < SEG; i++) weightFull[i] /= wmax;

  const offsets = [];
  for (let o = 0; o < length; o += stride) offsets.push(o);
  const mixL = new Float32Array(SEG), mixR = new Float32Array(SEG);
  const mag = new Float32Array(4 * BINS * FR);
  const mixData = new Float32Array(2 * SEG);
  const srcIndex = Object.fromEntries(DEMUCS.sources.map((s, i) => [s, i]));
  const tmpL = new Float32Array(SEG), tmpR = new Float32Array(SEG);

  for (let ci = 0; ci < offsets.length; ci++) {
    if (opts.shouldCancel && opts.shouldCancel()) throw new Error('cancelled');
    const offset = offsets[ci];
    const chunkLen = Math.min(SEG, length - offset);
    const delta = SEG - chunkLen;
    const start = offset - (delta >> 1);
    for (let i = 0; i < SEG; i++) {
      const j = start + i;
      if (j >= 0 && j < length) {
        mixL[i] = (left[j] - m) / std;
        mixR[i] = (right[j] - m) / std;
      } else { mixL[i] = 0; mixR[i] = 0; }
    }
    specChannel(mixL, mag, 0, 1);
    specChannel(mixR, mag, 2, 3);
    mixData.set(mixL, 0);
    mixData.set(mixR, SEG);
    const feeds = {
      mag: new ort.Tensor('float32', mag, [1, 4, BINS, FR]),
      mix: new ort.Tensor('float32', mixData, [1, 2, SEG]),
    };
    const res = await session.run(feeds);
    const spec = res.spec.data; // [1, 6, 4, BINS, FR]
    const wave = res.wave.data; // [1, 6, 2, SEG]
    const weight = weightFull; // indexed from 0 for the chunk (chunkLen <= SEG)
    for (const [name, mode] of Object.entries(keep)) {
      const s = srcIndex[name];
      tmpL.set(wave.subarray((s * 2 + 0) * SEG, (s * 2 + 1) * SEG));
      tmpR.set(wave.subarray((s * 2 + 1) * SEG, (s * 2 + 2) * SEG));
      const sBase = s * 4 * BINS * FR;
      ispecAdd(spec, spec, sBase + 0 * BINS * FR, sBase + 1 * BINS * FR, tmpL);
      ispecAdd(spec, spec, sBase + 2 * BINS * FR, sBase + 3 * BINS * FR, tmpR);
      addMonoOrStereo(out[name], tmpL, tmpR, offset, chunkLen, delta, weight, mode !== 'stereo');
    }
    for (let i = 0; i < chunkLen; i++) sumW[offset + i] += weight[i];
    if (res.spec.dispose) res.spec.dispose();
    if (res.wave.dispose) res.wave.dispose();
    if (opts.onProgress) opts.onProgress((ci + 1) / offsets.length);
    // yield to the event loop so progress messages get out
    await new Promise((r) => setTimeout(r, 0));
  }
  for (const name of Object.keys(out)) {
    for (const ch of out[name]) {
      for (let i = 0; i < length; i++) ch[i] = (ch[i] / sumW[i]) * std + m;
    }
  }
  return out;
}

// Exposed for tests
export const _internal = { specChannel, ispecAdd, reflect };
