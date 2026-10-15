// Source separation with Demucs v4 (htdemucs_6s), running the exported ONNX "core"
// (encoder/transformer/decoder) and doing the STFT / iSTFT here in JS, exactly as
// demucs.htdemucs does in PyTorch. Validated against the PyTorch model.

import { RealFFT, hannPeriodic } from './fft.js';

export const DEMUCS = {
  sr: 44100,
  segment: 343980, // 7.8 s training segment (default); shorter exports use less memory
  nfft: 4096,
  hop: 1024,
  bins: 2048,
  sources: ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
};

const NFFT = DEMUCS.nfft, HOP = DEMUCS.hop, BINS = DEMUCS.bins;
const PAD = (HOP / 2) * 3; // 1536
let WIN = null;

/** Frame geometry for a model exported with a given segment length (samples). */
const GEOMS = new Map();
export function geometry(SEG) {
  let g = GEOMS.get(SEG);
  if (g) return g;
  if (!WIN) WIN = hannPeriodic(NFFT);
  const FR = Math.ceil(SEG / HOP);
  const P1LEN = SEG + PAD + (PAD + FR * HOP - SEG); // length after the first reflect pad
  const NFR = FR + 4; // frames produced by torch.stft(center=True) on the padded signal
  // overlap-added squared window over all NFR frames (torch.istft window envelope)
  const olen = (NFR - 1) * HOP + NFFT;
  const WIN2SUM = new Float64Array(olen);
  for (let f = 0; f < NFR; f++) for (let k = 0; k < NFFT; k++) WIN2SUM[f * HOP + k] += WIN[k] * WIN[k];
  g = { SEG, FR, P1LEN, NFR, WIN2SUM };
  GEOMS.set(SEG, g);
  return g;
}

function reflect(i, n) {
  // numpy / torch 'reflect' padding (edge sample not repeated)
  if (n === 1) return 0;
  const period = 2 * n - 2;
  i = ((i % period) + period) % period;
  return i < n ? i : period - i;
}

// x: Float32Array(SEG). Writes complex-as-channels into mag at channel offsets (cRe, cIm).
function specChannel(g, x, mag, cRe, cIm) {
  const { SEG, FR, P1LEN } = g;
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

// Inverse: spec arrays laid out [f][t] for re and im. Adds istft(...) into out (Float32Array SEG).
function ispecAdd(g, specRe, specIm, offRe, offIm, out) {
  const { SEG, FR, WIN2SUM } = g;
  const rfft = RealFFT.get(NFFT);
  const re = new Float64Array(NFFT / 2 + 1), im = new Float64Array(NFFT / 2 + 1);
  const buf = new Float64Array(NFFT);
  const sq = Math.sqrt(NFFT);
  const startOut = NFFT / 2 + PAD; // offset of output sample 0 in OLA coordinates
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
      out[o] += (buf[k] * WIN[k]) / WIN2SUM[pos + k];
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
 * @param {object} opts { keep: {guitar:'stereo', drums:'mono', ...}, segment (samples the model was exported with),
 *                        overlap, onProgress(frac), shouldCancel(), check }
 * @returns {Promise<object>} map source -> [Float32Array] (1 or 2 channels)
 */
export async function separate(ort, session, left, right, opts = {}) {
  const keep = opts.keep || { guitar: 'stereo', other: 'stereo', drums: 'mono' };
  const overlap = opts.overlap ?? 0.25;
  const length = left.length;
  const g = geometry(opts.segment || session.segment || DEMUCS.segment);
  const { SEG, FR } = g;
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
  // models that can hand back only some sources (packed chains): ask for just the kept ones,
  // except on a checked chunk, which needs all six
  const allIdx = DEMUCS.sources.map((_, i) => i);
  const keptIdx = Object.keys(keep).map((n) => srcIndex[n]);
  const sourcesTensor = (idx) => new ort.Tensor('int64', BigInt64Array.from(idx.map(BigInt)), [idx.length]);

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
    specChannel(g, mixL, mag, 0, 1);
    specChannel(g, mixR, mag, 2, 3);
    mixData.set(mixL, 0);
    mixData.set(mixR, SEG);
    const feeds = {
      mag: new ort.Tensor('float32', mag, [1, 4, BINS, FR]),
      mix: new ort.Tensor('float32', mixData, [1, 2, SEG]),
    };
    const checking = opts.check && ci === 0;
    const want = session.select && !checking ? keptIdx : allIdx;
    if (session.select) feeds.sources = sourcesTensor(want);
    const res = await session.run(feeds);
    const spec = res.spec.data; // [1, k, 4, BINS, FR] (k = number of sources returned)
    const wave = res.wave.data; // [1, k, 2, SEG]
    if (checking) {
      // sanity check (used for the GPU path): the six sources should add back up to the mix
      const snr = reconstructionSnr(g, spec, wave, mixL, mixR, chunkLen, delta);
      if (opts.onCheck) opts.onCheck(snr);
      if (!(snr > (opts.minCheckSnr ?? 8))) throw new Error('separation-check-failed');
    }
    const weight = weightFull; // indexed from 0 for the chunk (chunkLen <= SEG)
    for (const [name, mode] of Object.entries(keep)) {
      const s = want.indexOf(srcIndex[name]); // position in the returned sources
      tmpL.set(wave.subarray((s * 2 + 0) * SEG, (s * 2 + 1) * SEG));
      tmpR.set(wave.subarray((s * 2 + 1) * SEG, (s * 2 + 2) * SEG));
      const sBase = s * 4 * BINS * FR;
      ispecAdd(g, spec, spec, sBase + 0 * BINS * FR, sBase + 1 * BINS * FR, tmpL);
      ispecAdd(g, spec, spec, sBase + 2 * BINS * FR, sBase + 3 * BINS * FR, tmpR);
      addMonoOrStereo(out[name], tmpL, tmpR, offset, chunkLen, delta, weight, mode !== 'stereo');
    }
    if (res.spec.dispose) res.spec.dispose();
    if (res.wave.dispose) res.wave.dispose();
    if (opts.onProgress) opts.onProgress((ci + 1) / offsets.length);
    // yield to the event loop so progress messages get out
    await new Promise((r) => setTimeout(r, 0));
  }
  // divide by the summed chunk weights (worked out per sample instead of stored: saves a
  // song-length array; same float32 accumulation order as demucs' sum_weight)
  const chans = Object.values(out).flat();
  const last = offsets.length - 1;
  for (let i = 0; i < length; i++) {
    const c0 = Math.max(0, Math.ceil((i - SEG + 1) / stride)), c1 = Math.min(last, Math.floor(i / stride));
    let w = 0;
    for (let c = c0; c <= c1; c++) w = Math.fround(w + weightFull[i - offsets[c]]);
    for (const ch of chans) ch[i] = (ch[i] / w) * std + m;
  }
  return out;
}

function reconstructionSnr(g, spec, wave, mixL, mixR, chunkLen, delta) {
  const { SEG, FR } = g;
  const S = DEMUCS.sources.length;
  const n = BINS * FR;
  const sumL = new Float32Array(SEG), sumR = new Float32Array(SEG);
  const accRe = new Float32Array(n), accIm = new Float32Array(n);
  for (const [ch, out] of [[0, sumL], [1, sumR]]) {
    accRe.fill(0); accIm.fill(0);
    for (let s = 0; s < S; s++) {
      const base = s * 4 * n;
      const re = spec.subarray(base + (2 * ch) * n, base + (2 * ch + 1) * n);
      const im = spec.subarray(base + (2 * ch + 1) * n, base + (2 * ch + 2) * n);
      for (let i = 0; i < n; i++) { accRe[i] += re[i]; accIm[i] += im[i]; }
      const w = wave.subarray((s * 2 + ch) * SEG, (s * 2 + ch + 1) * SEG);
      for (let i = 0; i < SEG; i++) out[i] += w[i];
    }
    ispecAdd(g, accRe, accIm, 0, 0, out);
  }
  const d0 = delta >> 1;
  let e = 0, r = 0;
  for (let i = d0; i < d0 + chunkLen; i++) {
    r += mixL[i] * mixL[i] + mixR[i] * mixR[i];
    e += (mixL[i] - sumL[i]) ** 2 + (mixR[i] - sumR[i]) ** 2;
  }
  if (r < 1e-9) return 99; // silence: nothing to check
  return 10 * Math.log10(r / Math.max(1e-12, e));
}

/**
 * Read a packed Demucs chain (tools/pack_demucs.py): the model split into parts that run
 * one after another (encoder layers, transformer, decoder layers). Between parts the
 * hand-over tensors leave the WebAssembly heap, which keeps its peak size low.
 */
export function parseChain(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8.length < 8 || String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== 'DMCH') throw new Error('not a Demucs chain file');
  const n = new DataView(u8.buffer, u8.byteOffset, u8.byteLength).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  const base = 8 + n;
  header.parts = header.parts.map((p) => ({ ...p, bytes: u8.subarray(base + p.offset, base + p.offset + p.length) }));
  return header;
}

let F16 = null;
function f16Table() {
  if (F16) return F16;
  F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sgn = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    F16[h] = e === 0 ? sgn * f * 2 ** -24 : e === 31 ? (f ? NaN : sgn * Infinity) : sgn * (1 + f / 1024) * 2 ** (e - 15);
  }
  return F16;
}

// A part's float32 weights, built from the shared (mostly float16) weights file.
function partData(weights, data) {
  const out = new Uint8Array(data.size);
  const table = f16Table();
  for (const [src, len, dst, kind] of data.tensors) {
    if (kind === 'h') {
      const h = new Uint16Array(weights.buffer, weights.byteOffset + src, len / 2);
      const f = new Float32Array(out.buffer, dst, len / 2);
      for (let i = 0; i < h.length; i++) f[i] = table[h[i]];
    } else {
      out.set(weights.subarray(src, src + len), dst);
    }
  }
  return out;
}

/**
 * Create an InferenceSession for every part of a packed chain (see tools/pack_demucs.py).
 * Returns the parts in order ({file, in, out, session}); on failure the ones already made
 * are released.
 * @param {object} ort onnxruntime module
 * @param {object} chain parseChain() result
 * @param {Uint8Array} weights contents of the shared weights file
 * @param {object|function} sessionOptions passed to every InferenceSession.create, or a
 *   function (part) => options (null skips that part)
 */
export async function loadParts(ort, chain, weights, sessionOptions = {}, onProgress) {
  const parts = [];
  if (weights.byteOffset % 4) weights = weights.slice();
  try {
    for (let i = 0; i < chain.parts.length; i++) {
      const p = chain.parts[i];
      const so = typeof sessionOptions === 'function' ? sessionOptions(p) : sessionOptions;
      if (!so) continue; // part not wanted
      // older packs referenced the shared file directly; current ones get float32 data per part
      const externalData = p.data
        ? [{ path: p.data.path, data: partData(weights, p.data) }]
        : [{ path: chain.weights, data: weights }];
      const session = await ort.InferenceSession.create(p.bytes, { ...so, externalData });
      externalData.length = 0;
      parts.push({ file: p.file, in: p.in, out: p.out, session });
      if (onProgress) onProgress((i + 1) / chain.parts.length);
    }
  } catch (err) {
    for (const p of parts) { try { await p.session.release(); } catch { /* ignore */ } }
    throw err;
  }
  return parts;
}

/**
 * Create a session for every part of a chain. Returns an object with the same run()
 * interface as a single InferenceSession of the whole model.
 */
export async function loadChain(ort, chain, weights, sessionOptions = {}, onProgress) {
  const parts = await loadParts(ort, chain, weights, sessionOptions, onProgress);
  const lastUse = {};
  parts.forEach((p, i) => { for (const name of Object.values(p.in)) lastUse[name] = i; });
  return {
    segment: chain.segment,
    select: !!chain.select,
    async run(feeds) {
      const env = { ...feeds };
      if (chain.select && !env.sources) {
        env.sources = new ort.Tensor('int64', BigInt64Array.from(chain.sources.map((_, i) => BigInt(i))), [chain.sources.length]);
      }
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        const f = {};
        for (const [g, name] of Object.entries(p.in)) f[g] = env[name];
        const res = await p.session.run(f);
        for (const [name, li] of Object.entries(lastUse)) {
          if (li !== i || name in feeds) continue;
          if (env[name] && env[name].dispose) env[name].dispose();
          delete env[name];
        }
        for (const [g, name] of Object.entries(p.out)) env[name] = res[g];
      }
      return { spec: env.spec, wave: env.wave };
    },
    async release() {
      for (const p of parts) { try { await p.session.release(); } catch { /* ignore */ } }
      parts.length = 0;
    },
  };
}

// Exposed for tests
export const _internal = { specChannel, ispecAdd, reflect, geometry };
