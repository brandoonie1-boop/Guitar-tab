// Source separation with BS-RoFormer SW (bass, drums, other, vocals, guitar, piano), or a cut of
// it that keeps only some of those stems (engine/onnxprune.js). The ONNX graph takes the STFT of
// a 4 s stereo chunk and returns the masked STFT of each stem; the STFT / iSTFT are done here
// exactly as torch.stft / torch.istft (n_fft 2048, hop 512, periodic Hann, center=True with
// reflect padding, not normalized). Validated against the PyTorch reference of the export.

import { RealFFT, hannPeriodic } from './fft.js';
import { spreadOrder, chunkEnergies } from './dsp.js';

export const BSR = {
  sr: 44100,
  chunk: 176400, // the export's fixed length (4 s, 345 frames)
  nfft: 2048,
  hop: 512,
  bins: 1025,
  frames: 345,
  stems: ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano'],
};

const C = BSR.chunk, NFFT = BSR.nfft, HOP = BSR.hop, BINS = BSR.bins, T = BSR.frames;
const PAD = NFFT / 2;
let WIN = null, WIN2SUM = null;

function tables() {
  if (WIN) return;
  WIN = hannPeriodic(NFFT);
  // overlap-added squared window of all frames (torch.istft envelope)
  WIN2SUM = new Float64Array((T - 1) * HOP + NFFT);
  for (let f = 0; f < T; f++) for (let k = 0; k < NFFT; k++) WIN2SUM[f * HOP + k] += WIN[k] * WIN[k];
}

function reflect(i, n) {
  if (n === 1) return 0;
  const period = 2 * n - 2;
  i = ((i % period) + period) % period;
  return i < n ? i : period - i;
}

// STFT of one channel (C samples) into re/im laid out [bin][frame] at offset `off`
function stftChannel(x, re, im, off) {
  tables();
  const rfft = RealFFT.get(NFFT);
  const frame = new Float64Array(NFFT);
  const fr = new Float64Array(BINS), fi = new Float64Array(BINS);
  for (let t = 0; t < T; t++) {
    const s0 = t * HOP - PAD;
    for (let k = 0; k < NFFT; k++) frame[k] = x[reflect(s0 + k, C)] * WIN[k];
    rfft.forward(frame, fr, fi);
    for (let b = 0; b < BINS; b++) {
      re[off + b * T + t] = fr[b];
      im[off + b * T + t] = fi[b];
    }
  }
}

// iSTFT of one channel ([bin][frame] at offset `off`) into out (C samples, overwritten)
function istftChannel(re, im, off, out) {
  tables();
  const rfft = RealFFT.get(NFFT);
  const fr = new Float64Array(BINS), fi = new Float64Array(BINS);
  const buf = new Float64Array(NFFT);
  const acc = new Float64Array(WIN2SUM.length);
  for (let t = 0; t < T; t++) {
    for (let b = 0; b < BINS; b++) { fr[b] = re[off + b * T + t]; fi[b] = im[off + b * T + t]; }
    fi[0] = 0; fi[BINS - 1] = 0; // irfft ignores the imaginary parts of DC and Nyquist
    rfft.inverse(fr, fi, buf);
    const p = t * HOP;
    for (let k = 0; k < NFFT; k++) acc[p + k] += buf[k] * WIN[k];
  }
  for (let i = 0; i < C; i++) {
    const w = WIN2SUM[i + PAD];
    out[i] = w > 1e-11 ? acc[i + PAD] / w : 0;
  }
}

/** Chunk weights: linear fades over the overlap, 1 in between. */
function chunkWeights(fade) {
  const w = new Float32Array(C).fill(1);
  for (let i = 0; i < fade; i++) {
    const v = (i + 1) / (fade + 1);
    w[i] = v;
    w[C - 1 - i] = v;
  }
  return w;
}

/**
 * Separate a stereo track.
 * @param {object} ort onnxruntime module
 * @param {object} session InferenceSession (inputs spec_real/spec_imag, outputs out_spec_real/
 *   out_spec_imag) with .stems = the stems it returns, in order (default: all six)
 * @param {Float32Array} left 44.1 kHz
 * @param {Float32Array} right 44.1 kHz
 * @param {object} opts { keep: {guitar: 'stereo'|'mono', ...}, overlap (0.25), onProgress(frac),
 *   shouldCancel(), check(chunkOut, chunkIn) -> throws if the output looks broken (first chunk),
 *   probe: {count, add(energies), stop()} -> look at `count` chunks spread over the song first }
 * @returns {Promise<object>} map source -> [Float32Array] (1 or 2 channels), or {alone: true} when
 *   probe.stop() said there is nothing to separate
 */
export async function separateBsr(ort, session, left, right, opts = {}) {
  const keep = opts.keep || { guitar: 'stereo' };
  const stems = session.stems || BSR.stems;
  const n = left.length;
  const overlap = opts.overlap ?? 0.25;
  const step = Math.max(1, Math.floor(C * (1 - overlap)));
  const fade = C - step;
  const w = chunkWeights(fade);
  // the song is padded (reflected) by `fade` at the start so its first sample gets full weight too
  const starts = [];
  for (let s = -fade; s < n; s += step) starts.push(s);
  const out = {};
  for (const [name, mode] of Object.entries(keep)) {
    if (!stems.includes(name)) throw new Error('separator has no ' + name + ' stem');
    out[name] = mode === 'stereo' ? [new Float32Array(n), new Float32Array(n)] : [new Float32Array(n)];
  }
  const xL = new Float32Array(C), xR = new Float32Array(C);
  const re = new Float32Array(2 * BINS * T), im = new Float32Array(2 * BINS * T);
  const yL = new Float32Array(C), yR = new Float32Array(C);
  // outside the song: its reflection, or silence for songs too short to reflect
  const mirror = n > fade + C;
  // opts.probe: a few chunks spread over the song go first, and probe.stop() can end the work
  // there (nothing to separate); the kept stems of those chunks add up in pL/pR
  const order = spreadOrder(starts.length, opts.probe ? opts.probe.count : 0);
  const nProbe = opts.probe && order.length > opts.probe.count + 1 ? opts.probe.count : 0;
  const pL = nProbe ? new Float32Array(C) : null, pR = nProbe ? new Float32Array(C) : null;
  for (let k = 0; k < order.length; k++) {
    if (opts.shouldCancel && opts.shouldCancel()) throw new Error('cancelled');
    const ci = order[k];
    const s0 = starts[ci];
    const probing = k < nProbe;
    if (probing) { pL.fill(0); pR.fill(0); }
    for (let i = 0; i < C; i++) {
      const j = s0 + i;
      const jj = j >= 0 && j < n ? j : mirror ? reflect(j, n) : -1;
      xL[i] = jj >= 0 ? left[jj] : 0;
      xR[i] = jj >= 0 ? right[jj] : 0;
    }
    stftChannel(xL, re, im, 0);
    stftChannel(xR, re, im, BINS * T);
    const res = await session.run({
      spec_real: new ort.Tensor('float32', re, [1, 2, BINS, T]),
      spec_imag: new ort.Tensor('float32', im, [1, 2, BINS, T]),
    });
    const oRe = res.out_spec_real.data, oIm = res.out_spec_imag.data;
    for (const [name, mode] of Object.entries(keep)) {
      const k = stems.indexOf(name);
      const base = k * 2 * BINS * T;
      istftChannel(oRe, oIm, base, yL);
      istftChannel(oRe, oIm, base + BINS * T, yR);
      if (k === 0 && opts.check) opts.check({ yL, yR, xL, xR });
      if (probing) for (let i = 0; i < C; i++) { pL[i] += yL[i]; pR[i] += yR[i]; }
      const [aL, aR] = out[name];
      for (let i = 0; i < C; i++) {
        const j = s0 + i;
        if (j < 0 || j >= n) continue;
        if (mode === 'stereo') { aL[j] += w[i] * yL[i]; aR[j] += w[i] * yR[i]; }
        else aL[j] += w[i] * 0.5 * (yL[i] + yR[i]);
      }
    }
    if (res.out_spec_real.dispose) res.out_spec_real.dispose();
    if (res.out_spec_imag.dispose) res.out_spec_imag.dispose();
    if (probing) {
      opts.probe.add(chunkEnergies(xL, xR, pL, pR, 0, C));
      if (k === nProbe - 1 && opts.probe.stop()) return { alone: true };
    }
    if (opts.onProgress) opts.onProgress((k + 1) / starts.length);
    await new Promise((r) => setTimeout(r, 0));
  }
  // divide by the summed chunk weights (worked out per sample instead of stored)
  const chans = Object.values(out).flat();
  for (let j = 0; j < n; j++) {
    const c0 = Math.max(0, Math.ceil((j - starts[0] - C + 1) / step)), c1 = Math.min(starts.length - 1, Math.floor((j - starts[0]) / step));
    let sw = 0;
    for (let c = c0; c <= c1; c++) sw += w[j - starts[c]];
    if (sw > 0) for (const ch of chans) ch[j] /= sw;
  }
  return out;
}

/**
 * Self-test for a graphics chip: a fixed test chunk (plucked-string tones and a little noise)
 * goes through the model and the first stem's output energy in 16 frequency bands (dB, both
 * channels) is compared with what the processor gives (BSR_CANARY, from the same model cut).
 * Some chips run the model but get the numbers wrong; this catches that before a song is
 * separated with it.
 */
export async function canaryCheck(ort, session, expected, tolDb = 1.5) {
  const x = [new Float32Array(C), new Float32Array(C)];
  let seed = 12345;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
  const tones = [[110, 0.0], [164.81, 0.7], [246.94, 1.4], [329.63, 2.1], [440, 2.8]];
  for (let i = 0; i < C; i++) {
    const t = i / BSR.sr;
    let v = 0;
    for (const [f, t0] of tones) {
      if (t < t0) continue;
      const e = Math.exp(-(t - t0) * 2.5);
      for (let h = 1; h <= 6; h++) v += (e / h) * Math.sin(2 * Math.PI * f * h * (t - t0));
    }
    x[0][i] = 0.08 * v + 0.01 * rnd();
    x[1][i] = 0.07 * v + 0.01 * rnd();
  }
  const re = new Float32Array(2 * BINS * T), im = new Float32Array(2 * BINS * T);
  stftChannel(x[0], re, im, 0);
  stftChannel(x[1], re, im, BINS * T);
  const res = await session.run({
    spec_real: new ort.Tensor('float32', re, [1, 2, BINS, T]),
    spec_imag: new ort.Tensor('float32', im, [1, 2, BINS, T]),
  });
  const oRe = res.out_spec_real.data, oIm = res.out_spec_imag.data;
  const got = [];
  const edges = Array.from({ length: 17 }, (_, i) => Math.round(2 * Math.pow(1024 / 2, i / 16)));
  for (let ch = 0; ch < 2; ch++) {
    for (let b = 0; b < 16; b++) {
      let e = 0;
      for (let k = edges[b]; k < Math.max(edges[b] + 1, edges[b + 1]); k++) {
        const o = ch * BINS * T + k * T;
        for (let t = 0; t < T; t++) e += oRe[o + t] ** 2 + oIm[o + t] ** 2;
      }
      got.push(10 * Math.log10(e + 1e-20));
    }
  }
  if (res.out_spec_real.dispose) res.out_spec_real.dispose();
  if (res.out_spec_imag.dispose) res.out_spec_imag.dispose();
  if (!expected) return { ok: true, got };
  const top = Math.max(...expected);
  let worst = 0;
  got.forEach((g, i) => { if (expected[i] > top - 60) worst = Math.max(worst, Number.isFinite(g) ? Math.abs(g - expected[i]) : 999); });
  return { ok: worst <= tolDb, worst, got };
}

// canaryCheck() on the processor, guitar stem (the same for every cut that keeps guitar first)
export const BSR_CANARY = [-70.92, -74.44, -42.69, -35.05, -31.06, -40.53, -44.5, -46.05, -50.37, -57.04, -67.02, -74.08, -73.31, -72.74, -74.23, -56.28,
  -73.77, -70.21, -36.45, -38.97, -36.61, -41.39, -46.11, -45.04, -50.08, -55.3, -69.96, -74.09, -73.54, -72.73, -73.83, -57.72];

/**
 * Sessions for a chained model (engine/onnxprune.js buildChain: 'BSRC' container), run one part
 * after another; looks like one InferenceSession to separateBsr. A tensor passed between parts is
 * let go after the last part that reads it.
 * sessionOptions: passed to every InferenceSession.create (on the processor, turn the memory arena
 * off: each part's working memory is then given back for the next part).
 */
export async function loadBsrChain(ort, bytes, sessionOptions = {}, onProgress) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== 'BSRC') throw new Error('not a separator chain');
  const n = new DataView(u8.buffer, u8.byteOffset, u8.byteLength).getUint32(4, true);
  const { parts } = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  const base = 8 + n;
  const sessions = [];
  try {
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      sessions.push(await ort.InferenceSession.create(u8.subarray(base + p.offset, base + p.offset + p.length), sessionOptions));
      if (onProgress) onProgress((i + 1) / parts.length);
    }
  } catch (err) {
    for (const s of sessions) { try { await s.release(); } catch { /* ignore */ } }
    throw err;
  }
  const lastUse = {};
  parts.forEach((p, i) => { for (const x of p.in) lastUse[x] = i; });
  return {
    kind: 'bsr',
    async run(feeds) {
      const env = { ...feeds };
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        const f = {};
        for (const x of p.in) f[x] = env[x];
        const res = await sessions[i].run(f);
        for (const x of p.in) {
          if (lastUse[x] === i && !(x in feeds)) { if (env[x] && env[x].dispose) env[x].dispose(); delete env[x]; }
        }
        for (const x of p.out) env[x] = res[x];
      }
      return { out_spec_real: env.out_spec_real, out_spec_imag: env.out_spec_imag };
    },
    async release() {
      for (const s of sessions) { try { await s.release(); } catch { /* ignore */ } }
      sessions.length = 0;
    },
  };
}

export const _internal = { stftChannel, istftChannel, reflect, chunkWeights };
