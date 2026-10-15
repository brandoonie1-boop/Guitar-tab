// High-resolution guitar transcription: Kong et al.'s onset/offset-regression CRNN
// (bytedance piano_transcription_inference, MIT) with weights trained for guitar by Riley,
// Edwards & Dixon (huggingface.co/xavriley/midi-transcription-models, MIT), exported to ONNX
// parts by tools/export_hrgt.py and packed like the Demucs chains.
//
// Front end = torchlibrosa Spectrogram + LogmelFilterBank as the model was trained with:
// 16 kHz mono, n_fft 2048, hop 160 (100 frames/s), periodic Hann, centered with reflect
// padding, power spectrum, 229 Slaney mel bands 30-8000 Hz (librosa.filters.mel, area
// normalised), 10*log10(max(x, 1e-10)).
//
// The four convolutional branches only look 8 frames to each side, so they run once per
// frame on short chunks; the recurrent head runs on 10 s windows (1001 frames) that overlap,
// keeping the middle of each, like the original inference code.
import { RealFFT, hannPeriodic } from './fft.js';

export const HR = { sr: 16000, nfft: 2048, hop: 160, nMels: 229, fmin: 30, fmax: 8000, fps: 100, window: 1001, context: 8, classes: 88, begin: 21 };
const BRANCHES = ['frame', 'onset', 'offset', 'velocity'];
const SILENCE_DB = -100; // 10*log10(1e-10): what zero padding looks like after the front end

function hzToMel(f) {
  const fsp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fsp, logstep = Math.log(6.4) / 27;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fsp;
}
function melToHz(m) {
  const fsp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fsp, logstep = Math.log(6.4) / 27;
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fsp * m;
}

let BANDS = null;
/** librosa.filters.mel(sr=16000, n_fft=2048, n_mels=229, fmin=30, fmax=8000) as sparse bands */
export function hrMelBands() {
  if (BANDS) return BANDS;
  const { sr, nfft, nMels, fmin, fmax } = HR;
  const nF = nfft / 2 + 1;
  const mMin = hzToMel(fmin), mMax = hzToMel(fmax);
  const f = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) f[i] = melToHz(mMin + ((mMax - mMin) * i) / (nMels + 1));
  BANDS = [];
  for (let m = 0; m < nMels; m++) {
    const lo = f[m], c = f[m + 1], hi = f[m + 2];
    const enorm = 2 / (hi - lo);
    let first = -1;
    const w = [];
    for (let i = 0; i < nF; i++) {
      const fi = (i * sr) / nfft;
      const v = Math.max(0, Math.min((fi - lo) / (c - lo), (hi - fi) / (hi - c)));
      if (v > 0) { if (first < 0) first = i; w.push(v * enorm); } else if (first >= 0) break;
    }
    BANDS.push({ first: Math.max(0, first), w: Float32Array.from(w) });
  }
  return BANDS;
}

/** Log-mel spectrogram of mono 16 kHz audio: {mel: Float32Array(T * 229), T} */
export function hrLogMel(x) {
  const { nfft, hop, nMels } = HR;
  const n = x.length;
  const pad = nfft / 2;
  const T = Math.floor(n / hop) + 1;
  const win = hannPeriodic(nfft);
  const rfft = RealFFT.get(nfft);
  const bins = nfft / 2 + 1;
  const buf = new Float64Array(nfft), re = new Float64Array(bins), im = new Float64Array(bins), pw = new Float64Array(bins);
  const bands = hrMelBands();
  const mel = new Float32Array(T * nMels);
  for (let t = 0; t < T; t++) {
    const start = t * hop - pad;
    for (let k = 0; k < nfft; k++) {
      let j = start + k;
      if (j < 0) j = -j;
      if (j >= n) j = 2 * n - 2 - j;
      buf[k] = (j >= 0 && j < n ? x[j] : 0) * win[k];
    }
    rfft.forward(buf, re, im);
    for (let b = 0; b < bins; b++) pw[b] = re[b] * re[b] + im[b] * im[b];
    const o = t * nMels;
    for (let m = 0; m < nMels; m++) {
      const { first, w } = bands[m];
      let s = 0;
      for (let k = 0; k < w.length; k++) s += w[k] * pw[first + k];
      mel[o + m] = 10 * Math.log10(Math.max(1e-10, s));
    }
  }
  return { mel, T };
}

/**
 * Sessions by part name from loadParts() (engine/demucs.js).
 * @param cpuConv optional conv_* parts on the processor, when `parts` has its convolutions on
 *   the graphics chip: each branch's first chunk is run on both and compared, and if the chip
 *   fails, returns non-finite numbers or disagrees, the run continues on the processor
 */
export function hrModel(parts, cpuConv = null) {
  const by = {};
  for (const p of parts) by[p.file] = p.session;
  for (const b of BRANCHES) if (!by[`conv_${b}`]) throw new Error(`transcription model part conv_${b} missing`);
  if (!by.head) throw new Error('transcription model part head missing');
  const all = [parts];
  const cpu = {};
  if (cpuConv) { all.push(cpuConv); for (const p of cpuConv) cpu[p.file.slice(5)] = p.session; }
  const checked = new Set();
  const model = {
    conv: Object.fromEntries(BRANCHES.map((b) => [b, by[`conv_${b}`]])),
    head: by.head,
    onGpu: !!cpuConv,
    toCpu(err) {
      console.warn('note model: the GPU result was not usable, continuing on the processor:', (err && err.message) || err);
      for (const b of BRANCHES) model.conv[b] = cpu[b];
      model.onGpu = false;
      model.gpuError = String((err && err.message) || err);
    },
    async runConv(branch, feeds) {
      if (!model.onGpu) return model.conv[branch].run(feeds);
      let r;
      try {
        r = await model.conv[branch].run(feeds);
        const d = r.feat.data;
        for (let i = 0; i < d.length; i += 97) if (!Number.isFinite(d[i])) throw new Error('non-finite output');
        if (!checked.has(branch)) {
          checked.add(branch);
          const ref = (await cpu[branch].run(feeds)).feat.data;
          let err = 0, mag = 0;
          for (let i = 0; i < ref.length; i++) { err = Math.max(err, Math.abs(ref[i] - d[i])); mag = Math.max(mag, Math.abs(ref[i])); }
          if (err > 1e-3 * (1 + mag)) throw new Error(`differs from the processor by ${err.toPrecision(3)}`);
        }
        return r;
      } catch (err) {
        if (err && err.message === 'cancelled') throw err;
        model.toCpu(err);
        return model.conv[branch].run(feeds);
      }
    },
    async release() {
      for (const ps of all) for (const p of ps) { try { await p.session.release(); } catch { /* ignore */ } }
      all.length = 0;
    },
  };
  return model;
}

/**
 * Frame-wise model outputs for mono 16 kHz audio.
 * @returns {{T, onset, offset, frame, velocity}} Float32Array(T * 88) each, 100 frames/s
 * @param opts.hop window step in frames (500 = the original's 50 % overlap; 750 is cheaper)
 * @param opts.chunk frames per convolution run
 */
export async function hrActivations(ort, model, x16, opts = {}) {
  const { onProgress, shouldCancel } = opts;
  const W = HR.window, K = HR.context, nM = HR.nMels, C = HR.classes;
  const H = Math.min(W - 1, opts.hop || 500);
  const M = Math.floor((W - 1 - H) / 2);
  const CH = opts.chunk || 250;
  const { mel, T } = hrLogMel(x16);
  let nWin = 1;
  while ((nWin - 1) * H + W - 1 < T) nWin++;
  // near-silent stretches (no guitar) are skipped: whole 10 s windows, and 2.5 s chunks of
  // the convolutions (filled with what the network gives for silence)
  const loud = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let m = -1e9;
    for (let k = t * nM, e = k + nM; k < e; k++) if (mel[k] > m) m = mel[k];
    loud[t] = m;
  }
  const ref = loud.slice().sort()[Math.floor(0.95 * (T - 1))];
  const quiet = ref - (opts.silenceDb ?? 50);
  const silentWindow = (s) => {
    for (let t = s; t < Math.min(T, s + W); t++) if (loud[t] > quiet) return false;
    return true;
  };
  const Tpad = (nWin - 1) * H + W; // frames the windows cover; past T is silence
  const melAt = (t) => (t < T ? mel.subarray(t * nM, (t + 1) * nM) : null);
  const out = { T, onset: new Float32Array(T * C), offset: new Float32Array(T * C), frame: new Float32Array(T * C), velocity: new Float32Array(T * C) };
  const FEAT = 768;
  // conv features per chunk, in a small pool of reused buffers (a window spans at most
  // ceil(W / CH) + 1 chunks): no per-chunk garbage beyond what the runtime returns
  const chunks = new Map(); // chunk index -> slot
  const nSlots = Math.ceil(W / CH) + 2;
  const slots = Array.from({ length: nSlots }, () => ({ free: true, len: 0, f: Object.fromEntries(BRANCHES.map((b) => [b, new Float32Array(CH * FEAT)])) }));
  const inpBuf = new Float32Array((CH + 2 * K) * nM);
  const runConv = (br, feeds) => (model.runConv ? model.runConv(br, feeds) : model.conv[br].run(feeds));
  // what the convolutions give for silence, computed once; near-silent chunks get this
  let silentFeat = null;
  const silenceFeature = async () => {
    if (silentFeat) return silentFeat;
    const n = 2 * K + 1;
    const feeds = { mel: new ort.Tensor('float32', new Float32Array(n * nM).fill(SILENCE_DB), [1, n, nM]) };
    silentFeat = {};
    for (const br of BRANCHES) {
      const r = await runConv(br, feeds);
      silentFeat[br] = r.feat.data.slice(K * FEAT, (K + 1) * FEAT);
      if (r.feat.dispose) r.feat.dispose();
    }
    return silentFeat;
  };
  const convChunk = async (c) => {
    let slot = slots.find((x) => x.free);
    if (!slot) { slot = { free: true, len: 0, f: Object.fromEntries(BRANCHES.map((b) => [b, new Float32Array(CH * FEAT)])) }; slots.push(slot); }
    slot.free = false;
    const a = c * CH, b = Math.min(Tpad, a + CH);
    slot.len = b - a;
    const lo = Math.max(0, a - K), hi = Math.min(Tpad, b + K);
    let silent = true;
    for (let t = lo; t < Math.min(hi, T) && silent; t++) if (loud[t] > quiet) silent = false;
    if (silent) {
      const sf = await silenceFeature();
      for (const br of BRANCHES) for (let t = 0; t < b - a; t++) slot.f[br].set(sf[br], t * FEAT);
      return slot;
    }
    const inp = inpBuf.subarray(0, (hi - lo) * nM);
    for (let t = lo; t < hi; t++) {
      const row = melAt(t);
      if (row) inp.set(row, (t - lo) * nM); else inp.fill(SILENCE_DB, (t - lo) * nM, (t - lo + 1) * nM);
    }
    const feeds = { mel: new ort.Tensor('float32', inp, [1, hi - lo, nM]) };
    for (const br of BRANCHES) {
      const r = await runConv(br, feeds);
      const f = r.feat;
      slot.f[br].set(f.data.subarray((a - lo) * FEAT, (b - lo) * FEAT));
      if (f.dispose) f.dispose();
    }
    return slot;
  };
  const winFeat = { };
  for (const br of BRANCHES) winFeat[br] = new Float32Array(W * FEAT);
  for (let k = 0; k < nWin; k++) {
    if (shouldCancel && shouldCancel()) throw new Error('cancelled');
    const s = k * H;
    if (silentWindow(s)) { if (onProgress) onProgress((k + 1) / nWin); continue; }
    const c0 = Math.floor(s / CH), c1 = Math.floor((s + W - 1) / CH);
    for (const [key, slot] of [...chunks]) if (key < c0) { slot.free = true; chunks.delete(key); }
    for (let c = c0; c <= c1; c++) if (!chunks.has(c)) chunks.set(c, await convChunk(c));
    for (const br of BRANCHES) {
      const dst = winFeat[br];
      for (let c = c0; c <= c1; c++) {
        const slot = chunks.get(c);
        const src = slot.f[br];
        const a = Math.max(s, c * CH), b = Math.min(s + W, c * CH + slot.len);
        dst.set(src.subarray((a - c * CH) * FEAT, (b - c * CH) * FEAT), (a - s) * FEAT);
      }
    }
    const feeds = {};
    for (const br of BRANCHES) feeds[`feat_${br}`] = new ort.Tensor('float32', winFeat[br], [1, W, FEAT]);
    const r = await model.head.run(feeds);
    const keepA = k === 0 ? 0 : M, keepB = k === nWin - 1 ? W : M + H;
    for (const name of ['onset', 'offset', 'frame', 'velocity']) {
      const src = r[name].data;
      const dst = out[name];
      const a = s + keepA, b = Math.min(T, s + keepB);
      if (b > a) dst.set(src.subarray(keepA * C, (keepA + b - a) * C), a * C);
      if (r[name].dispose) r[name].dispose();
    }
    if (onProgress) onProgress((k + 1) / nWin);
  }
  return out;
}

// RegressionPostProcessor (piano_transcription_inference/utilities.py) ---------------------

function binarize(reg, T, C, k, threshold, neighbour, bin, shift) {
  bin.fill(0);
  shift.fill(0);
  const x = (t) => reg[t * C + k];
  for (let n = neighbour; n < T - neighbour; n++) {
    const v = x(n);
    if (!(v > threshold)) continue;
    let mono = true;
    for (let i = 0; i < neighbour; i++) {
      if (x(n - i) < x(n - i - 1) || x(n + i) < x(n + i + 1)) { mono = false; break; }
    }
    if (!mono) continue;
    bin[n] = 1;
    const a = x(n - 1), b = x(n + 1);
    shift[n] = a > b ? (b - a) / (v - b) / 2 : (b - a) / (v - a) / 2;
  }
}

/**
 * Note events from the frame-wise outputs (onset regression peaks, then offsets from the
 * offset regression and the frame activation), with sub-frame onset/offset times.
 * @returns [{t0, t1, pitch, velocity, amp (mean frame activation), onsetProb}]
 */
export function hrNotes(act, opts = {}) {
  const onsetThr = opts.onsetThreshold ?? 0.3, offsetThr = opts.offsetThreshold ?? 0.3, frameThr = opts.frameThreshold ?? 0.1;
  const { T } = act;
  const C = HR.classes, fps = HR.fps;
  const on = new Uint8Array(T), onShift = new Float32Array(T), off = new Uint8Array(T), offShift = new Float32Array(T);
  const notes = [];
  for (let k = 0; k < C; k++) {
    binarize(act.onset, T, C, k, onsetThr, 2, on, onShift);
    binarize(act.offset, T, C, k, offsetThr, 4, off, offShift);
    let bgn = -1, frameGone = -1, offAt = -1;
    const emit = (fin, offsetShift) => {
      let s = 0;
      for (let t = bgn; t < Math.max(bgn + 1, fin); t++) s += act.frame[t * C + k];
      notes.push({
        t0: (bgn + onShift[bgn]) / fps, t1: (fin + offsetShift) / fps, pitch: k + HR.begin,
        velocity: act.velocity[bgn * C + k], amp: s / Math.max(1, fin - bgn), onsetProb: act.onset[bgn * C + k],
      });
    };
    for (let i = 0; i < T; i++) {
      if (on[i]) {
        if (bgn >= 0) { emit(Math.max(i - 1, 0), 0); frameGone = -1; offAt = -1; }
        bgn = i;
      }
      if (bgn >= 0 && i > bgn) {
        if (act.frame[i * C + k] <= frameThr && frameGone < 0) frameGone = i;
        if (off[i] && offAt < 0) offAt = i;
        if (frameGone >= 0) {
          const fin = offAt >= 0 && offAt - bgn > frameGone - offAt ? offAt : frameGone;
          emit(fin, offShift[fin]);
          bgn = -1; frameGone = -1; offAt = -1;
        }
        if (bgn >= 0 && (i - bgn >= 600 || i === T - 1)) {
          emit(i, offShift[i]);
          bgn = -1; frameGone = -1; offAt = -1;
        }
      }
    }
  }
  notes.sort((a, b) => a.t0 - b.t0 || a.pitch - b.pitch);
  return notes;
}
