// Beats and downbeats with "Beat This!" (Foscarin, Schlüter, Widmer, ISMIR 2024; CPJKU/beat_this,
// MIT license), small model exported to ONNX by tools/export_beat_this.py.
// The log-mel front end below reproduces beat_this.preprocessing.LogMelSpect (torchaudio):
// 22.05 kHz mono, n_fft 1024, hop 441 (50 frames/s), periodic Hann, centered with reflect
// padding, magnitude / sqrt(n_fft), 128 Slaney-scale mel bands 30-11000 Hz (no area
// normalisation), then log1p(1000 x).
import { RealFFT, hannPeriodic } from './fft.js';
import { resample } from './dsp.js';

export const BT = { sr: 22050, nfft: 1024, hop: 441, nMels: 128, fmin: 30, fmax: 11000, frames: 1500, border: 6, fps: 50 };

function hzToMelSlaney(f) {
  const fsp = 200 / 3;
  const minLogHz = 1000, minLogMel = minLogHz / fsp, logstep = Math.log(6.4) / 27;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fsp;
}
function melToHzSlaney(m) {
  const fsp = 200 / 3;
  const minLogHz = 1000, minLogMel = minLogHz / fsp, logstep = Math.log(6.4) / 27;
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fsp * m;
}

let FB = null;
/** torchaudio.functional.melscale_fbanks(513, 30, 11000, 128, 22050, norm=None, mel_scale='slaney'), [freq][mel] */
export function melFilters() {
  if (FB) return FB;
  const nF = BT.nfft / 2 + 1, nM = BT.nMels;
  const all = new Float64Array(nF);
  for (let i = 0; i < nF; i++) all[i] = (Math.floor(BT.sr / 2) * i) / (nF - 1);
  const mMin = hzToMelSlaney(BT.fmin), mMax = hzToMelSlaney(BT.fmax);
  const fpts = new Float64Array(nM + 2);
  for (let i = 0; i < nM + 2; i++) fpts[i] = melToHzSlaney(mMin + ((mMax - mMin) * i) / (nM + 1));
  const fb = new Float32Array(nF * nM);
  // sparse form: for each mel band the first bin and its weights
  const bands = [];
  for (let m = 0; m < nM; m++) {
    const lo = fpts[m], c = fpts[m + 1], hi = fpts[m + 2];
    let first = -1;
    const w = [];
    for (let i = 0; i < nF; i++) {
      const down = (all[i] - lo) / (c - lo), up = (hi - all[i]) / (hi - c);
      const v = Math.max(0, Math.min(down, up));
      fb[i * nM + m] = v;
      if (v > 0) { if (first < 0) first = i; w.push(v); }
    }
    bands.push({ first: Math.max(0, first), w: Float32Array.from(w) });
  }
  FB = { fb, bands };
  return FB;
}

/** Log-mel spectrogram of mono 22.05 kHz audio. Returns {spect: Float32Array(T*128), T}. */
export function logMel(x) {
  const { nfft, hop, nMels } = BT;
  const n = x.length;
  const pad = nfft / 2;
  const T = Math.floor(n / hop) + 1; // torch.stft(center=True)
  const win = hannPeriodic(nfft);
  const rfft = RealFFT.get(nfft);
  const bins = nfft / 2 + 1;
  const buf = new Float64Array(nfft), re = new Float64Array(bins), im = new Float64Array(bins), mag = new Float64Array(bins);
  const { bands } = melFilters();
  const spect = new Float32Array(T * nMels);
  const scale = 1 / Math.sqrt(nfft);
  for (let t = 0; t < T; t++) {
    const start = t * hop - pad;
    for (let k = 0; k < nfft; k++) {
      let j = start + k;
      if (j < 0) j = -j; // reflect
      if (j >= n) j = 2 * n - 2 - j;
      buf[k] = (j >= 0 && j < n ? x[j] : 0) * win[k];
    }
    rfft.forward(buf, re, im);
    for (let b = 0; b < bins; b++) mag[b] = Math.hypot(re[b], im[b]) * scale;
    const o = t * nMels;
    for (let m = 0; m < nMels; m++) {
      const { first, w } = bands[m];
      let s = 0;
      for (let k = 0; k < w.length; k++) s += w[k] * mag[first + k];
      spect[o + m] = Math.log1p(1000 * s);
    }
  }
  return { spect, T };
}

/**
 * Frame-wise beat and downbeat logits for mono 44.1 kHz (or 22.05 kHz) audio.
 * Long pieces are cut in 1500-frame (30 s) windows that overlap by the 6-frame borders the
 * model was not trained on, like beat_this.inference.split_predict_aggregate ('keep_first').
 */
export async function beatActivations(ort, session, mono, sr, { onProgress, shouldCancel } = {}) {
  // a steep low-pass like soxr's (which the model was trained with): flat to ~10.2 kHz
  const x = sr === BT.sr ? mono : resample(mono, sr, BT.sr, 64, 0.958);
  const { spect, T } = logMel(x);
  const { frames: C, border: B, nMels } = BT;
  let starts = [];
  for (let s = -B; s < T - B; s += C - 2 * B) starts.push(s);
  if (T > C - 2 * B) starts[starts.length - 1] = T - (C - B);
  if (!starts.length) starts = [-B];
  const beat = new Float32Array(T).fill(-1000), down = new Float32Array(T).fill(-1000);
  const inp = new Float32Array(C * nMels);
  const inName = session.inputNames ? session.inputNames[0] : 'spect';
  // keep_first: later windows are written first, so earlier ones win where they overlap
  for (let k = starts.length - 1; k >= 0; k--) {
    if (shouldCancel && shouldCancel()) throw new Error('cancelled');
    const s = starts[k];
    inp.fill(0);
    for (let t = 0; t < C; t++) {
      const src = s + t;
      if (src < 0 || src >= T) continue;
      inp.set(spect.subarray(src * nMels, (src + 1) * nMels), t * nMels);
    }
    const res = await session.run({ [inName]: new ort.Tensor('float32', inp, [1, C, nMels]) });
    const b = res.beat.data, d = res.downbeat.data;
    for (let t = B; t < C - B; t++) {
      const dst = s + t;
      if (dst < 0 || dst >= T) continue;
      beat[dst] = b[t];
      down[dst] = d[t];
    }
    if (onProgress) onProgress((starts.length - k) / starts.length);
  }
  return { beat, down, fps: BT.fps, T };
}

/** beat_this "minimal" post-processing: maxima within +-70 ms with logit > 0, adjacent ones merged. */
export function pickPeaks(logits, fps = BT.fps) {
  const T = logits.length, frames = [];
  for (let i = 0; i < T; i++) {
    const v = logits[i];
    if (!(v > 0)) continue;
    let isMax = true;
    for (let k = Math.max(0, i - 3); k <= Math.min(T - 1, i + 3); k++) if (logits[k] > v) { isMax = false; break; }
    if (isMax) frames.push(i);
  }
  // deduplicate_peaks(width=1): runs of neighbouring frames become their mean
  const out = [];
  let p = null, c = 0;
  for (const f of frames) {
    if (p !== null && f - p <= 1) { c++; p += (f - p) / c; } else { if (p !== null) out.push(p); p = f; c = 1; }
  }
  if (p !== null) out.push(p);
  return out.map((f) => f / fps);
}

const sigmoid = (v) => 1 / (1 + Math.exp(-v));

// how readily each bar length is used (log scale): 4/4 most common, then 3/4; odd and
// compound lengths need clear evidence (midwest emo and math rock do use them)
const BAR_PRIOR = { 2: -2.0, 3: -0.7, 4: 0, 5: -2.2, 6: -2.0, 7: -2.6 };

/**
 * Bar lines by Viterbi over (bar length, position in bar) at each beat, scored by the model's
 * downbeat probability. The meter can change, but only where the evidence clearly outweighs
 * the cost of switching. Returns the beat indices where bars start.
 */
export function barStarts(strength, opts = {}) {
  const lens = opts.meter ? [opts.meter] : (opts.lengths || [3, 4, 5, 6, 7]);
  const switchCost = opts.switchCost ?? 8;
  const N = strength.length;
  const states = [];
  for (const L of lens) for (let k = 0; k < L; k++) states.push({ L, k });
  const S = states.length;
  const idx = new Map(states.map((st, i) => [st.L * 16 + st.k, i]));
  const em = (i, k) => { const p = Math.min(0.98, Math.max(0.02, strength[i])); return k === 0 ? Math.log(p) : Math.log(1 - p); };
  let score = new Float64Array(S), next = new Float64Array(S);
  const back = new Int32Array(N * S);
  for (let s2 = 0; s2 < S; s2++) score[s2] = (BAR_PRIOR[states[s2].L] ?? -3) + em(0, states[s2].k);
  for (let i = 1; i < N; i++) {
    // best way to end a bar of each length (k = L-1) at beat i-1
    let bestEnd = -Infinity, bestEndS = -1;
    for (const L of lens) { const s2 = idx.get(L * 16 + L - 1); if (score[s2] > bestEnd) { bestEnd = score[s2]; bestEndS = s2; } }
    for (let s2 = 0; s2 < S; s2++) {
      const { L, k } = states[s2];
      let v, from;
      if (k > 0) { from = idx.get(L * 16 + k - 1); v = score[from]; } else {
        // a new bar: same length as the one that just ended, or switch (costly)
        const same = idx.get(L * 16 + L - 1);
        v = score[same]; from = same;
        const sw = bestEnd - switchCost + (BAR_PRIOR[L] ?? -3);
        if (bestEndS !== same && sw > v) { v = sw; from = bestEndS; }
      }
      next[s2] = v + em(i, k);
      back[i * S + s2] = from;
    }
    [score, next] = [next, score];
  }
  let s2 = 0;
  for (let j = 1; j < S; j++) if (score[j] > score[s2]) s2 = j;
  const starts = [];
  for (let i = N - 1; i >= 0; i--) {
    if (states[s2].k === 0) starts.push(i);
    if (i > 0) s2 = back[i * S + s2];
  }
  return starts.reverse();
}

/**
 * Beats, bar lines and meter from the activations.
 * Returns {beats, downbeats, bars: [{beat, len}], meter (most common bar length), downbeatPhase, bpm}.
 * opts.variable === false keeps one meter for the whole song.
 */
export function beatsAndBars(act, opts = {}) {
  const { fps } = act;
  const beats = pickPeaks(act.beat, fps);
  if (beats.length < 4) return { beats, downbeats: [], bars: [], meter: 4, downbeatPhase: 0, bpm: 120 };
  const strength = beats.map((t) => {
    const i = Math.round(t * fps);
    let m = -1000;
    for (let k = Math.max(0, i - 2); k <= Math.min(act.T - 1, i + 2); k++) m = Math.max(m, act.down[k]);
    return sigmoid(m);
  });
  const lengths = opts.variable === false ? null : undefined;
  let starts;
  if (opts.variable === false || opts.meter) {
    // one meter: the length (3 or 4, or the given one) and phase that fit best
    let best = null;
    for (const L of (opts.meter ? [opts.meter] : [3, 4])) {
      const st = barStarts(strength, { lengths: [L], meter: L });
      let sc = BAR_PRIOR[L] ?? 0;
      for (const i of st) sc += Math.log(Math.max(0.02, strength[i]));
      if (!best || sc > best.sc) best = { sc, st };
    }
    starts = best.st;
  } else {
    starts = barStarts(strength, { lengths });
  }
  const bars = starts.map((b, k) => ({ beat: b, len: (k + 1 < starts.length ? starts[k + 1] : beats.length) - b }));
  const counts = {};
  for (const b of bars.slice(0, -1)) counts[b.len] = (counts[b.len] || 0) + b.len;
  let meter = 4, mc = 0;
  for (const [len, c] of Object.entries(counts)) if (c > mc) { mc = c; meter = Number(len); }
  if (bars.length) bars[bars.length - 1].len = Math.max(bars[bars.length - 1].len, 1);
  const ibis = [];
  for (let i = 1; i < beats.length; i++) ibis.push(beats[i] - beats[i - 1]);
  ibis.sort((a, b) => a - b);
  return {
    beats,
    downbeats: starts.map((i) => beats[i]),
    bars,
    meter,
    downbeatPhase: starts.length ? starts[0] % meter : 0,
    bpm: 60 / ibis[ibis.length >> 1],
  };
}
