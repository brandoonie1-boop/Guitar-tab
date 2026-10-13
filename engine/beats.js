// Tempo and beat tracking (onset strength + dynamic programming, after Ellis 2007 / librosa),
// plus a simple meter / downbeat estimate.
import { stftMag, melFilterbank, applyFilterbank } from './spectral.js';

/** Onset strength envelope (librosa-style log-mel spectral flux). */
export function onsetEnvelope(x, sr, opts = {}) {
  const nfft = opts.nfft || 2048, hop = opts.hop || 512;
  const spec = stftMag(x, sr, nfft, hop);
  const bands = melFilterbank(sr, nfft, opts.nMels || 96, opts.fmin || 30, opts.fmax || 11000);
  const mel = applyFilterbank(spec, bands);
  const nB = bands.length, T = spec.frames;
  // power -> dB with 80 dB floor below the max
  let mx = 1e-10;
  for (let i = 0; i < mel.length; i++) if (mel[i] > mx) mx = mel[i];
  const floorDb = 10 * Math.log10(mx) - 80;
  const db = new Float32Array(mel.length);
  for (let i = 0; i < mel.length; i++) db[i] = Math.max(floorDb, 10 * Math.log10(Math.max(1e-10, mel[i])));
  const env = new Float32Array(T);
  const lowEnv = new Float32Array(T);
  const lowBands = bands.filter((b) => b.center < 180).length;
  for (let t = 1; t < T; t++) {
    let s = 0, sl = 0;
    for (let m = 0; m < nB; m++) {
      const d = db[t * nB + m] - db[(t - 1) * nB + m];
      if (d > 0) { s += d; if (m < lowBands) sl += d; }
    }
    env[t] = s / nB;
    lowEnv[t] = sl / Math.max(1, lowBands);
  }
  return { env, lowEnv, fps: sr / hop, hop, sr };
}

function autocorr(x, maxLag) {
  const n = x.length;
  const out = new Float64Array(maxLag + 1);
  for (let lag = 0; lag <= maxLag; lag++) {
    let s = 0;
    for (let i = lag; i < n; i++) s += x[i] * x[i - lag];
    out[lag] = s;
  }
  return out;
}

/** Global tempo estimate with a log-normal prior around 110 BPM. Returns {bpm, candidates}. */
export function estimateTempo(env, fps, opts = {}) {
  const startBpm = opts.startBpm || 110, stdOct = opts.stdOct || 1.0;
  const win = Math.round((opts.acSize || 8) * fps);
  const maxLag = win;
  const acc = new Float64Array(maxLag + 1);
  // average local autocorrelation over hops of half a window
  for (let s = 0; s + win <= env.length || s === 0; s += Math.max(1, win >> 1)) {
    const seg = env.subarray(s, Math.min(env.length, s + win));
    if (seg.length < fps * 2) break;
    const ac = autocorr(seg, Math.min(maxLag, seg.length - 1));
    const n0 = ac[0] || 1;
    for (let l = 0; l < ac.length; l++) acc[l] += ac[l] / n0;
    if (s + win >= env.length) break;
  }
  let best = -Infinity, bestLag = 0;
  const scores = [];
  for (let lag = 1; lag <= maxLag; lag++) {
    const bpm = (60 * fps) / lag;
    if (bpm > 260 || bpm < 40) continue;
    const prior = -0.5 * Math.pow(Math.log2(bpm / startBpm) / stdOct, 2);
    const sc = Math.log1p(1e6 * Math.max(0, acc[lag])) + prior;
    scores.push({ bpm, sc });
    if (sc > best) { best = sc; bestLag = lag; }
  }
  // parabolic refinement of the lag
  let lag = bestLag;
  if (bestLag > 1 && bestLag < maxLag) {
    const a = acc[bestLag - 1], b = acc[bestLag], c = acc[bestLag + 1];
    const den = a - 2 * b + c;
    if (den < 0) lag = bestLag + (0.5 * (a - c)) / den;
  }
  scores.sort((x, y) => y.sc - x.sc);
  return { bpm: (60 * fps) / lag, candidates: scores.slice(0, 5).map((s) => s.bpm) };
}

/** Dynamic programming beat tracker. Returns beat times in seconds. */
export function trackBeats(env, fps, bpm, opts = {}) {
  const tightness = opts.tightness ?? 100;
  const n = env.length;
  if (n < 4) return [];
  // normalise
  let sd = 0, mean = 0;
  for (let i = 0; i < n; i++) mean += env[i];
  mean /= n;
  for (let i = 0; i < n; i++) sd += (env[i] - mean) ** 2;
  sd = Math.sqrt(sd / n) || 1;
  const period = (60 * fps) / bpm;
  // local score: smooth with a gaussian of width period/32
  const half = Math.round(period);
  const kern = [];
  for (let k = -half; k <= half; k++) kern.push(Math.exp(-0.5 * Math.pow((k * 32) / period, 2)));
  const local = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = -half; k <= half; k++) { const j = i + k; if (j >= 0 && j < n) s += (env[j] / sd) * kern[k + half]; }
    local[i] = s;
  }
  const cum = new Float64Array(n);
  const back = new Int32Array(n).fill(-1);
  const wMin = Math.round(-2 * period), wMax = -Math.max(1, Math.round(period / 2));
  let firstBeat = true;
  let runMax = 0;
  for (let i = 0; i < n; i++) {
    if (local[i] > runMax) runMax = local[i];
    let best = -Infinity, bi = -1;
    for (let w = wMin; w <= wMax; w++) {
      const j = i + w;
      if (j < 0) continue;
      const txwt = -tightness * Math.pow(Math.log(-w / period), 2);
      const v = cum[j] + txwt;
      if (v > best) { best = v; bi = j; }
    }
    cum[i] = local[i] + (bi >= 0 ? best : 0);
    // librosa: until we find the first local max, don't link back
    if (firstBeat && local[i] < 0.01 * runMax) back[i] = -1;
    else { back[i] = bi; firstBeat = false; }
  }
  // last beat: last local max of cumscore above half the median of local maxima
  const maxima = [];
  for (let i = 1; i < n - 1; i++) if (cum[i] > cum[i - 1] && cum[i] >= cum[i + 1]) maxima.push(i);
  if (!maxima.length) return [];
  const vals = maxima.map((i) => cum[i]).sort((a, b) => a - b);
  const med = vals[vals.length >> 1];
  let last = maxima[maxima.length - 1];
  for (let k = maxima.length - 1; k >= 0; k--) if (cum[maxima[k]] >= 0.5 * med) { last = maxima[k]; break; }
  const beats = [];
  for (let i = last; i >= 0; i = back[i]) { beats.push(i); if (back[i] < 0) break; }
  beats.reverse();
  // trim weak leading/trailing beats
  let rmsLocal = 0;
  for (const b of beats) rmsLocal += local[b] * local[b];
  rmsLocal = Math.sqrt(rmsLocal / beats.length);
  let a = 0, z = beats.length - 1;
  while (a < z && local[beats[a]] < 0.3 * rmsLocal) a++;
  while (z > a && local[beats[z]] < 0.3 * rmsLocal) z--;
  return beats.slice(a, z + 1).map((f) => f / fps);
}

/**
 * Meter + downbeat phase. accentAt(t) should return accent strength near time t (e.g. low
 * frequency onsets or chord changes). Returns {meter, phase} with beats[phase] the first downbeat.
 */
export function estimateDownbeats(beats, accentFns, meters = [4, 3]) {
  let best = null;
  for (const m of meters) {
    for (let ph = 0; ph < m; ph++) {
      let on = 0, nOn = 0, off = 0, nOff = 0;
      beats.forEach((t, i) => {
        let a = 0;
        for (const f of accentFns) a += f(t);
        if ((i - ph) % m === 0) { on += a; nOn++; } else { off += a; nOff++; }
      });
      const contrast = nOn && nOff ? on / nOn - off / nOff : 0;
      const sc = contrast * (m === 4 ? 1.12 : 1); // mild preference for 4/4
      if (!best || sc > best.sc) best = { meter: m, phase: ph, sc };
    }
  }
  return best || { meter: 4, phase: 0, sc: 0 };
}

/** Helper that samples an envelope (max in a small window) at time t. */
export function envSampler(env, fps, radius = 0.05) {
  const r = Math.max(1, Math.round(radius * fps));
  let mx = 0;
  for (let i = 0; i < env.length; i++) if (env[i] > mx) mx = env[i];
  mx = mx || 1;
  return (t) => {
    const c = Math.round(t * fps);
    let v = 0;
    for (let k = c - r; k <= c + r; k++) if (k >= 0 && k < env.length && env[k] > v) v = env[k];
    return v / mx;
  };
}

function strengthAt(env, fps, times) {
  let s = 0;
  for (const t of times) {
    const c = Math.round(t * fps);
    let m = 0;
    for (let k = c - 2; k <= c + 2; k++) if (k >= 0 && k < env.length && env[k] > m) m = env[k];
    s += m;
  }
  return times.length ? s / times.length : 0;
}

/** Full analysis: tempo, beats, meter, downbeats. */
export function analyzeRhythm(mono, sr, opts = {}) {
  const oe = onsetEnvelope(mono, sr);
  // tempo prior tuned on GuitarSet (tests/eval_tempo.mjs): 130 BPM, 0.6 octave spread
  const tempo = estimateTempo(oe.env, oe.fps, { startBpm: opts.startBpm || 130, stdOct: opts.stdOct || 0.6 });
  let bpm = opts.bpm || tempo.bpm;
  let beats = trackBeats(oe.env, oe.fps, bpm);
  // half-tempo check: if the off-beats are nearly as strong as the beats, the real pulse is twice as fast
  if (!opts.bpm && beats.length > 8) {
    const mids = [];
    for (let i = 0; i + 1 < beats.length; i++) mids.push((beats[i] + beats[i + 1]) / 2);
    const sOn = strengthAt(oe.env, oe.fps, beats), sMid = strengthAt(oe.env, oe.fps, mids);
    const thr = opts.halfRatio ?? 99; // disabled: on GuitarSet it caused more octave errors than it fixed
    if (sOn > 0 && sMid / sOn > thr && bpm * 2 <= 230) {
      bpm *= 2;
      beats = trackBeats(oe.env, oe.fps, bpm);
    }
  }
  // phase check: beats should sit on the low-frequency hits (kick, bass), not on the off-beat hats
  if (beats.length > 8 && opts.phaseCheck !== false) {
    const half = [];
    for (let i = 0; i + 1 < beats.length; i++) half.push((beats[i] + beats[i + 1]) / 2);
    const lowOn = strengthAt(oe.lowEnv, oe.fps, beats), lowOff = strengthAt(oe.lowEnv, oe.fps, half);
    if (lowOff > (opts.phaseRatio ?? 1.35) * lowOn) beats = half;
  }
  const accents = [envSampler(oe.lowEnv, oe.fps)];
  if (opts.extraAccent) accents.push(opts.extraAccent);
  const db = opts.meter ? { meter: opts.meter, phase: 0 } : estimateDownbeats(beats, accents);
  if (opts.meter) {
    const r = estimateDownbeats(beats, accents, [opts.meter]);
    db.phase = r.phase;
  }
  let bpmBeats = bpm;
  if (beats.length > 4) {
    const ibis = [];
    for (let i = 1; i < beats.length; i++) ibis.push(beats[i] - beats[i - 1]);
    ibis.sort((a, b) => a - b);
    bpmBeats = 60 / ibis[ibis.length >> 1];
  }
  return { bpm: bpmBeats, bpmCandidates: tempo.candidates, beats, meter: db.meter, downbeatPhase: db.phase, envelope: oe };
}
