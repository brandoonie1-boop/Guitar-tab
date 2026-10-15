// Generic DSP helpers: resampling, mixing, simple statistics.

function bessel0(x) {
  let sum = 1, term = 1;
  const q = (x * x) / 4;
  for (let k = 1; k < 64; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < 1e-12 * sum) break;
  }
  return sum;
}

const kernelCache = new Map();

function sincKernel(halfZeros, oversample, beta) {
  const key = halfZeros + ':' + oversample + ':' + beta;
  let k = kernelCache.get(key);
  if (k) return k;
  const len = halfZeros * oversample + 2;
  k = new Float64Array(len);
  const i0b = bessel0(beta);
  for (let i = 0; i < len; i++) {
    const x = i / oversample; // in zero-crossings
    const s = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    const r = x / halfZeros;
    const w = r <= 1 ? bessel0(beta * Math.sqrt(1 - r * r)) / i0b : 0;
    k[i] = s * w;
  }
  kernelCache.set(key, k);
  return k;
}

// Band-limited resampling with a Kaiser-windowed sinc. Returns Float32Array.
// cutoffFrac: where the low-pass sits, as a fraction of the lower Nyquist frequency.
export function resample(input, srIn, srOut, halfZeros = 24, cutoffFrac = 0.97) {
  if (srIn === srOut) return Float32Array.from(input);
  const ratio = srOut / srIn;
  const cutoff = Math.min(1, ratio) * cutoffFrac; // fraction of input Nyquist
  const oversample = 256;
  const kernel = sincKernel(halfZeros, oversample, 8.6);
  const outLen = Math.max(1, Math.round(input.length * ratio));
  const out = new Float32Array(outLen);
  const width = halfZeros / cutoff; // in input samples
  const n = input.length;
  const step = cutoff * oversample; // kernel index per input sample
  for (let i = 0; i < outLen; i++) {
    const p = i / ratio;
    const j0 = Math.max(0, Math.ceil(p - width));
    const j1 = Math.min(n - 1, Math.floor(p + width));
    let acc = 0;
    for (let j = j0; j <= j1; j++) {
      const d = Math.abs(p - j) * step;
      const di = d | 0;
      const frac = d - di;
      if (di + 1 >= kernel.length) continue;
      const kv = kernel[di] + (kernel[di + 1] - kernel[di]) * frac;
      acc += input[j] * kv;
    }
    out[i] = acc * cutoff;
  }
  return out;
}

export function toMono(channels) {
  if (channels.length === 1) return Float32Array.from(channels[0]);
  const n = channels[0].length;
  const out = new Float32Array(n);
  const c = channels.length;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < c; k++) s += channels[k][i];
    out[i] = s / c;
  }
  return out;
}

export function rms(x, start = 0, end = x.length) {
  let s = 0;
  for (let i = start; i < end; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, end - start));
}

export function peak(x) {
  let m = 0;
  for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > m) m = a; }
  return m;
}

export function median(arr) {
  if (!arr.length) return 0;
  const a = Array.from(arr).sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

export function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return arr.length ? s / arr.length : 0;
}

export function midiToHz(m) { return 440 * Math.pow(2, (m - 69) / 12); }
export function hzToMidi(f) { return 69 + 12 * Math.log2(f / 440); }

export const NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
export function midiName(m, withOctave = true) {
  const n = NOTE_NAMES_SHARP[((m % 12) + 12) % 12];
  return withOctave ? n + (Math.floor(m / 12) - 1) : n;
}
