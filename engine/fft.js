// Radix-2 FFT and real-FFT helpers. Pure JS, no dependencies.
// Works in browsers, web workers and Node.

const cache = new Map();

export class FFT {
  constructor(n) {
    if (n & (n - 1)) throw new Error('FFT size must be a power of two: ' + n);
    this.n = n;
    const rev = new Uint32Array(n);
    let bits = 0;
    while ((1 << bits) < n) bits++;
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      rev[i] = r;
    }
    this.rev = rev;
    const half = n >> 1;
    this.cos = new Float64Array(half);
    this.sin = new Float64Array(half);
    for (let i = 0; i < half; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = -Math.sin((2 * Math.PI * i) / n);
    }
  }

  static get(n) {
    let f = cache.get(n);
    if (!f) { f = new FFT(n); cache.set(n, f); }
    return f;
  }

  // In-place complex FFT. inverse=true computes the unscaled inverse.
  transform(re, im, inverse = false) {
    const n = this.n, rev = this.rev, cs = this.cos, sn = this.sin;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sgn = inverse ? -1 : 1;
    for (let size = 2; size <= n; size <<= 1) {
      const halfSize = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        let k = 0;
        for (let j = start; j < start + halfSize; j++) {
          const wr = cs[k], wi = sgn * sn[k];
          const l = j + halfSize;
          const tr = re[l] * wr - im[l] * wi;
          const ti = re[l] * wi + im[l] * wr;
          re[l] = re[j] - tr; im[l] = im[j] - ti;
          re[j] += tr; im[j] += ti;
          k += step;
        }
      }
    }
  }
}

// Real FFT of size n using a complex FFT of size n/2.
export class RealFFT {
  constructor(n) {
    this.n = n;
    this.m = n >> 1;
    this.fft = FFT.get(this.m);
    this.zr = new Float64Array(this.m);
    this.zi = new Float64Array(this.m);
    this.wr = new Float64Array(this.m + 1);
    this.wi = new Float64Array(this.m + 1);
    for (let k = 0; k <= this.m; k++) {
      this.wr[k] = Math.cos((2 * Math.PI * k) / n);
      this.wi[k] = -Math.sin((2 * Math.PI * k) / n);
    }
  }

  static get(n) {
    const key = 'r' + n;
    let f = cache.get(key);
    if (!f) { f = new RealFFT(n); cache.set(key, f); }
    return f;
  }

  // x: array-like length n (real). outRe/outIm length n/2+1.
  forward(x, outRe, outIm, xOffset = 0) {
    const m = this.m, zr = this.zr, zi = this.zi;
    for (let k = 0; k < m; k++) {
      zr[k] = x[xOffset + 2 * k];
      zi[k] = x[xOffset + 2 * k + 1];
    }
    this.fft.transform(zr, zi, false);
    const wr = this.wr, wi = this.wi;
    for (let k = 0; k <= m; k++) {
      const a = k === m ? 0 : k;
      const b = k === 0 ? 0 : m - k;
      const ar = zr[a], ai = zi[a];
      const br = zr[b], bi = -zi[b]; // conj(Z[m-k])
      const er = 0.5 * (ar + br), ei = 0.5 * (ai + bi);
      // O = (Z[k] - conj(Z[m-k])) / (2i)
      const dr = ar - br, di = ai - bi;
      const or = 0.5 * di, oi = -0.5 * dr;
      outRe[k] = er + wr[k] * or - wi[k] * oi;
      outIm[k] = ei + wr[k] * oi + wi[k] * or;
    }
  }

  // inverse with 1/n scaling. inRe/inIm length n/2+1, out length n.
  inverse(inRe, inIm, out, outOffset = 0) {
    const m = this.m, zr = this.zr, zi = this.zi, wr = this.wr, wi = this.wi;
    for (let k = 0; k < m; k++) {
      const xr = inRe[k], xi = inIm[k];
      const yr = inRe[m - k], yi = -inIm[m - k]; // conj(X[m-k])
      const er = 0.5 * (xr + yr), ei = 0.5 * (xi + yi);
      // O = (X[k] - conj(X[m-k])) / (2 W^k); 1/W^k = conj(W^k)
      const dr = 0.5 * (xr - yr), di = 0.5 * (xi - yi);
      const cr = wr[k], ci = -wi[k];
      const or = dr * cr - di * ci;
      const oi = dr * ci + di * cr;
      // Z = E + i O
      zr[k] = er - oi;
      zi[k] = ei + or;
    }
    this.fft.transform(zr, zi, true);
    const s = 1 / m;
    for (let k = 0; k < m; k++) {
      out[outOffset + 2 * k] = zr[k] * s;
      out[outOffset + 2 * k + 1] = zi[k] * s;
    }
  }
}

export function hannPeriodic(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

export function hannSymmetric(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}
