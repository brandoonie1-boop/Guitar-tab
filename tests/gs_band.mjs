function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

export function addBand(x, beats, chordsRoot) {
  // drums (kick 1&3, snare 2&4, hats 8ths) + a simple bass on the beat
  const sr = 44100, r = rng(5);
  const out = Float32Array.from(x);
  let gpk = 0; for (let i = 0; i < out.length; i++) gpk = Math.max(gpk, Math.abs(out[i])); gpk = gpk || 1;
  beats.forEach(([t, pos], i) => {
    const s = Math.floor(t * sr);
    const isKick = pos % 2 === 1, isSnare = pos % 2 === 0;
    const L = Math.floor(0.25 * sr);
    for (let k = 0; k < L && s + k < out.length; k++) {
      const env = Math.exp(-k / (0.05 * sr));
      if (isKick) out[s + k] += 0.5 * gpk * Math.sin(2 * Math.PI * 55 * (k / sr) * (1 + 1.5 * env)) * env;
      if (isSnare) out[s + k] += 0.25 * gpk * (r() * 2 - 1) * Math.exp(-k / (0.07 * sr));
    }
    // hat on the off-beat
    const next = beats[i + 1] ? beats[i + 1][0] : t + 0.5;
    const hs = Math.floor(((t + next) / 2) * sr);
    for (let k = 0; k < 0.04 * sr && hs + k < out.length; k++) out[hs + k] += 0.08 * gpk * (r() * 2 - 1) * Math.exp(-k / (0.01 * sr));
    // bass: root-ish low note (E1/A1 alternating) for the beat duration
    const f = pos % 4 === 1 ? 41.2 : 55;
    const bl = Math.floor((next - t) * sr * 0.9);
    for (let k = 0; k < bl && s + k < out.length; k++) out[s + k] += 0.18 * gpk * Math.sin(2 * Math.PI * f * k / sr) * Math.min(1, k / 200) * Math.exp(-k / (0.6 * sr));
  });
  return out;
}

