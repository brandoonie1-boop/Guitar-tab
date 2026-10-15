// Note clean-up between raw detection and fingering.

export const GUITAR_MIN = 33; // A1 (covers down-tuned low strings)
export const GUITAR_MAX = 91; // G6

export function cleanNotes(notes, opts = {}) {
  const minP = opts.minPitch ?? GUITAR_MIN, maxP = opts.maxPitch ?? GUITAR_MAX;
  let ns = notes.filter((n) => n.pitch >= minP && n.pitch <= maxP && n.t1 > n.t0);
  ns.sort((a, b) => a.t0 - b.t0 || a.pitch - b.pitch);
  // 1. merge fragments of one note (same pitch, tiny gap, weak re-attack)
  const out = [];
  const lastByPitch = new Map();
  for (const n of ns) {
    const prev = opts.mergeFragments === false ? null : lastByPitch.get(n.pitch);
    if (prev && n.t0 - prev.t1 < 0.035 && (n.onsetProb ?? 1) < 0.25) {
      prev.t1 = Math.max(prev.t1, n.t1);
      if (n.bends && prev.bends) prev.bends = prev.bends.concat(n.bends);
      continue;
    }
    const c = { ...n };
    out.push(c);
    lastByPitch.set(n.pitch, c);
  }
  // 1b. a ringing note re-triggered by another note's attack (common with octaves and open
  //     strings): when its "new" onset is clearly weaker than the attack that caused it, keep it
  //     as one sustained note
  if (opts.mergeRetrigger !== false) {
    out.sort((a, b) => a.t0 - b.t0);
    const ratio = opts.retriggerRatio ?? 0.8, absMax = opts.retriggerAbs ?? 0.8;
    const lastEnd = new Map();
    const merged = [];
    for (let i = 0; i < out.length; i++) {
      const n = out[i];
      const prev = lastEnd.get(n.pitch);
      if (prev && Math.abs(n.t0 - prev.t1) < 0.03 && (n.onsetProb ?? 1) < absMax) {
        let other = 0;
        for (let j = Math.max(0, i - 8); j < Math.min(out.length, i + 9); j++) {
          if (j === i || Math.abs(out[j].t0 - n.t0) > 0.03 || out[j].pitch === n.pitch) continue;
          other = Math.max(other, out[j].onsetProb ?? 0);
        }
        if (other > 0 && (n.onsetProb ?? 1) < ratio * other) {
          prev.t1 = Math.max(prev.t1, n.t1);
          if (n.bends && prev.bends) prev.bends = prev.bends.concat(n.bends);
          continue;
        }
      }
      merged.push(n);
      lastEnd.set(n.pitch, n);
    }
    out.length = 0;
    out.push(...merged);
  }
  // 2. drop faint octave / twelfth ghosts that start with a much louder lower note
  const keep = new Array(out.length).fill(true);
  for (let i = 0; i < out.length; i++) {
    const n = out[i];
    for (let j = 0; j < out.length; j++) {
      if (i === j) continue;
      const m = out[j];
      if (Math.abs(m.t0 - n.t0) > 0.04) { if (m.t0 > n.t0 + 0.04) break; else continue; }
      const iv = n.pitch - m.pitch;
      if ((iv === 12 || iv === 19 || iv === 24) && n.amp < 0.33 * m.amp) {
        const ov = Math.min(n.t1, m.t1) - Math.max(n.t0, m.t0);
        if (ov > 0.7 * (n.t1 - n.t0)) { keep[i] = false; break; }
      }
    }
  }
  ns = out.filter((_, i) => keep[i]);
  // 3. very short, quiet blips
  const minLen = opts.minLen ?? 0.045;
  ns = ns.filter((n) => n.t1 - n.t0 >= minLen || n.amp > 0.4);
  // 4. a guitar has six strings: if more than six notes ring, end the oldest ones
  ns.sort((a, b) => a.t0 - b.t0);
  const active = [];
  for (const n of ns) {
    for (let k = active.length - 1; k >= 0; k--) if (active[k].t1 <= n.t0) active.splice(k, 1);
    while (active.length >= 6) {
      active.sort((a, b) => a.t0 - b.t0);
      const old = active.shift();
      old.t1 = Math.max(old.t0 + 0.03, n.t0);
    }
    active.push(n);
  }
  ns.forEach((n, i) => { n.id = opts.idPrefix ? `${opts.idPrefix}${i}` : i; });
  return ns;
}
