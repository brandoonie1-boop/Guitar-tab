// Rhythm quantization: places note attacks on a beat grid and splits the song into measures.

export const TICKS_PER_QUARTER = 960;

/** Extend tracked beats to cover [0, duration] using the local beat period. */
export function extendBeats(beats, duration, bpm) {
  const period = 60 / bpm;
  let b = beats.length >= 2 ? beats.slice() : [0];
  if (b.length < 2) { b = []; for (let t = 0; t <= duration + period; t += period) b.push(t); return b; }
  const p0 = Math.min(period * 1.5, Math.max(period * 0.5, b[1] - b[0]));
  while (b[0] > 0) b.unshift(b[0] - p0);
  const n = b.length;
  const p1 = Math.min(period * 1.5, Math.max(period * 0.5, b[n - 1] - b[n - 2]));
  while (b[b.length - 1] < duration + p1) b.push(b[b.length - 1] + p1);
  return b;
}

/** fractional beat index of time t (beats must be increasing) */
export function beatPos(beats, t) {
  let lo = 0, hi = beats.length - 1;
  if (t <= beats[0]) return (t - beats[0]) / Math.max(1e-6, beats[1] - beats[0]);
  if (t >= beats[hi]) return hi + (t - beats[hi]) / Math.max(1e-6, beats[hi] - beats[hi - 1]);
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (beats[m] <= t) lo = m; else hi = m; }
  return lo + (t - beats[lo]) / (beats[lo + 1] - beats[lo]);
}

export function timeOfPos(beats, p) {
  const i = Math.max(0, Math.min(beats.length - 2, Math.floor(p)));
  return beats[i] + (p - i) * (beats[i + 1] - beats[i]);
}

/**
 * Quantize a part.
 * @param notes [{id, t0, t1, string, fret, pitch, tech}] (only placed notes)
 * @param rhythm {beats (extended), meter, downbeatBeat (index of first downbeat in beats)}
 * @returns {measures: [{start (beat index), beats:[{grid, segs:[{s, len, notes|null}]}]}], measureTimes}
 */
export function quantize(notes, rhythm, opts = {}) {
  const { beats, meter } = rhythm;
  const grids = opts.grids || [4, 3, 8];
  const chordTol = opts.chordTol ?? 0.04;
  // attack events
  const sorted = notes.slice().sort((a, b) => a.t0 - b.t0);
  const events = [];
  for (const n of sorted) {
    const last = events[events.length - 1];
    if (last && n.t0 - last.t0 <= chordTol && !last.notes.some((m) => m.string === n.string)) last.notes.push(n);
    else events.push({ t0: n.t0, notes: [n] });
  }
  for (const e of events) {
    e.p = beatPos(beats, e.t0);
    e.release = Math.max(...e.notes.map((n) => n.t1));
    e.pr = beatPos(beats, e.release);
  }
  // choose a grid for each beat
  const nBeats = beats.length - 1;
  const beatGrid = new Array(nBeats).fill(4);
  const byBeat = new Map();
  for (const e of events) {
    const b = Math.floor(e.p + 1 / 16);
    if (!byBeat.has(b)) byBeat.set(b, []);
    byBeat.get(b).push(e);
  }
  for (const [b, evs] of byBeat) {
    if (b < 0 || b >= nBeats) continue;
    let best = 4, bestErr = Infinity;
    for (const g of grids) {
      let err = 0;
      for (const e of evs) { const x = (e.p - b) * g; err += Math.abs(x - Math.round(x)) / g; }
      const pen = g === 3 ? 0.035 * evs.length : g === 8 ? 0.02 * evs.length : 0;
      if (err + pen < bestErr - 1e-9) { bestErr = err + pen; best = g; }
    }
    // a triplet beat needs at least two attacks that really sit on triplet positions
    if (best === 3 && evs.length < 2) best = 4;
    beatGrid[b] = best;
  }
  // snap events to slots: slot = (beat, k)
  const snapped = new Map(); // key beat*100+k -> event
  for (const e of events) {
    let b = Math.floor(e.p);
    let g = beatGrid[Math.max(0, Math.min(nBeats - 1, b))] || 4;
    let k = Math.round((e.p - b) * g);
    if (k >= g) { b += 1; k = 0; g = beatGrid[Math.max(0, Math.min(nBeats - 1, b))] || 4; }
    if (b < 0 || b >= nBeats) continue;
    const key = b * 100 + k;
    const prev = snapped.get(key);
    if (prev) {
      for (const n of e.notes) if (!prev.notes.some((m) => m.string === n.string)) prev.notes.push(n);
      prev.release = Math.max(prev.release, e.release);
      prev.pr = Math.max(prev.pr, e.pr);
    } else snapped.set(key, { b, k, notes: e.notes.slice(), release: e.release, pr: e.pr, t0: e.t0 });
  }
  const evs = Array.from(snapped.values()).sort((x, y) => x.b - y.b || x.k / (beatGrid[x.b] || 4) - y.k / (beatGrid[y.b] || 4));
  // absolute slot positions in "beat units"
  const posOf = (b, k) => b + k / (beatGrid[b] || 4);
  // durations: until next event, or until release (+ rest) when the notes clearly stop earlier
  const segsByBeat = new Map();
  const addSeg = (b, seg) => { if (!segsByBeat.has(b)) segsByBeat.set(b, []); segsByBeat.get(b).push(seg); };
  for (let i = 0; i < evs.length; i++) {
    const e = evs[i];
    const start = posOf(e.b, e.k);
    const nextStart = i + 1 < evs.length ? posOf(evs[i + 1].b, evs[i + 1].k) : Math.min(nBeats, Math.ceil(e.pr + 0.001));
    let end = nextStart;
    const relSnap = e.pr;
    if (relSnap < nextStart - 0.3) end = Math.max(start + 1 / (beatGrid[e.b] || 4), snapUp(relSnap, beatGrid));
    end = Math.min(end, nextStart);
    // split across beats
    let first = true;
    let cur = start;
    while (cur < end - 1e-9) {
      const b = Math.floor(cur + 1e-9);
      const g = beatGrid[b] || 4;
      const k0 = Math.round((cur - b) * g);
      const segEnd = Math.min(end, b + 1);
      let k1 = Math.round((segEnd - b) * g);
      if (k1 <= k0) k1 = k0 + 1;
      addSeg(b, { s: k0, len: k1 - k0, notes: e.notes, tie: !first });
      first = false;
      cur = b + k1 / g;
    }
  }
  // measures
  const m = meter || 4;
  let firstDown = rhythm.downbeatBeat ?? 0;
  while (firstDown - m >= 0) firstDown -= m;
  // measure 0 starts at firstDown - m if there's music before firstDown (pickup padded with rests)
  const firstEventBeat = evs.length ? evs[0].b : 0;
  let mStart = firstDown;
  while (mStart > firstEventBeat) mStart -= m;
  const lastBeat = evs.length ? Math.max(...Array.from(segsByBeat.keys())) : 0;
  const measures = [];
  for (let s = mStart; s <= lastBeat; s += m) {
    const mb = [];
    for (let j = 0; j < m; j++) {
      const b = s + j;
      const g = b >= 0 ? beatGrid[b] || 4 : 4;
      const segs = (segsByBeat.get(b) || []).slice().sort((x, y) => x.s - y.s);
      // fill gaps with rests
      const filled = [];
      let k = 0;
      for (const sg of segs) {
        if (sg.s > k) filled.push({ s: k, len: sg.s - k, notes: null });
        if (sg.s < k) { // overlapping (shouldn't happen) - clip
          const nl = sg.s + sg.len - k;
          if (nl > 0) filled.push({ ...sg, s: k, len: nl });
          k = Math.max(k, sg.s + sg.len);
          continue;
        }
        filled.push(sg);
        k = sg.s + sg.len;
      }
      if (k < g) filled.push({ s: k, len: g - k, notes: null });
      mb.push({ beat: b, grid: g, segs: filled, time: b >= 0 && b < beats.length ? beats[b] : null });
    }
    measures.push({ startBeat: s, beats: mb });
  }
  return { measures, beatGrid };
}

function snapUp(p, beatGrid) {
  const b = Math.floor(p);
  const g = beatGrid[b] || 4;
  return b + Math.ceil((p - b) * g - 0.25) / g;
}
