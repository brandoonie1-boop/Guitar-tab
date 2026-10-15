// Guitar techniques from the detected notes + pitch contours:
// bends, vibrato, hammer-ons / pull-offs, slides.
// Conservative on purpose: a missing "h" is less confusing than a wrong one.

export function detectTechniques(notes, opts = {}) {
  const tech = new Map();
  const get = (id) => { if (!tech.has(id)) tech.set(id, {}); return tech.get(id); };
  // the pitch track follows the strongest pitch within several semitones of a note, so a
  // neighbouring note that rings at the same time can look like a bend or vibrato
  const byStart = notes.slice().sort((a, b) => a.t0 - b.t0);
  const neighbour = (n, lo, hi) => byStart.some((m) => m !== n && m.t0 < n.t1 - 0.05 && m.t1 > n.t0 + 0.05 && m.pitch !== n.pitch && m.pitch >= n.pitch + lo && m.pitch <= n.pitch + hi);
  // bends / vibrato from the pitch-bend track (1/3 semitone units per frame)
  for (const n of notes) {
    const b = n.bends;
    if (!b || b.length < 6) continue;
    if (n.fret === 0) continue; // an open string can't be bent or shaken
    const start = median(b.slice(0, Math.min(3, b.length)));
    let mx = -99, mxi = 0;
    for (let i = 0; i < b.length; i++) if (b[i] > mx) { mx = b[i]; mxi = i; }
    const rise = mx - start;
    // a bend: stable start, then a rise of at least a semitone that is held for a while
    const head = b.slice(0, Math.min(4, b.length));
    const stableStart = Math.max(...head) - Math.min(...head) <= 1;
    let held = 0;
    for (let i = mxi; i < b.length && b[i] >= start + rise - 1; i++) held++;
    if (stableStart && rise >= 3 && held >= 4 && mxi >= 3 && n.t1 - n.t0 >= 0.18) {
      const semis = Math.round((rise / 3) * 2) / 2; // half-step resolution
      if (semis >= 0.5 && !neighbour(n, 1, Math.ceil(semis) + 1)) get(n.id).bend = Math.min(3, semis);
      continue;
    }
    // vibrato: several swings of at least 1/3 semitone around the centre
    const c = median(b);
    let swings = 0, sign = 0;
    for (const v of b) {
      const d = v - c;
      if (Math.abs(d) >= 1) {
        const s = Math.sign(d);
        if (s !== sign) { swings++; sign = s; }
      }
    }
    if (swings >= 4 && n.t1 - n.t0 > 0.35 && !neighbour(n, -2, 2)) get(n.id).vibrato = true;
  }
  // legato & slides: consecutive notes on one string, second one barely re-attacked
  const byString = new Map();
  for (const n of notes) {
    if (n.string == null) continue;
    if (!byString.has(n.string)) byString.set(n.string, []);
    byString.get(n.string).push(n);
  }
  const weak = opts.weakOnset ?? 0.32;
  for (const list of byString.values()) {
    list.sort((a, b) => a.t0 - b.t0);
    for (let i = 0; i + 1 < list.length; i++) {
      const a = list[i], b = list[i + 1];
      const gap = b.t0 - a.t1;
      if (gap > 0.04 || gap < -0.12) continue;
      if (b.t0 - a.t0 > 0.6) continue;
      if ((b.onsetProb ?? 1) > weak) continue;
      const df = b.fret - a.fret;
      if (df === 0 || b.fret === 0 && a.fret === 0) continue;
      // a glide at the end of the first note toward the second one means slide
      const glide = a.bends && a.bends.length > 3 ? a.bends[a.bends.length - 1] - a.bends[Math.max(0, a.bends.length - 4)] : 0;
      const t = get(a.id);
      if (Math.abs(df) >= 2 && Math.sign(glide) === Math.sign(df) && Math.abs(glide) >= 2) {
        t.slide = true; t.slideUp = df > 0;
      } else if (Math.abs(df) <= 5) {
        t.legato = true; t.legatoUp = df > 0;
      }
    }
  }
  return tech;
}

function median(a) {
  const s = a.slice().sort((x, y) => x - y);
  return s.length ? s[s.length >> 1] : 0;
}
