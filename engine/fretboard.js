// Fingering (string/fret) assignment and tuning + capo detection.
//
// The fingering is found with a beam search over "attack events" (notes that start
// together). Each hypothesis tracks what is still ringing on every string, where the
// fretting hand is (a 4-fret window), and how much it cost to get there: fret height,
// hand movement, stretches, cutting off notes that are still ringing, and so on.
// Tuning and capo are detected by asking which setup makes the transcribed notes
// easiest and most natural to play - the same reasoning a person uses by ear.

import { candidateSetups, soundingOpen, tuningName, TUNINGS, STANDARD } from './tunings.js';

export const FINGER_COSTS = {
  // tuned on GuitarSet: real players' string choices, on both annotated and transcribed notes
  // (tests/sweep_fingering.mjs, tests/sweep_finger_est.mjs)
  fretted: 0, // fretting a note vs. an open string
  fretHeight: 0, // per fret above the capo
  highFret: 0.3, // extra per fret above 12
  reach: 3, // frets the hand covers without moving (window start .. start+reach)
  stretch: 0.2, // per (fret of stretch beyond the window)^2
  maxSpan: 6, // never hold frets further apart than this
  move: 0.3, // per fret of hand movement (scaled by speed)
  moveBase: 0, // any position shift
  cut: 2.0, // cutting off a note that is still clearly ringing
  cutTime: 0.3, // seconds of remaining ring that count as a full cut
  drop: 7.0, // a note that cannot be played in this setup
  fingers: 2.5, // per finger needed beyond 4
  stringGap: 0.7, // per skipped string inside a chord of 3+ notes
  maxAbsFret: 22,
};

// Setup detection fingers the notes with the same profile (plus a slight tie-break toward
// open strings), then scores each setup from per-note features so long songs and short
// clips are judged the same way.
export const DETECT_COSTS = { ...FINGER_COSTS, fretted: 0.04 };
export const DETECT_WEIGHTS = { cost: 1.0, prior: 0.7, open: 1.0, openTime: 0.0, drop: 4.0, excess: 4.0, theta: 0.45 }; // tuned: tests/sweep_detect2.mjs
export const DEFAULT_COSTS = FINGER_COSTS;

const NSTR = 6;

/** Group notes into attack events (notes starting together). */
export function buildEvents(notes, chordWindow = 0.045) {
  const order = notes.map((n, i) => i).sort((a, b) => notes[a].t0 - notes[b].t0);
  const events = [];
  let cur = null;
  for (const i of order) {
    const n = notes[i];
    if (cur && n.t0 - cur.t <= chordWindow && !cur.notes.some((j) => notes[j].pitch === n.pitch) && cur.notes.length < NSTR) {
      cur.notes.push(i);
    } else {
      cur = { t: n.t0, notes: [i] };
      events.push(cur);
    }
  }
  for (const e of events) e.notes.sort((a, b) => notes[b].pitch - notes[a].pitch);
  return events;
}

function positions(pitch, setup, costs) {
  const out = [];
  for (let s = 0; s < NSTR; s++) {
    const rel = pitch - setup.tuning[s] - setup.capo;
    if (rel < 0) continue;
    if (rel + setup.capo > costs.maxAbsFret) continue;
    out.push({ s, f: rel });
  }
  return out;
}

function noteCost(f, costs) {
  if (f === 0) return 0;
  return costs.fretted + costs.fretHeight * f + (f > 12 ? costs.highFret * (f - 12) : 0);
}

/** enumerate assignments of an event's new notes to distinct strings (or drop) */
function enumerateAssignments(evNotes, notes, setup, costs, maxOut) {
  const opts = evNotes.map((i) => {
    const all = positions(notes[i].pitch, setup, costs);
    // a string the user picked by hand is kept when it still works
    const pin = notes[i].pin;
    if (pin != null) { const p = all.filter((q) => q.s === pin); if (p.length) return p; }
    return all;
  });
  const results = [];
  const cur = new Array(evNotes.length);
  const used = new Array(NSTR).fill(false);
  function rec(k, cost, nDrop) {
    if (k === evNotes.length) {
      let mn = 99, mx = -1, lo = 99, hi = -1, placed = 0;
      const frets = new Set();
      for (let j = 0; j < cur.length; j++) {
        const p = cur[j];
        if (!p) continue;
        placed++;
        lo = Math.min(lo, p.s); hi = Math.max(hi, p.s);
        if (p.f > 0) { mn = Math.min(mn, p.f); mx = Math.max(mx, p.f); frets.add(p.f); }
      }
      let c = cost;
      if (mx >= 0) {
        const span = mx - mn;
        if (span > costs.maxSpan) return;
        if (span > costs.reach) c += costs.stretch * (span - costs.reach) * (span - costs.reach);
        if (frets.size > 4) c += costs.fingers * (frets.size - 4);
      }
      if (placed >= 3 && hi > lo) {
        let gaps = 0;
        for (let s = lo; s <= hi; s++) if (!used[s]) gaps++;
        c += costs.stringGap * gaps;
      }
      results.push({ cost: c, place: cur.slice(), mn: mx >= 0 ? mn : -1, mx });
      return;
    }
    const o = opts[k];
    for (const p of o) {
      if (used[p.s]) continue;
      used[p.s] = true;
      cur[k] = p;
      rec(k + 1, cost + noteCost(p.f, costs), nDrop);
      used[p.s] = false;
    }
    cur[k] = null;
    const amp = notes[evNotes[k]].amp ?? 0.5;
    const dropCost = costs.drop * (o.length === 0 ? 1 : 1.5) * (0.5 + 0.5 * Math.min(1, amp / 0.5));
    rec(k + 1, cost + dropCost, nDrop + 1);
  }
  rec(0, 0, 0);
  results.sort((a, b) => a.cost - b.cost);
  return results.slice(0, maxOut);
}

// new hand window start given the previous one and the fretted range of the new notes
function moveHand(prev, mn, mx, reach) {
  if (mn < 0) return prev; // only open strings: hand stays
  const lo = Math.max(1, mx - reach), hi = mn; // feasible window starts (may be empty if span > reach)
  if (lo > hi) return mn;
  if (prev < 0) return Math.min(hi, Math.max(lo, mn));
  return Math.min(hi, Math.max(lo, prev));
}

/**
 * Beam search over events.
 * Returns {cost, assign: Map noteIndex -> {string, fret}} (string 0 = low string), dropped:Set}
 */
export function assignFingering(notes, events, setup, opts = {}) {
  const costs = { ...FINGER_COSTS, ...(opts.costs || {}) };
  const beamWidth = opts.beam ?? 48;
  const maxAssign = opts.maxAssign ?? 40;
  const ringTol = 0.05;
  const evRange = opts.eventRange || [0, events.length];
  let beam = [{ cost: 0, ringEnd: new Float64Array(NSTR).fill(-1), ringFret: new Int8Array(NSTR), ringW: new Float32Array(NSTR), hand: -1, lastT: -10, parent: null, place: null, ev: -1 }];
  for (let ei = evRange[0]; ei < evRange[1]; ei++) {
    const ev = events[ei];
    const assigns = enumerateAssignments(ev.notes, notes, setup, costs, maxAssign);
    const next = new Map();
    for (const st of beam) {
      const dt = Math.max(0, ev.t - st.lastT);
      const speed = Math.min(2.5, Math.max(0.1, 0.3 / (dt + 0.05)));
      for (const a of assigns) {
        let c = st.cost + a.cost;
        const ringEnd = Float64Array.from(st.ringEnd);
        const ringFret = Int8Array.from(st.ringFret);
        const ringW = Float32Array.from(st.ringW);
        for (let s = 0; s < NSTR; s++) if (ringEnd[s] >= 0 && ringEnd[s] <= ev.t + ringTol) ringEnd[s] = -1;
        const newOn = [false, false, false, false, false, false];
        for (let j = 0; j < a.place.length; j++) {
          const p = a.place[j];
          if (!p) continue;
          if (ringEnd[p.s] >= 0) c += costs.cut * Math.min(1, (ringEnd[p.s] - ev.t) / costs.cutTime) * ringW[p.s];
          const n = notes[ev.notes[j]];
          ringEnd[p.s] = n.t1;
          ringFret[p.s] = p.f;
          ringW[p.s] = 0.5 + 0.5 * Math.min(1, (n.amp ?? 0.4) / 0.4);
          newOn[p.s] = true;
        }
        const hand = moveHand(st.hand, a.mn, a.mx, costs.reach);
        if (a.mn >= 0 && st.hand >= 0 && hand !== st.hand) {
          c += (costs.moveBase + costs.move * Math.abs(hand - st.hand)) * speed;
        }
        if (a.mn >= 0) {
          // fretted notes still ringing from before must stay under the hand, or get released
          for (let s = 0; s < NSTR; s++) {
            if (ringEnd[s] < 0 || newOn[s] || ringFret[s] === 0) continue;
            const f = ringFret[s];
            if (f < hand - 1 || f > hand + costs.reach + 1) {
              c += costs.cut * 0.6 * Math.min(1, (ringEnd[s] - ev.t) / costs.cutTime) * ringW[s];
              ringEnd[s] = -1;
            }
          }
        }
        let key = hand + 1;
        for (let s = 0; s < NSTR; s++) key = key * 31 + (ringEnd[s] >= 0 ? ringFret[s] + 1 : 0);
        const prev = next.get(key);
        if (!prev || prev.cost > c) {
          next.set(key, { cost: c, ringEnd, ringFret, ringW, hand, lastT: ev.t, parent: st, place: a.place, ev: ei });
        }
      }
    }
    beam = Array.from(next.values()).sort((x, y) => x.cost - y.cost).slice(0, beamWidth);
    // free memory of long parent chains when only cost matters
    if (opts.costOnly) for (const st of beam) st.parent = null;
  }
  const best = beam[0];
  if (opts.costOnly) return { cost: best.cost };
  const assign = new Map();
  const dropped = new Set();
  let nOpen = 0, nPlaced = 0, wOpen = 0, wAll = 0;
  for (let st = best; st && st.ev >= 0; st = st.parent) {
    const ev = events[st.ev];
    st.place.forEach((p, j) => {
      const ni = ev.notes[j];
      if (p) {
        assign.set(ni, { string: p.s, fret: p.f });
        const d = Math.min(1.5, Math.max(0.05, notes[ni].t1 - notes[ni].t0));
        nPlaced++; wAll += d;
        if (p.f === 0) { nOpen++; wOpen += d; }
      } else dropped.add(ni);
    });
  }
  const nNotes = nPlaced + dropped.size;
  return { cost: best.cost, assign, dropped, stats: { openRatio: nPlaced ? nOpen / nPlaced : 0, openTime: wAll ? wOpen / wAll : 0, dropRatio: nNotes ? dropped.size / nNotes : 0 } };
}

function pickEventRanges(nEvents, maxEvents) {
  if (nEvents <= maxEvents) return [[0, nEvents]];
  const nSeg = 4;
  const segLen = Math.floor(maxEvents / nSeg);
  const ranges = [];
  for (let k = 0; k < nSeg; k++) {
    const start = Math.floor(((nEvents - segLen) * k) / (nSeg - 1));
    ranges.push([start, start + segLen]);
  }
  return ranges;
}

const openKey = (s) => soundingOpen(s.tuning, s.capo).join(',');

/** Candidate setup groups (setups that sound identical are grouped). */
export function candidateGroups(notes, opts = {}) {
  const pitches = notes.map((n) => n.pitch).sort((a, b) => a - b);
  let cands = candidateSetups({ tuning: opts.tuning, capo: opts.capo, maxCapo: opts.maxCapo ?? 10 });
  if (opts.preferTunings && opts.preferTunings.length) {
    for (const c of cands) if (opts.preferTunings.includes(c.tuning.join(','))) c.prior *= 3;
  }
  const filtered = cands.filter((c) => {
    const lowest = Math.min(...soundingOpen(c.tuning, c.capo));
    let below = 0;
    for (const p of pitches) { if (p < lowest) below++; else break; }
    return below <= Math.max(2, pitches.length * 0.04);
  });
  if (filtered.length) cands = filtered;
  const groups = new Map();
  for (const c of cands) {
    const k = openKey(c);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(c);
  }
  return Array.from(groups.entries()).map(([key, members]) => {
    members.sort((a, b) => b.prior - a.prior);
    return { key, members, prior: members.reduce((s, m) => s + m.prior, 0) };
  });
}

/** Per-group features: fingering cost per event, share of open strings, dropped notes. */
export function groupFeatures(notes, events, group, opts = {}) {
  const costs = { ...DETECT_COSTS, ...(opts.detectCosts || {}) };
  const ranges = opts.ranges || [[0, events.length]];
  let cost = 0, nEv = 0, open = 0, openT = 0, drop = 0, wsum = 0;
  for (const r of ranges) {
    const res = assignFingering(notes, events, group.members[0], { beam: opts.beam ?? 8, maxAssign: opts.maxAssign ?? 14, eventRange: r, costs });
    const n = r[1] - r[0];
    cost += res.cost; nEv += n;
    open += res.stats.openRatio * n; openT += res.stats.openTime * n; drop += res.stats.dropRatio * n; wsum += n;
  }
  return { costPerEvent: cost / Math.max(1, nEv), openRatio: open / Math.max(1, wsum), openTime: openT / Math.max(1, wsum), dropRatio: drop / Math.max(1, wsum) };
}

export function scoreGroup(f, prior, w = DETECT_WEIGHTS) {
  // open strings only count as evidence for a capo / open tuning when they dominate the sound
  const excess = Math.max(0, f.openTime - (w.theta ?? 0.35));
  return w.cost * f.costPerEvent + w.drop * f.dropRatio - w.open * f.openRatio - w.openTime * f.openTime - (w.excess || 0) * excess - w.prior * Math.log(prior);
}

/**
 * Rank (tuning, capo) setups for a set of notes.
 * opts: { tuning?: number[], capo?: number, preferTunings?: string[] (keys "38,45,..."), onProgress }
 */
export function rankSetups(notes, opts = {}) {
  const events = buildEvents(notes);
  if (!events.length) return { ranked: [], events };
  const w = { ...DETECT_WEIGHTS, ...(opts.weights || {}) };
  const groups = candidateGroups(notes, opts);
  const ranges = pickEventRanges(events.length, opts.quickEvents ?? 300);
  let done = 0;
  for (const g of groups) {
    g.f = groupFeatures(notes, events, g, { ranges, detectCosts: opts.detectCosts });
    g.score = scoreGroup(g.f, g.prior, w);
    done++;
    if (opts.onProgress && done % 8 === 0) opts.onProgress((0.6 * done) / groups.length);
  }
  groups.sort((a, b) => a.score - b.score);
  const top = groups.slice(0, opts.topK ?? 8);
  if (events.length > (opts.quickEvents ?? 300)) {
    let k = 0;
    for (const g of top) {
      g.f = groupFeatures(notes, events, g, { ranges: [[0, events.length]], beam: 16, maxAssign: 20, detectCosts: opts.detectCosts });
      g.score = scoreGroup(g.f, g.prior, w);
      k++;
      if (opts.onProgress) opts.onProgress(0.6 + (0.4 * k) / top.length);
    }
    top.sort((a, b) => a.score - b.score);
  }
  // confidence from score gaps (scores are per-event so a fixed temperature works)
  const temp = opts.temperature ?? 0.12;
  const z = top.map((g) => Math.exp(-(g.score - top[0].score) / temp));
  const zs = z.reduce((a, b) => a + b, 0);
  const ranked = top.map((g, i) => ({
    setup: g.members[0],
    equivalents: g.members.slice(1),
    features: g.f,
    score: g.score,
    confidence: z[i] / zs,
  }));
  return { ranked, events };
}

export { tuningName, TUNINGS };
