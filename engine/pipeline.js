// End-to-end: audio -> isolated guitar -> notes -> tuning/capo -> tab.
import { separate, DEMUCS } from './demucs.js';
import { runModel, outputToNotes, addPitchBends, frameTime, BP } from './basicpitch.js';
import { resample, toMono, rms } from './dsp.js';
import { analyzeRhythm, envSampler } from './beats.js';
import { estimateTuningCents } from './pitchref.js';
import { splitStereo, chooseParts } from './stereo.js';
import { cleanNotes } from './notes.js';
import { rankSetups, assignFingering, buildEvents } from './fretboard.js';
import { detectTechniques } from './techniques.js';
import { quantize, extendBeats } from './rhythm.js';
import { toAlphaTex, toAscii } from './export.js';
import { tuningName } from './tunings.js';

export const DEFAULT_OPTS = {
  isolate: 'guitar', // 'guitar' | 'guitar+other' | 'off'
  split: 'auto', // 'auto' | 'off'
  tuning: null,
  capo: null,
  bpm: null,
  meter: null,
  range: null,
  // note detection thresholds tuned on GuitarSet (onset F1 0.768 -> 0.815)
  onsetThresh: 0.6,
  frameThresh: 0.4,
  minNoteMs: 58, // 6+ frames; same F1 as 8+ on GuitarSet but keeps fast 16ths
  energyTol: 5,
  melodia: false,
  overlap: 0.25,
  preferTunings: [],
};

const STAGES = [
  ['prepare', 0.02], ['separate', 0.62], ['rhythm', 0.04], ['split', 0.04], ['notes', 0.14], ['setup', 0.1], ['tab', 0.04],
];

function makeProgress(cb) {
  const starts = {};
  let acc = 0;
  for (const [k, w] of STAGES) { starts[k] = [acc, w]; acc += w; }
  return (stage, frac, label) => {
    if (!cb) return;
    const [s, w] = starts[stage] || [0, 0];
    cb({ stage, frac: Math.min(1, s + w * Math.max(0, Math.min(1, frac))), label });
  };
}

/** Run basic-pitch on a mono 44.1 kHz signal, correcting the tuning offset first. */
export async function detectNotes(env, mono44, opts, cents, onProgress) {
  const session = await env.getBasicPitch();
  // shift pitch so the instrument lines up with A440: play it r times faster
  const r = Math.abs(cents) >= 12 ? Math.pow(2, -cents / 1200) : 1;
  const a22 = resample(mono44, DEMUCS.sr * r, BP.sr);
  const peakNorm = Math.max(1e-6, rms(a22)) ;
  // basic-pitch was trained on normalised audio; keep levels sane
  const g = Math.min(20, 0.12 / peakNorm);
  for (let i = 0; i < a22.length; i++) a22[i] *= g;
  const model = await runModel(env.ort, session, a22, { onProgress, shouldCancel: env.cancelled });
  const minLen = Math.max(3, Math.round((opts.minNoteMs / 1000) * (BP.sr / BP.hop)));
  let notes = outputToNotes(model, { onsetThresh: opts.onsetThresh, frameThresh: opts.frameThresh, minNoteLen: minLen, energyTol: opts.energyTol, melodiaTrick: opts.melodia });
  notes = addPitchBends(model, notes);
  const nF = BP.nNotes;
  return notes.map((n) => {
    // raw onset probability around the start (used for legato detection)
    let op = 0;
    for (let t = Math.max(0, n.s - 2); t <= Math.min(model.nT - 1, n.s + 2); t++) op = Math.max(op, model.onset[t * nF + (n.pitch - 21)]);
    return { t0: frameTime(n.s) * r, t1: frameTime(n.e) * r, pitch: n.pitch, amp: n.amp, bends: n.bends, onsetProb: op };
  });
}

/**
 * Stage 1: isolate the guitar and build the signal the beat tracker listens to.
 * The app runs this in its own worker and ends that worker afterwards, so the separation
 * model's memory is gone before stage 2 starts.
 * @param audio {left, right, sr(=44100), name, offset?}. Takes ownership of the arrays (it
 *   clears audio.left/right) so memory can be released as soon as they are used.
 * @param env {ort, getDemucs, releaseDemucs?, progress, cancelled, checkSeparation?, fallbackToCpu?}
 * @returns {gL, gR, rhythm, stereo, sr, offset, duration, name}
 */
export async function separateStage(audio, userOpts = {}, env) {
  const opts = { ...DEFAULT_OPTS, ...userOpts };
  const progress = makeProgress(env.progress);
  const cancelled = env.cancelled || (() => false);
  const sr = audio.sr;
  if (sr !== DEMUCS.sr) throw new Error('audio must be 44.1 kHz');
  let left = audio.left, right = audio.right || audio.left;
  const stereo = !!audio.right && audio.right !== audio.left;
  audio.left = null; audio.right = null;
  let offset = audio.offset || 0;
  if (opts.range) {
    // copy the section so the rest of the song can be freed
    const a = Math.max(0, Math.floor(opts.range[0] * sr)), b = Math.min(left.length, Math.ceil(opts.range[1] * sr));
    left = left.slice(a, b);
    right = stereo ? right.slice(a, b) : left;
    offset += a / sr;
  }
  const meta = { stereo, sr, offset, duration: left.length / sr, name: audio.name || 'Song' };
  progress('prepare', 1, 'Getting ready');
  if (opts.isolate === 'off') return { ...meta, gL: left, gR: right, rhythm: null };
  progress('separate', 0, 'Isolating the guitar');
  let session = await env.getDemucs((f) => progress('separate', 0.02 * f, 'Loading the separation model'));
  // only what is used later: guitar (stereo, for the two-guitar split), drums (mono, for the beat)
  const keep = { guitar: 'stereo', drums: 'mono' };
  if (opts.isolate === 'guitar+other') keep.other = 'stereo';
  const sepOpts = {
    keep,
    overlap: opts.overlap,
    onProgress: (f) => progress('separate', 0.02 + 0.98 * f, 'Isolating the guitar'),
    shouldCancel: cancelled,
    check: !!env.checkSeparation,
  };
  let st;
  try {
    st = await separate(env.ort, session, left, right, sepOpts);
  } catch (err) {
    // the fast (GPU) path failed or produced bad output: redo it on the processor
    if (!env.fallbackToCpu || (err && err.message === 'cancelled')) throw err;
    progress('separate', 0, 'Switching to the processor');
    const cpu = await env.fallbackToCpu(err);
    session = cpu.session;
    st = await separate(cpu.ort, cpu.session, left, right, { ...sepOpts, check: false });
  }
  session = null;
  if (env.releaseDemucs) await env.releaseDemucs();
  const gL = st.guitar[0], gR = st.guitar[1];
  if (st.other) {
    const [oL, oR] = st.other;
    for (let i = 0; i < gL.length; i++) { gL[i] += oL[i]; gR[i] += oR[i]; }
  }
  // the beat tracker hears the mix with the drums brought forward, built in place in the drums array
  const rhythm = st.drums[0];
  for (let i = 0; i < rhythm.length; i++) rhythm[i] = 0.5 * Math.fround((left[i] + right[i]) / 2) + rhythm[i];
  st = null; left = null; right = null;
  if (cancelled()) throw new Error('cancelled');
  return { ...meta, gL, gR, rhythm };
}

/**
 * Stage 2: beat, two-guitar split, notes, tuning and capo, fingering, tab.
 * @param stems separateStage() result (takes ownership of its arrays)
 * @param env {ort, getBasicPitch, progress, cancelled}
 */
export async function analyzeStage(stems, userOpts = {}, env) {
  const opts = { ...DEFAULT_OPTS, ...userOpts };
  const progress = makeProgress(env.progress);
  const { sr, stereo, offset, duration } = stems;
  const gL = stems.gL, gR = stems.gR || stems.gL;
  // 2. rhythm
  progress('rhythm', 0, 'Finding the beat');
  let rhythmSrc = stems.rhythm || toMono([gL, gR]);
  stems.rhythm = null;
  const rh = analyzeRhythm(rhythmSrc, sr, { bpm: opts.bpm, meter: opts.meter });
  rhythmSrc = null;
  progress('rhythm', 1, 'Finding the beat');
  // 3. split guitars by stereo position
  progress('split', 0, 'Checking for more than one guitar');
  let guitarMono = toMono([gL, gR]);
  let partDefs = [{ id: 'all', name: 'Guitar', audio: guitarMono }];
  let splitInfo = null;
  if (opts.split !== 'off' && stereo) {
    const sp = splitStereo(gL, gR);
    splitInfo = { ...sp.stats, ...chooseParts(sp.stats) };
    if (splitInfo.mode === 'split') {
      const names = ['left', 'right'];
      if (splitInfo.parts.includes('center')) names.push('center');
      const r = sp.render(names);
      partDefs = [];
      partDefs.push({ id: 'left', name: 'Guitar 1 (left)', audio: r.left });
      partDefs.push({ id: 'right', name: 'Guitar 2 (right)', audio: r.right });
      if (r.center) partDefs.push({ id: 'center', name: 'Guitar 3 (center)', audio: r.center });
    }
  }
  progress('split', 1, 'Checking for more than one guitar');
  // 4. tuning reference + notes per part
  const tuningRef = estimateTuningCents(guitarMono, sr);
  guitarMono = null;
  const cents = tuningRef.confidence > 0.15 ? tuningRef.cents : 0;
  const parts = [];
  for (let pi = 0; pi < partDefs.length; pi++) {
    const pd = partDefs[pi];
    const level = rms(pd.audio);
    if (level < 1e-4) continue;
    const raw = await detectNotes(env, pd.audio, opts, cents, (f) => progress('notes', (pi + f) / partDefs.length, `Hearing the notes${partDefs.length > 1 ? ` (${pd.name})` : ''}`));
    const notes = cleanNotes(raw, { idPrefix: pd.id + ':', mergeRetrigger: opts.mergeRetrigger, retriggerRatio: opts.retriggerRatio, retriggerAbs: opts.retriggerAbs });
    if (notes.length < 3 && partDefs.length > 1) continue;
    // a single part's audio is the guitar stem itself; only split parts keep theirs (for playback)
    parts.push({ id: pd.id, name: pd.name, audio: partDefs.length > 1 ? pd.audio : null, notes });
  }
  partDefs = null;
  if (!parts.length) throw new Error('No guitar notes found in this file.');
  // drop a side part that just duplicates another (double-tracked guitars)
  if (parts.length > 1) dedupeParts(parts);
  // 5. tuning + capo per part, then fingering
  for (let pi = 0; pi < parts.length; pi++) {
    const p = parts[pi];
    const rank = rankSetups(p.notes, {
      tuning: opts.tuning || undefined,
      capo: opts.capo ?? undefined,
      preferTunings: opts.preferTunings,
      onProgress: (f) => progress('setup', (pi + f) / parts.length, 'Working out tuning and capo'),
    });
    p.ranking = rank.ranked.slice(0, 6).map((r) => ({
      tuning: r.setup.tuning, capo: r.setup.capo, name: tuningName(r.setup.tuning), confidence: r.confidence,
      equivalents: r.equivalents.filter((e) => e.prior > 0.02).slice(0, 3).map((e) => ({ tuning: e.tuning, capo: e.capo, name: tuningName(e.tuning) })),
    }));
    applySetup(p, p.ranking[0]);
  }
  // 6. rhythm grid + tab
  progress('tab', 0, 'Writing the tab');
  const result = {
    name: stems.name || 'Song',
    duration, offset, sr,
    tuningCents: cents, tuningRef,
    splitInfo,
    rhythm: { bpm: rh.bpm, bpmCandidates: rh.bpmCandidates, beats: rh.beats, meter: rh.meter, downbeatPhase: rh.downbeatPhase },
    stems: { guitar: [gL, gR] },
    parts,
  };
  layout(result);
  progress('tab', 1, 'Done');
  return result;
}

/** Both stages in one go (Node tools and tests). */
export async function transcribe(audio, userOpts = {}, env) {
  const stems = await separateStage(audio, userOpts, env);
  return analyzeStage(stems, userOpts, env);
}

function dedupeParts(parts) {
  // if two parts share most notes (same pitch, close onsets) they are the same riff doubled
  const sim = (a, b) => {
    let hit = 0;
    let j = 0;
    const bs = b.notes;
    for (const n of a.notes) {
      while (j < bs.length && bs[j].t0 < n.t0 - 0.05) j++;
      for (let k = j; k < bs.length && bs[k].t0 <= n.t0 + 0.05; k++) if (bs[k].pitch === n.pitch) { hit++; break; }
    }
    return hit / Math.max(1, Math.min(a.notes.length, b.notes.length));
  };
  for (let i = 0; i < parts.length; i++) {
    for (let j = i + 1; j < parts.length; j++) {
      if (sim(parts[i], parts[j]) > 0.7) {
        parts[i].name = parts[i].name.replace(/ \((left|right|center)\)/, '') + ' (doubled left/right)';
        parts.splice(j, 1);
        j--;
      }
    }
  }
  parts.forEach((p, i) => { p.name = p.name.replace(/^Guitar \d+/, parts.length > 1 ? `Guitar ${i + 1}` : 'Guitar'); });
}

/** (Re)assign strings/frets for a part with a chosen setup {tuning, capo}. */
export function applySetup(part, setup) {
  part.setup = { tuning: setup.tuning.slice(), capo: setup.capo, name: tuningName(setup.tuning) };
  const events = buildEvents(part.notes);
  const fa = assignFingering(part.notes, events, part.setup, { beam: 48 });
  part.notes.forEach((n, i) => {
    const a = fa.assign.get(i);
    if (a) { n.string = a.string; n.fret = a.fret; n.dropped = false; }
    else { n.string = null; n.fret = null; n.dropped = true; }
  });
  part.tech = detectTechniques(part.notes.filter((n) => !n.dropped && !n.deleted));
}

/** Rhythm grid, measures and text outputs for all parts. */
export function layout(result, override = {}) {
  const rh = result.rhythm;
  if (override.bpm || override.meter || override.shift != null) {
    if (override.meter) rh.meter = override.meter;
    if (override.shift != null) rh.downbeatPhase = ((rh.downbeatPhase + override.shift) % rh.meter + rh.meter) % rh.meter;
  }
  const beats = extendBeats(rh.beats, result.duration, rh.bpm);
  // index of first downbeat in the extended list
  const firstTracked = rh.beats.length ? beats.findIndex((b) => Math.abs(b - rh.beats[0]) < 1e-6) : 0;
  const downbeatBeat = Math.max(0, firstTracked) + (rh.downbeatPhase || 0);
  result.grid = { beats, meter: rh.meter, downbeatBeat };
  for (const p of result.parts) {
    const placed = p.notes.filter((n) => !n.dropped && !n.deleted && n.string != null);
    const q = quantize(placed, { beats, meter: rh.meter, downbeatBeat });
    p.measures = q.measures;
  }
  // all parts should have the same number of measures
  const nM = Math.max(...result.parts.map((p) => p.measures.length));
  for (const p of result.parts) {
    while (p.measures.length < nM) {
      const last = p.measures[p.measures.length - 1];
      const s = last ? last.startBeat + rh.meter : 0;
      p.measures.push({ startBeat: s, beats: Array.from({ length: rh.meter }, (_, j) => ({ beat: s + j, grid: 4, segs: [{ s: 0, len: 4, notes: null }] })) });
    }
    // align start beat with the first part
    const s0 = Math.min(...result.parts.map((q) => q.measures[0]?.startBeat ?? 0));
    while (p.measures.length && p.measures[0].startBeat > s0) {
      const s = p.measures[0].startBeat - rh.meter;
      p.measures.unshift({ startBeat: s, beats: Array.from({ length: rh.meter }, (_, j) => ({ beat: s + j, grid: 4, segs: [{ s: 0, len: 4, notes: null }] })) });
    }
  }
  const doc = docOf(result);
  const at = toAlphaTex(doc);
  result.alphaTex = at.tex;
  result.texMaps = at.maps;
  result.ascii = toAscii(doc);
  // sync points: audio time of each measure start (for following the recording)
  const ref = result.parts[0];
  result.syncPoints = ref.measures.map((m, i) => {
    const b = m.startBeat;
    const t = b >= 0 && b < beats.length ? beats[b] : (b < 0 ? beats[0] + b * (beats[1] - beats[0]) : beats[beats.length - 1]);
    return { barIndex: i, barOccurence: 0, barPosition: 0, millisecondOffset: Math.max(0, (t + (result.offset || 0)) * 1000) };
  });
  return result;
}

export function docOf(result) {
  const title = result.name.replace(/\.[a-z0-9]+$/i, '');
  return {
    title,
    subtitle: result.parts.length === 1 ? setupLine(result.parts[0]) : '',
    bpm: result.rhythm.bpm,
    meter: result.rhythm.meter,
    parts: result.parts.map((p) => ({ name: p.name, tuning: p.setup.tuning, capo: p.setup.capo, measures: p.measures, tech: p.tech, notes: p.notes })),
  };
}

export function setupLine(p) {
  return `Tuning: ${tuningName(p.setup.tuning)}${p.setup.capo ? ` · Capo ${p.setup.capo}` : ''}`;
}
