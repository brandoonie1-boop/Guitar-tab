// End-to-end: audio -> isolated guitar -> notes -> tuning/capo -> tab.
import { separate, DEMUCS } from './demucs.js';
import { runModel, outputToNotes, addPitchBends, frameTime, timeToFrame, BP } from './basicpitch.js';
import { HR, hrActivations, hrNotes } from './hrgt.js';
import { resample, toMono, rms } from './dsp.js';
import { analyzeRhythm, envSampler } from './beats.js';
import { beatActivations, beatsAndBars, BT } from './beatthis.js';
import { estimateTuningCents } from './pitchref.js';
import { splitStereo, chooseParts } from './stereo.js';
import { cleanNotes } from './notes.js';
import { rankSetups, assignFingering, buildEvents } from './fretboard.js';
import { detectTechniques } from './techniques.js';
import { quantize, extendBeats, beatPos } from './rhythm.js';
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
  // notes: 'hr' = the guitar transcription model when it is available, 'basic' = basic-pitch
  notesModel: 'hr',
  hrHop: 500, // frames between the model's 10 s windows (500 = half overlap, as trained)
  hrOnset: 0.4, // onset threshold (GuitarSet onset F1 89.5 at 0.3, 90.2 at 0.4)
  hrOffset: 0.3,
  hrFrame: 0.1,
  // basic-pitch note thresholds tuned on GuitarSet (onset F1 0.768 -> 0.815)
  onsetThresh: 0.6,
  frameThresh: 0.4,
  minNoteMs: 58, // 6+ frames; same F1 as 8+ on GuitarSet but keeps fast 16ths
  energyTol: 5,
  melodia: false,
  overlap: 0.25,
  preferTunings: [],
};

// share of the progress bar per stage (the note model takes about half as long as the
// guitar isolation per guitar part, on the processor)
export const STAGES = [
  ['prepare', 0.02], ['separate', 0.5], ['rhythm', 0.04], ['split', 0.03], ['notes', 0.3], ['setup', 0.08], ['tab', 0.03],
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

/**
 * Notes of a mono 44.1 kHz guitar signal, correcting the tuning offset first.
 * With the guitar transcription model (env.getTranscriber) its notes are used, and
 * basic-pitch only supplies the pitch-bend tracks and attack strengths that the technique
 * detection reads; without it basic-pitch finds the notes too.
 */
export async function detectNotes(env, mono44, opts, cents, onProgress) {
  const session = await env.getBasicPitch();
  // shift pitch so the instrument lines up with A440: play it r times faster
  const r = Math.abs(cents) >= 12 ? Math.pow(2, -cents / 1200) : 1;
  let hr = null;
  if (opts.notesModel !== 'basic' && env.getTranscriber) {
    try {
      hr = await env.getTranscriber();
    } catch (err) {
      // e.g. offline before the model was ever downloaded: basic-pitch still works
      if (env.cancelled && env.cancelled()) throw err;
      console.warn('note model unavailable, using basic-pitch:', (err && err.message) || err);
    }
  }
  const a22 = resample(mono44, DEMUCS.sr * r, BP.sr);
  const peakNorm = Math.max(1e-6, rms(a22)) ;
  // basic-pitch was trained on normalised audio; keep levels sane
  const g = Math.min(20, 0.12 / peakNorm);
  for (let i = 0; i < a22.length; i++) a22[i] *= g;
  const share = hr ? 0.12 : 1;
  const model = await runModel(env.ort, session, a22, { onProgress: (f) => onProgress(share * f), shouldCancel: env.cancelled });
  const nF = BP.nNotes;
  const onsetAt = (s, pitch) => {
    let op = 0;
    for (let t = Math.max(0, s - 2); t <= Math.min(model.nT - 1, s + 2); t++) op = Math.max(op, model.onset[t * nF + (pitch - 21)]);
    return op;
  };
  if (hr) {
    const x16 = resample(mono44, DEMUCS.sr * r, HR.sr, 64, 0.958);
    // bigger chunks keep a graphics chip busy; small ones keep the processor's memory low
    const act = await hrActivations(env.ort, hr, x16, { hop: opts.hrHop, chunk: hr.onGpu ? 500 : 250, onProgress: (f) => onProgress(share + (1 - share) * f), shouldCancel: env.cancelled });
    const found = hrNotes(act, { onsetThreshold: opts.hrOnset, offsetThreshold: opts.hrOffset, frameThreshold: opts.hrFrame });
    const last = model.nT - 1;
    const framed = found.map((n) => {
      const s = Math.min(last, Math.max(0, Math.round(timeToFrame(n.t0))));
      return { s, e: Math.min(last + 1, Math.max(s + 1, Math.round(timeToFrame(n.t1)))), pitch: n.pitch, note: n };
    });
    addPitchBends(model, framed);
    // how strongly basic-pitch hears the same note (distortion makes the transcription model
    // hear confident "ghost" notes, e.g. difference tones below the chord, that it doesn't)
    const support = (s, pitch) => {
      let v = 0;
      for (let t = Math.max(0, s - 3); t <= Math.min(model.nT - 1, s + 3); t++) v = Math.max(v, model.note[t * nF + (pitch - 21)]);
      return v;
    };
    return framed.map((f) => ({
      t0: f.note.t0 * r, t1: f.note.t1 * r, pitch: f.pitch, amp: f.note.amp, velocity: f.note.velocity,
      bends: f.bends, onsetProb: onsetAt(f.s, f.pitch), hr: true, conf: f.note.onsetProb, support: support(f.s, f.pitch),
    }));
  }
  const minLen = Math.max(3, Math.round((opts.minNoteMs / 1000) * (BP.sr / BP.hop)));
  let notes = outputToNotes(model, { onsetThresh: opts.onsetThresh, frameThresh: opts.frameThresh, minNoteLen: minLen, energyTol: opts.energyTol, melodiaTrick: opts.melodia });
  notes = addPitchBends(model, notes);
  // raw onset probability around the start (used for legato detection)
  return notes.map((n) => ({ t0: frameTime(n.s) * r, t1: frameTime(n.e) * r, pitch: n.pitch, amp: n.amp, bends: n.bends, onsetProb: onsetAt(n.s, n.pitch) }));
}

/**
 * Stage 1: isolate the guitar and build the signal the beat tracker listens to.
 * The app runs this in its own worker and ends that worker afterwards, so the separation
 * model's memory is gone before stage 2 starts.
 * @param audio {left, right, sr(=44100), name, offset?}. Takes ownership of the arrays (it
 *   clears audio.left/right) so memory can be released as soon as they are used.
 * @param env {ort, getDemucs, releaseDemucs?, progress, cancelled, checkSeparation?, fallbackToCpu?}
 * @returns {gL, gR, mix22 (mono mix at 22.05 kHz for the beat tracker), stereo, sr, offset, duration, name}
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
  if (opts.isolate === 'off') return { ...meta, gL: left, gR: right, mix22: null };
  progress('separate', 0, 'Isolating the guitar');
  let session = await env.getDemucs((f) => progress('separate', 0.02 * f, 'Loading the separation model'));
  // only what is used later: the guitar (stereo, for the two-guitar split)
  const keep = { guitar: 'stereo' };
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
  st = null;
  // the beat tracker listens to the whole mix (what it was trained on), at 22.05 kHz
  const mix22 = mixForBeats(left, right);
  left = null; right = null;
  if (cancelled()) throw new Error('cancelled');
  return { ...meta, gL, gR, mix22 };
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
  // 2. beats and bars
  progress('rhythm', 0, 'Finding the beat');
  let mix22 = stems.mix22 || mixForBeats(gL, gR);
  stems.mix22 = null;
  let rh = null;
  if (env.getBeatModel) {
    try {
      const bs = await env.getBeatModel();
      const act = await beatActivations(env.ort, bs, mix22, BT.sr, { onProgress: (f) => progress('rhythm', 0.95 * f, 'Finding the beat'), shouldCancel: env.cancelled });
      const bb = beatsAndBars(act, { meter: opts.meter || undefined });
      if (bb.beats.length >= 8) rh = { bpm: bb.bpm, bpmCandidates: [bb.bpm], beats: bb.beats, meter: bb.meter, downbeatPhase: bb.downbeatPhase, bars: bb.bars, tracker: 'beat-this' };
    } catch (err) {
      if (err && err.message === 'cancelled') throw err;
      console.warn('beat model failed, using the simple beat tracker', err);
    }
  }
  if (!rh) {
    const r = analyzeRhythm(mix22, BT.sr, { bpm: opts.bpm, meter: opts.meter });
    rh = { bpm: r.bpm, bpmCandidates: r.bpmCandidates, beats: r.beats, meter: r.meter, downbeatPhase: r.downbeatPhase, bars: null, tracker: 'simple' };
  }
  mix22 = null;
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
    // the transcription model hears re-attacks itself; basic-pitch's need the re-trigger check
    const hrNotesFound = raw.length > 0 && raw[0].hr;
    const notes = cleanNotes(raw, { idPrefix: pd.id + ':', mergeFragments: !hrNotesFound, mergeRetrigger: hrNotesFound ? false : opts.mergeRetrigger, retriggerRatio: opts.retriggerRatio, retriggerAbs: opts.retriggerAbs });
    if (notes.length < 3 && partDefs.length > 1) continue;
    // a single part's audio is the guitar stem itself; only split parts keep theirs (for playback)
    parts.push({ id: pd.id, name: pd.name, audio: partDefs.length > 1 ? pd.audio : null, notes });
  }
  partDefs = null;
  if (!parts.length) throw new Error('No guitar notes found in this file.');
  // drop a side part that just duplicates another (double-tracked guitars)
  if (parts.length > 1) dedupeParts(parts);
  // 5. tuning + capo, judged from all the parts together (one song, one guitarist's setup as
  //    a rule; each part can still be changed afterwards), then fingering per part.
  //    The notes both detectors agree on (or that the transcription model is sure of) decide,
  //    so a few distortion ghosts can't drag the tuning down.
  const sureNotes = parts.map((p) => {
    const sure = p.notes.filter((n) => !(n.hr && (n.support ?? 1) < 0.2 && (n.conf ?? 1) < 0.7));
    return sure.length >= 0.5 * p.notes.length ? sure : p.notes;
  });
  const rank = rankSetups(sureNotes, {
    tuning: opts.tuning || undefined,
    capo: opts.capo ?? undefined,
    preferTunings: opts.preferTunings,
    onProgress: (f) => progress('setup', 0.8 * f, 'Working out tuning and capo'),
  });
  const ranking = rank.ranked.slice(0, 6).map((r) => ({
    tuning: r.setup.tuning, capo: r.setup.capo, name: tuningName(r.setup.tuning), confidence: r.confidence,
    equivalents: r.equivalents.filter((e) => e.prior > 0.02).slice(0, 3).map((e) => ({ tuning: e.tuning, capo: e.capo, name: tuningName(e.tuning) })),
  }));
  for (let pi = 0; pi < parts.length; pi++) {
    const p = parts[pi];
    p.ranking = ranking.map((r) => ({ ...r, equivalents: r.equivalents.slice() }));
    applySetup(p, p.ranking[0]);
    progress('setup', 0.8 + (0.2 * (pi + 1)) / parts.length, 'Working out tuning and capo');
  }
  // 6. rhythm grid + tab
  progress('tab', 0, 'Writing the tab');
  const result = {
    name: stems.name || 'Song',
    duration, offset, sr,
    tuningCents: cents, tuningRef,
    splitInfo,
    rhythm: rh,
    stems: { guitar: [gL, gR] },
    parts,
  };
  layout(result);
  progress('tab', 1, 'Done');
  return result;
}

/** Mono mix at 22.05 kHz with a steep low-pass like soxr's (what the beat model expects). */
export function mixForBeats(left, right) {
  const n = left.length;
  const mono = new Float32Array(n);
  if (right && right !== left) for (let i = 0; i < n; i++) mono[i] = (left[i] + right[i]) / 2;
  else mono.set(left);
  return resample(mono, DEMUCS.sr, BT.sr, 64, 0.958);
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
    rh.bars = null;
  }
  const beats = extendBeats(rh.beats, result.duration, rh.bpm);
  // index of first tracked beat in the extended list
  const firstTracked = rh.beats.length ? Math.max(0, beats.findIndex((b) => Math.abs(b - rh.beats[0]) < 1e-6)) : 0;
  const meter = rh.meter || 4;
  // bar lines: the beat model's (the meter may change) or a steady meter from the first downbeat
  const barStarts = rh.bars && rh.bars.length ? rh.bars.map((b) => firstTracked + b.beat) : [firstTracked + (rh.downbeatPhase || 0)];
  const downbeatBeat = barStarts[0];
  result.grid = { beats, meter, downbeatBeat, barStarts };
  // every part gets the same bars, covering the notes of all parts
  let t0 = Infinity, t1 = -Infinity;
  const placed = result.parts.map((p) => p.notes.filter((n) => !n.dropped && !n.deleted && n.string != null));
  for (const ns of placed) for (const n of ns) { t0 = Math.min(t0, n.t0); t1 = Math.max(t1, n.t1); }
  const range = Number.isFinite(t0) ? { firstBeat: Math.floor(beatPos(beats, t0)), lastBeat: Math.max(0, Math.ceil(beatPos(beats, t1)) - 1) } : { firstBeat: 0, lastBeat: 0 };
  result.parts.forEach((p, i) => {
    const q = quantize(placed[i], { beats, meter, downbeatBeat, barStarts }, range);
    p.measures = q.measures;
  });
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
