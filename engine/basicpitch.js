// Polyphonic note detection with Spotify's basic-pitch model (ONNX), plus a JS port
// of its note-creation post-processing (basic_pitch/note_creation.py, Apache-2.0).

export const BP = {
  sr: 22050,
  hop: 256,
  nSamples: 43844, // 2 s window minus one hop
  nFrames: 172,
  overlapFrames: 30,
  nNotes: 88,
  nContour: 264,
  midiOffset: 21,
};
const OVERLAP_LEN = BP.overlapFrames * BP.hop; // 7680
const HOP_SIZE = BP.nSamples - OVERLAP_LEN; // 36164
const N_OLAP = BP.overlapFrames / 2; // 15
const FRAMES_PER_WINDOW = BP.nFrames - BP.overlapFrames; // 142

/** time in seconds of unwrapped output frame f (exact for the windowing used here) */
export function frameTime(f) {
  const w = Math.floor(f / FRAMES_PER_WINDOW);
  return (f * BP.hop - w * (FRAMES_PER_WINDOW * BP.hop - HOP_SIZE)) / BP.sr;
}

/** inverse of frameTime (approximate, monotonic) */
export function timeToFrame(t) {
  return t * BP.sr / (HOP_SIZE / FRAMES_PER_WINDOW);
}

/**
 * Run the model over mono 22.05 kHz audio.
 * @returns {{nT:number, note:Float32Array, onset:Float32Array, contour:Float32Array}}
 */
export async function runModel(ort, session, audio, opts = {}) {
  const batch = opts.batch || 4;
  const n = audio.length;
  const padded = new Float32Array(OVERLAP_LEN / 2 + n);
  padded.set(audio, OVERLAP_LEN / 2);
  const starts = [];
  for (let i = 0; i < padded.length; i += HOP_SIZE) starts.push(i);
  const nExpected = Math.floor((n / HOP_SIZE) * FRAMES_PER_WINDOW);
  const nT = nExpected;
  const note = new Float32Array(nT * BP.nNotes);
  const onset = new Float32Array(nT * BP.nNotes);
  const contour = new Float32Array(nT * BP.nContour);
  const inName = session.inputNames[0];
  // output names: note=":1", onset=":2", contour=":0"
  const outNames = session.outputNames;
  const nameNote = outNames.find((s) => s.endsWith(':1'));
  const nameOnset = outNames.find((s) => s.endsWith(':2'));
  const nameContour = outNames.find((s) => s.endsWith(':0'));
  for (let b0 = 0; b0 < starts.length; b0 += batch) {
    if (opts.shouldCancel && opts.shouldCancel()) throw new Error('cancelled');
    const nb = Math.min(batch, starts.length - b0);
    const x = new Float32Array(nb * BP.nSamples);
    for (let k = 0; k < nb; k++) {
      const s = starts[b0 + k];
      const seg = padded.subarray(s, Math.min(padded.length, s + BP.nSamples));
      x.set(seg, k * BP.nSamples);
    }
    const res = await session.run({ [inName]: new ort.Tensor('float32', x, [nb, BP.nSamples, 1]) });
    const outs = [[res[nameNote].data, note, BP.nNotes], [res[nameOnset].data, onset, BP.nNotes], [res[nameContour].data, contour, BP.nContour]];
    for (let k = 0; k < nb; k++) {
      const w = b0 + k;
      for (let lf = N_OLAP; lf < BP.nFrames - N_OLAP; lf++) {
        const f = w * FRAMES_PER_WINDOW + (lf - N_OLAP);
        if (f >= nT) break;
        for (const [src, dst, nb2] of outs) {
          const so = (k * BP.nFrames + lf) * nb2;
          dst.set(src.subarray(so, so + nb2), f * nb2);
        }
      }
    }
    for (const key of [nameNote, nameOnset, nameContour]) if (res[key].dispose) res[key].dispose();
    if (opts.onProgress) opts.onProgress(Math.min(1, (b0 + nb) / starts.length));
    await new Promise((r) => setTimeout(r, 0));
  }
  return { nT, note, onset, contour };
}

/** Port of get_infered_onsets */
function inferOnsets(onsets, frames, nT, nF, nDiff = 2) {
  const diff = new Float32Array(nT * nF);
  let maxDiff = 0, maxOnset = 0;
  for (let i = 0; i < nT * nF; i++) if (onsets[i] > maxOnset) maxOnset = onsets[i];
  for (let t = 0; t < nT; t++) {
    for (let f = 0; f < nF; f++) {
      let mn = Infinity;
      for (let n = 1; n <= nDiff; n++) {
        const prev = t - n >= 0 ? frames[(t - n) * nF + f] : 0;
        const d = frames[t * nF + f] - prev;
        if (d < mn) mn = d;
      }
      if (mn < 0 || t < nDiff) mn = 0;
      diff[t * nF + f] = mn;
      if (mn > maxDiff) maxDiff = mn;
    }
  }
  const out = new Float32Array(nT * nF);
  for (let i = 0; i < nT * nF; i++) {
    const fd = maxDiff > 0 ? (maxOnset * diff[i]) / maxDiff : 0;
    out[i] = Math.max(onsets[i], fd);
  }
  return out;
}

/**
 * Port of output_to_notes_polyphonic. Returns [{s, e, pitch, amp}] in frames.
 */
export function outputToNotes(model, opts = {}) {
  const onsetThresh = opts.onsetThresh ?? 0.5;
  const frameThresh = opts.frameThresh ?? 0.3;
  const minNoteLen = opts.minNoteLen ?? 11;
  const energyTol = opts.energyTol ?? 11;
  const inferOn = opts.inferOnsets ?? true;
  const melodia = opts.melodiaTrick ?? true;
  const minMidi = opts.minMidi ?? 0, maxMidi = opts.maxMidi ?? 200;
  const { nT } = model;
  const nF = BP.nNotes;
  const frames = Float32Array.from(model.note);
  let onsets = Float32Array.from(model.onset);
  // constrain frequency
  const minIdx = Math.max(0, Math.round(minMidi - BP.midiOffset));
  const maxIdx = Math.min(nF, Math.round(maxMidi - BP.midiOffset));
  for (let t = 0; t < nT; t++) {
    for (let f = 0; f < nF; f++) {
      if (f < minIdx || f >= maxIdx) { frames[t * nF + f] = 0; onsets[t * nF + f] = 0; }
    }
  }
  if (inferOn) onsets = inferOnsets(onsets, frames, nT, nF);
  // peaks along time (strict local maxima, edges excluded) >= threshold
  const cand = [];
  for (let t = 1; t < nT - 1; t++) {
    for (let f = 0; f < nF; f++) {
      const v = onsets[t * nF + f];
      if (v >= onsetThresh && v > onsets[(t - 1) * nF + f] && v > onsets[(t + 1) * nF + f]) cand.push(t * nF + f);
    }
  }
  const remaining = Float32Array.from(frames);
  const notes = [];
  for (let ci = cand.length - 1; ci >= 0; ci--) {
    const start = Math.floor(cand[ci] / nF), f = cand[ci] % nF;
    if (start >= nT - 1) continue;
    let i = start + 1, k = 0;
    while (i < nT - 1 && k < energyTol) {
      if (remaining[i * nF + f] < frameThresh) k++; else k = 0;
      i++;
    }
    i -= k;
    if (i - start <= minNoteLen) continue;
    for (let t = start; t < i; t++) {
      remaining[t * nF + f] = 0;
      if (f < 87) remaining[t * nF + f + 1] = 0;
      if (f > 0) remaining[t * nF + f - 1] = 0;
    }
    let amp = 0;
    for (let t = start; t < i; t++) amp += frames[t * nF + f];
    amp /= i - start;
    notes.push({ s: start, e: i, pitch: f + BP.midiOffset, amp, onsetStrength: onsets[start * nF + f] });
  }
  if (melodia) {
    // argmax loop, implemented with a sorted candidate list (values only ever get zeroed)
    const cells = [];
    for (let idx = 0; idx < nT * nF; idx++) if (remaining[idx] > frameThresh) cells.push(idx);
    cells.sort((a, b) => (remaining[b] - remaining[a]) || (a - b));
    const orig = new Float32Array(cells.length);
    for (let c = 0; c < cells.length; c++) orig[c] = remaining[cells[c]];
    for (let c = 0; c < cells.length; c++) {
      const idx = cells[c];
      if (remaining[idx] !== orig[c] || remaining[idx] <= frameThresh) continue;
      const iMid = Math.floor(idx / nF), f = idx % nF;
      remaining[idx] = 0;
      let i = iMid + 1, k = 0;
      while (i < nT - 1 && k < energyTol) {
        if (remaining[i * nF + f] < frameThresh) k++; else k = 0;
        remaining[i * nF + f] = 0;
        if (f < 87) remaining[i * nF + f + 1] = 0;
        if (f > 0) remaining[i * nF + f - 1] = 0;
        i++;
      }
      const iEnd = i - 1 - k;
      i = iMid - 1; k = 0;
      while (i > 0 && k < energyTol) {
        if (remaining[i * nF + f] < frameThresh) k++; else k = 0;
        remaining[i * nF + f] = 0;
        if (f < 87) remaining[i * nF + f + 1] = 0;
        if (f > 0) remaining[i * nF + f - 1] = 0;
        i--;
      }
      const iStart = i + 1 + k;
      if (iEnd - iStart <= minNoteLen) continue;
      let amp = 0;
      for (let t = iStart; t < iEnd; t++) amp += frames[t * nF + f];
      amp /= iEnd - iStart;
      notes.push({ s: iStart, e: iEnd, pitch: f + BP.midiOffset, amp, onsetStrength: 0, melodia: true });
    }
  }
  return notes;
}

let GAUSS = null;
/** Port of get_pitch_bends: per-note list of bends in 1/3 semitone units (one per frame). */
export function addPitchBends(model, notes, nBinsTol = 25) {
  const nC = BP.nContour;
  const winLen = nBinsTol * 2 + 1;
  if (!GAUSS) {
    GAUSS = new Float64Array(winLen);
    for (let i = 0; i < winLen; i++) { const x = i - (winLen - 1) / 2; GAUSS[i] = Math.exp(-0.5 * (x / 5) ** 2); }
  }
  for (const n of notes) {
    const fi = 3 * (n.pitch - BP.midiOffset);
    const f0 = Math.max(fi - nBinsTol, 0);
    const f1 = Math.min(nC, fi + nBinsTol + 1);
    const g0 = Math.max(0, nBinsTol - fi);
    const pbShift = nBinsTol - Math.max(0, nBinsTol - fi);
    const bends = [];
    for (let t = n.s; t < n.e; t++) {
      let best = -1, bi = 0;
      for (let f = f0; f < f1; f++) {
        const v = model.contour[t * nC + f] * GAUSS[g0 + (f - f0)];
        if (v > best) { best = v; bi = f - f0; }
      }
      bends.push(bi - pbShift);
    }
    n.bends = bends;
  }
  return notes;
}

/** Convert frame-based notes to seconds. */
export function notesToSeconds(notes) {
  return notes.map((n) => ({ ...n, t0: frameTime(n.s), t1: frameTime(n.e) }));
}
