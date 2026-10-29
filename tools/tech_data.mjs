// Training data for the technique classifier (engine/techniques.js): Guitar-TECHS's technique
// takes (every note of a take played with that technique) and plain scales and chords; each note
// the note model finds there (engine/pipeline.js detectNotes), its engine/techfeat.js features,
// labelled with its take's technique. Writes <out>.json: {features, rows: [[take, player, kind, label, ...features]]}.
//   GuitarSet (acoustic; labels from tools/gs_techniques.py, read off its per-string pitch contours):
//   the annotated notes themselves (no note detection), plain / bend / vibrato, and every move to
//   the next note on the same string, slide / step, in <out>.trans.json.
//   EG-IPT (one guitarist, a Gibson SG; each file one gesture; notes found by the engine): the
//   gesture's note labelled by its file - bend (up, or released), vibrato, natural / artificial
//   harmonic, palm mute ("muted"), plain (ordinario, staccato); a bend or vibrato the detection split
//   is joined back into one note first (what engine/techniques.js does with a split it hears).
//   Also writes <out>.notes.json, {file: {notes: [[t0, dur, pitch, string]]}} (string 0 = low E,
//   from the file name), the plain, muted and harmonic notes, for the string classifier
//   (tools/string_data.mjs egset).
// usage: node tools/tech_data.mjs <Guitar-TECHS dir> <out> [--kinds directinput,micamp] [--seconds 150]
//        node tools/tech_data.mjs gs <gs_tech.json> <audio dir> <out> [--every 3 --offset 1]
//        node tools/tech_data.mjs egipt <EG-IPT dir (44.1 kHz)> <out> [--pickups HB-neck,HB-couple] [--sources DI,dyn] [--per 150]
// env: MODELS = dir with basic_pitch.onnx, beat_this_small.f16.onnx, hrgt.chain, hrgt.weights.bin; ORT
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { detectNotes, DEFAULT_OPTS } from '../engine/pipeline.js';
import { cleanNotes } from '../engine/notes.js';
import { estimateTuningCents } from '../engine/pitchref.js';
import { parseChain, loadParts } from '../engine/demucs.js';
import { hrModel } from '../engine/hrgt.js';
import { techFeatures, TECH_FEATURES, transFeatures, TRANS_FEATURES } from '../engine/techfeat.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
if (args[0] === 'gs') {
  const [, labelsPath, audioDir, out] = args;
  const every = Number(opt('--every', 1)), offset = Number(opt('--offset', 0));
  const labels = JSON.parse(fs.readFileSync(labelsPath, 'utf8'));
  const rows = [], trans = [];
  const names = Object.keys(labels).sort().filter((_, i) => i % every === offset);
  for (const [ti, name] of names.entries()) {
    const w = decodeWav(fs.readFileSync(path.join(audioDir, `${name}_mic.wav`)));
    let x = w.channels[0];
    if (w.sr !== 44100) x = resample(x, w.sr, 44100);
    const notes = labels[name].map(([t0, d, m, s, mk]) => ({ t0, t1: t0 + d, pitch: Math.round(m), string: s, mk }));
    for (const n of notes) {
      const others = notes.filter((o) => o !== n && o.t0 < n.t1 && o.t1 > n.t0);
      const label = n.mk.bend ? 'bend' : n.mk.vibrato ? 'vibrato' : 'plain';
      rows.push([name, name.slice(0, 2), 'mic', label, ...techFeatures(x, 44100, n, others)]);
    }
    const bySt = new Map();
    for (const n of notes) { if (!bySt.has(n.string)) bySt.set(n.string, []); bySt.get(n.string).push(n); }
    for (const list of bySt.values()) {
      list.sort((p, q) => p.t0 - q.t0);
      for (let i = 1; i < list.length; i++) {
        const a = list[i - 1], b = list[i];
        if (b.t0 - a.t1 > 0.03 || a.pitch === b.pitch) continue;
        const others = notes.filter((o) => o !== a && o !== b && o.t0 < b.t1 && o.t1 > a.t0);
        trans.push([name, name.slice(0, 2), 'mic', b.mk.slideFrom ? 'slide' : 'step', ...transFeatures(x, 44100, a, b, others)]);
      }
    }
    process.stderr.write(`\r${ti + 1}/${names.length} ${name}      `);
  }
  fs.writeFileSync(out + '.json', JSON.stringify({ features: TECH_FEATURES, rows }));
  fs.writeFileSync(out + '.trans.json', JSON.stringify({ features: TRANS_FEATURES, rows: trans }));
  console.error(`\n${out}: ${rows.length} notes, ${trans.length} moves`);
  process.exit(0);
}
if (args[0] === 'egipt') {
  const [, dir, out] = args;
  const pickups = opt('--pickups', 'HB-neck,HB-couple').split(','), sources = opt('--sources', 'DI,dyn').split(',');
  const per = Number(opt('--per', 150));
  const ortMod = await import(process.env.ORT || 'onnxruntime-node');
  const ort = ortMod.default || ortMod;
  let bp = null, hr = null;
  const M = process.env.MODELS;
  const env = {
    ort,
    getBasicPitch: async () => bp || (bp = await ort.InferenceSession.create(path.join(M, 'basic_pitch.onnx'))),
    getTranscriber: async () => hr || (hr = hrModel(await loadParts(ort, parseChain(fs.readFileSync(path.join(M, 'hrgt.chain'))), new Uint8Array(fs.readFileSync(path.join(M, 'hrgt.weights.bin'))), { intraOpNumThreads: 2 }))),
  };
  const LABEL = (v) => (/^bend/.test(v) ? 'bend' : /^vibrato/.test(v) ? 'vibrato' : /harmonic$/.test(v) ? 'harmonic' : v === 'muted' ? 'palmMute' : /^(ordinario|staccato)$/.test(v) ? 'plain' : null);
  // the notes of many short files at once, one after another with a second of silence between
  const notesOfAll = async (xs) => {
    const starts = [];
    let len = 0;
    for (const x of xs) { starts.push(len); len += x.length + 44100; }
    const y = new Float32Array(len);
    xs.forEach((x, i) => y.set(x, starts[i]));
    const ref = estimateTuningCents(y, 44100);
    const cents = ref.confidence > 0.15 ? ref.cents : 0;
    const all = cleanNotes(await detectNotes(env, y, { ...DEFAULT_OPTS, hrOnly: true }, cents, () => {}), { mergeFragments: false, mergeRetrigger: false });
    return xs.map((x, i) => {
      const t0 = starts[i] / 44100, t1 = (starts[i] + x.length) / 44100;
      const notes = all.filter((n) => n.t0 >= t0 - 0.02 && n.t0 < t1).map((n) => ({ ...n, t0: n.t0 - t0, t1: Math.min(n.t1, t1 + 0.05) - t0 })).sort((p, q) => p.t0 - q.t0);
      notes.cents = cents;
      return notes;
    });
  };
  const rows = [], strings = {};
  const OPEN = [40, 45, 50, 55, 59, 64];
  for (const pu of pickups) {
    for (const src of sources) {
      const base = path.join(dir, pu, src);
      if (!fs.existsSync(base)) continue;
      for (const folder of fs.readdirSync(base).sort()) {
        const byVariant = new Map();
        for (const f of fs.readdirSync(path.join(base, folder)).sort()) {
          const m = f.match(/^[^_]+_([^_]+)_/);
          if (!m || !LABEL(m[1])) continue;
          if (!byVariant.has(m[1])) byVariant.set(m[1], []);
          byVariant.get(m[1]).push(f);
        }
        const todo = [];
        for (const [variant, list] of byVariant) {
          const step = Math.max(1, list.length / per);
          for (let i = 0; i < Math.min(per, list.length); i++) todo.push({ variant, f: list[Math.floor(i * step)] });
        }
        for (let k0 = 0; k0 < todo.length; k0 += 24) {
          const batch = todo.slice(k0, k0 + 24).map((t) => ({ ...t, x: decodeWav(fs.readFileSync(path.join(base, folder, t.f))).channels[0] }));
          const found = await notesOfAll(batch.map((t) => t.x));
          for (const [bi, { variant, f, x }] of batch.entries()) {
            const notes = found[bi], cents = notes.cents;
            if (!notes.length) continue;
            const label = LABEL(variant);
            // the gesture's note: for a bend or vibrato everything from the first note on (split
            // or not), otherwise the longest note
            const n = label === 'bend' || label === 'vibrato'
              ? { t0: notes[0].t0, t1: Math.max(...notes.map((q) => q.t1)), pitch: notes[0].pitch }
              : notes.reduce((p, q) => (q.t1 - q.t0 > p.t1 - p.t0 ? q : p));
            const others = label === 'bend' || label === 'vibrato' ? [] : notes.filter((o) => o !== n && o.t0 < n.t1 && o.t1 > n.t0);
            rows.push([`${pu}/${src}/${folder}/${f}`, pu, `${src}:${variant}`, label, ...techFeatures(x, 44100, n, others, undefined, cents)]);
            // the string it was played on (a note the detection got right: on that string, frets 0-24)
            const sm = f.match(/_(\d)s_[^_]+\.wav$/);
            const st = sm ? 6 - Number(sm[1]) : null;
            if (st != null && label !== 'bend' && label !== 'vibrato' && notes.length === 1 && n.pitch - OPEN[st] >= 0 && n.pitch - OPEN[st] <= 24) {
              strings[`${pu}/${src}/${folder}/${f.replace(/\.wav$/, '')}`] = { notes: [[n.t0, n.t1 - n.t0, n.pitch, st]] };
            }
          }
        }
        process.stderr.write(`\r${pu} ${src} ${folder}: ${rows.length} notes      `);
      }
    }
  }
  fs.writeFileSync(out + '.json', JSON.stringify({ features: TECH_FEATURES, rows }));
  fs.writeFileSync(out + '.notes.json', JSON.stringify(strings));
  console.error(`\n${out}.json: ${rows.length} notes; ${Object.keys(strings).length} with their string`);
  process.exit(0);
}
const [dir, out] = args;
const kinds = opt('--kinds', 'directinput').split(',');
const seconds = Number(opt('--seconds', 150));
const M = process.env.MODELS;
const ortMod = await import(process.env.ORT || 'onnxruntime-node');
const ort = ortMod.default || ortMod;
let bp = null, bm = null, hr = null;
const env = {
  ort,
  getBasicPitch: async () => bp || (bp = await ort.InferenceSession.create(path.join(M, 'basic_pitch.onnx'))),
  getBeatModel: async () => bm || (bm = await ort.InferenceSession.create(path.join(M, 'beat_this_small.f16.onnx'), { intraOpNumThreads: 2 })),
  getTranscriber: async () => hr || (hr = hrModel(await loadParts(ort, parseChain(fs.readFileSync(path.join(M, 'hrgt.chain'))), new Uint8Array(fs.readFileSync(path.join(M, 'hrgt.weights.bin'))), { intraOpNumThreads: 4 }))),
};
const TAKES = [
  ['techniques', 'Bendings', 'bend'], ['techniques', 'Vibrato', 'vibrato'], ['techniques', 'PalmMute', 'palmMute'],
  ['techniques', 'Harmonics', 'harmonic'], ['techniques', 'PinchHarmonics', 'pinch'],
  ['scales', 'A', 'plain'], ['scales', 'C', 'plain'], ['scales', 'E', 'plain'], ['scales', 'G', 'plain'],
  ['chords', 'Drop3_Maj7', 'plain'], ['chords', 'Set1_7', 'plain'], ['chords', 'Drop3_m7', 'plain'],
];
const rows = [];
for (const p of ['P1', 'P2']) {
  for (const kind of kinds) {
    for (const [set, key, label] of TAKES) {
      const file = path.join(dir, `${p}_${set}`, 'audio', kind, `${kind}_${key}.wav`);
      if (!fs.existsSync(file)) continue;
      const w = decodeWav(fs.readFileSync(file));
      let x = w.channels.length > 1 ? Float32Array.from(w.channels[0], (v, i) => 0.5 * (v + w.channels[1][i])) : w.channels[0];
      if (w.sr !== 44100) x = resample(x, w.sr, 44100);
      x = x.subarray(0, Math.min(x.length, seconds * 44100));
      // the notes as the app finds them (the note model's; tuning offset measured first)
      const ref = estimateTuningCents(x, 44100);
      const cents = ref.confidence > 0.15 ? ref.cents : 0;
      const live = cleanNotes(await detectNotes(env, x, { ...DEFAULT_OPTS, hrOnly: true }, cents, () => {}), { mergeFragments: false, mergeRetrigger: false });
      let n0 = rows.length;
      for (const n of live) {
        const others = live.filter((o) => o !== n && o.t0 < n.t1 && o.t1 > n.t0);
        rows.push([`${set}/${key}`, p, kind, label, ...techFeatures(x, 44100, n, others, undefined, cents)]);
      }
      console.error(`${p} ${kind} ${set}/${key}: ${rows.length - n0} notes`);
    }
  }
}
fs.writeFileSync(out + '.json', JSON.stringify({ features: TECH_FEATURES, rows }));
console.error(`${out}.json: ${rows.length} notes`);
