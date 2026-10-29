// How each found note began (engine/techniques.js attachOnsetEvidence), checked on recordings
// whose notes come with their strings (EGDB, GuitarSet): a hammer-on, pull-off or slide stays on
// one string, so a move between two notes that the annotation puts on different strings and that
// the classifier calls legato or slide is certainly wrong. Reports, over moves whose two notes
// match annotated notes: on different strings, how many would be marked (wrong for sure); on the
// same string, how many (the truth isn't annotated: picked or not).
// usage: node tests/eval_onsets.mjs <annotations.json> <audio dir> [--suffix _mic] [--every 1] [--model onset.json]
// env: MODELS = dir with basic_pitch.onnx, hrgt.chain, hrgt.weights.bin; ORT
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { detectNotes, DEFAULT_OPTS } from '../engine/pipeline.js';
import { cleanNotes } from '../engine/notes.js';
import { estimateTuningCents } from '../engine/pitchref.js';
import { parseChain, loadParts } from '../engine/demucs.js';
import { hrModel } from '../engine/hrgt.js';
import { attachOnsetEvidence, onsetNet, defaultOnsetNet, MARK_AT } from '../engine/techniques.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const [annPath, audioDir] = args;
const suffix = opt('--suffix', ''), every = Number(opt('--every', 1));
const net = opt('--model') ? onsetNet(JSON.parse(fs.readFileSync(opt('--model'), 'utf8'))) : defaultOnsetNet();
if (!net) throw new Error('no onset model: pass --model');
const M = process.env.MODELS;
const ortMod = await import(process.env.ORT || 'onnxruntime-node');
const ort = ortMod.default || ortMod;
let bp = null, hr = null;
const env = {
  ort,
  getBasicPitch: async () => bp || (bp = await ort.InferenceSession.create(path.join(M, 'basic_pitch.onnx'))),
  getTranscriber: async () => hr || (hr = hrModel(await loadParts(ort, parseChain(fs.readFileSync(path.join(M, 'hrgt.chain'))), new Uint8Array(fs.readFileSync(path.join(M, 'hrgt.weights.bin'))), { intraOpNumThreads: 4 }))),
};
const ann = JSON.parse(fs.readFileSync(annPath, 'utf8'));
const names = Object.keys(ann).sort().filter((_, i) => i % every === 0);
const stats = { cross: [], same: [] };
for (const [k, name] of names.entries()) {
  const w = decodeWav(fs.readFileSync(path.join(audioDir, `${name}${suffix}.wav`)));
  let x = w.channels.length > 1 ? Float32Array.from(w.channels[0], (v, i) => 0.5 * (v + w.channels[1][i])) : w.channels[0];
  if (w.sr !== 44100) x = resample(x, w.sr, 44100);
  const ref = estimateTuningCents(x, 44100);
  const cents = ref.confidence > 0.15 ? ref.cents : 0;
  const found = cleanNotes(await detectNotes(env, x, { ...DEFAULT_OPTS, hrOnly: true }, cents, () => {}), { idPrefix: 'n:', mergeFragments: false, mergeRetrigger: false });
  const notes = attachOnsetEvidence(found, x, 44100, cents, net);
  // each found note's annotated twin: same pitch, starts within 50 ms
  const truth = ann[name].notes.map(([t, d, v, s]) => ({ t0: t, pitch: Math.round(v), string: s }));
  const twin = new Map();
  for (const n of notes) {
    let best = null;
    for (const r of truth) if (r.pitch === n.pitch && Math.abs(r.t0 - n.t0) <= 0.05 && (!best || Math.abs(r.t0 - n.t0) < Math.abs(best.t0 - n.t0))) best = r;
    if (best) twin.set(n.id, best);
  }
  for (const b of notes) {
    if (!b.how) continue;
    const ra = twin.get(b.from), rb = twin.get(b.id);
    if (!ra || !rb) continue;
    stats[ra.string === rb.string ? 'same' : 'cross'].push(b.how);
  }
  process.stderr.write(`\r${k + 1}/${names.length} ${name}      `);
}
process.stderr.write('\n');
const pct = (a, b) => (b ? ((100 * a) / b).toFixed(1) : '-');
for (const [k, list] of Object.entries(stats)) {
  const leg = list.filter((h) => h.legato >= MARK_AT.legato && h.legato >= h.slide).length;
  const sl = list.filter((h) => h.slide >= MARK_AT.slide && h.slide > h.legato).length;
  console.log(`${k === 'cross' ? 'different strings (marks are wrong)' : 'same string (truth unknown)'}: ${list.length} moves; legato marks ${pct(leg, list.length)}%, slide marks ${pct(sl, list.length)}%`);
}
