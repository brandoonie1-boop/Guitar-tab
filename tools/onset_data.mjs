// Training data for telling how each note began (engine/techfeat.js onsetFeatures): picked, a
// hammer-on or pull-off, a slide, or the rest of a bent or shaken note that note detection split.
// EG-IPT (one guitarist, a Gibson SG; each file one gesture), notes found by the engine itself:
// the move into the second note of a hammer-on, pull-off or trill is legato, of a slide a slide,
// any split inside a bend or vibrato a bend. Picked moves are made from two plain notes on one
// string (the first cut where the second is picked: --joins per pickup and source) and on two
// strings (the first rings on: --mixes).
// Writes {features, rows: [[file, pickup, source, label, kind, ...features]]}.
// usage: node tools/onset_data.mjs <EG-IPT dir (44.1 kHz)> <out.json> [--pickups HB-neck,HB-couple] [--sources DI,dyn]
//        [--per 240] [--joins 150] [--mixes 100] [--shard 0/1]
// (--per: at most this many files of each kind; slides, bends and vibrato get half and a sixth of
// that, takes cut to 1.6 s)
// env: MODELS = dir with basic_pitch.onnx, hrgt.chain, hrgt.weights.bin; ORT
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { detectNotes, DEFAULT_OPTS } from '../engine/pipeline.js';
import { cleanNotes } from '../engine/notes.js';
import { estimateTuningCents } from '../engine/pitchref.js';
import { parseChain, loadParts } from '../engine/demucs.js';
import { hrModel } from '../engine/hrgt.js';
import { onsetFeatures, onsetSources, ONSET_FEATURES } from '../engine/techfeat.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const [dir, out] = args;
const pickups = opt('--pickups', 'HB-neck,HB-couple').split(',');
const sources = opt('--sources', 'DI,dyn').split(',');
const per = Number(opt('--per', 240)), nJoins = Number(opt('--joins', 150)), nMixes = Number(opt('--mixes', 100));
const [shard, nShards] = opt('--shard', '0/1').split('/').map(Number);
const M = process.env.MODELS;
const ortMod = await import(process.env.ORT || 'onnxruntime-node');
const ort = ortMod.default || ortMod;
let bp = null, hr = null;
const env = {
  ort,
  getBasicPitch: async () => bp || (bp = await ort.InferenceSession.create(path.join(M, 'basic_pitch.onnx'))),
  getTranscriber: async () => hr || (hr = hrModel(await loadParts(ort, parseChain(fs.readFileSync(path.join(M, 'hrgt.chain'))), new Uint8Array(fs.readFileSync(path.join(M, 'hrgt.weights.bin'))), { intraOpNumThreads: 2 }))),
};

const LABEL = (v) => (/^(hammer-on|pull-off|trill)/.test(v) ? 'legato' : /^slide/.test(v) ? 'slide' : /^(bend|vibrato)/.test(v) ? 'bend' : /^(ordinario|staccato)$/.test(v) ? 'plain' : null);
// the same pseudo-random sequence for every run
let seed = 12345 + shard;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);

// the notes of many short files at once (the note models work in chunks of seconds): the files one
// after another with a quarter second of silence between, cut apart again afterwards
async function notesOfAll(xs) {
  const gap = 11025, starts = [];
  let len = 0;
  for (const x of xs) { starts.push(len); len += x.length + gap; }
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
}
const BATCH = 24;

const rows = [];
let nFiles = 0;
for (const pu of pickups) {
  for (const src of sources) {
    const base = path.join(dir, pu, src);
    if (!fs.existsSync(base)) continue;
    const plain = []; // single plain notes: what joins and mixes are made of
    const files = [];
    for (const folder of fs.readdirSync(base).sort()) {
      const byVariant = new Map();
      for (const f of fs.readdirSync(path.join(base, folder)).sort()) {
        const m = f.match(/^[^_]+_([^_]+)_.*?(?:_(\d)s)?_[^_]+\.wav$/);
        if (!m || !LABEL(m[1])) continue;
        if (!byVariant.has(m[1])) byVariant.set(m[1], []);
        byVariant.get(m[1]).push({ file: path.join(base, folder, f), variant: m[1], string: m[2] ? Number(m[2]) : null });
      }
      for (const [v, list] of byVariant) {
        const cap = /^slide/.test(v) ? per / 2 : /^(bend|vibrato)/.test(v) ? per / 6 : per;
        const step = Math.max(1, list.length / cap);
        for (let i = 0; i < Math.min(cap, list.length); i++) files.push(list[Math.floor(i * step)]);
      }
    }
    // split between shards (each makes its joins and mixes from its own plain notes)
    const mine = files.filter((f, fi) => fi % nShards === shard);
    for (let k0 = 0; k0 < mine.length; k0 += BATCH) {
     const batch = mine.slice(k0, k0 + BATCH).map((f) => ({ ...f, x: decodeWav(fs.readFileSync(f.file)).channels[0].subarray(0, Math.round(1.6 * 44100)) }));
     const found = await notesOfAll(batch.map((f) => f.x));
     for (const [bi, f] of batch.entries()) {
      const label = LABEL(f.variant), x = f.x, notes = found[bi];
      nFiles++;
      if (label === 'plain') {
        if (notes.length === 1 && f.string) plain.push({ ...f, x, note: notes[0] });
        continue;
      }
      const ps = new Set(notes.map((n) => n.pitch));
      if (label === 'legato' && !/trill/.test(f.variant)) {
        if (notes.length !== 2 || Math.sign(notes[1].pitch - notes[0].pitch) !== (f.variant === 'hammer-on' ? 1 : -1)) continue;
      }
      if (/trill/.test(f.variant) && ps.size !== 2) continue;
      if (label === 'slide' && (notes.length !== 2 || Math.abs(notes[1].pitch - notes[0].pitch) !== (/half/.test(f.variant) ? 1 : 2))) continue;
      const srcOf = onsetSources(notes);
      for (const b of notes) {
        const a = srcOf.get(b);
        if (!a) continue;
        const others = notes.filter((o) => o !== a && o !== b && o.t0 < b.t0 + 0.03 && o.t1 > b.t0);
        rows.push([path.relative(dir, f.file), pu, src, label, 'real', ...onsetFeatures(x, 44100, a, b, others, notes.cents)]);
      }
     }
     process.stderr.write(`\r${pu} ${src} ${nFiles} files, ${rows.length} moves   `);
    }
    // picked moves made from two plain notes of this pickup and source
    const attack = (x) => {
      let pk = 0;
      for (let i = 0; i < Math.min(x.length, 4410); i++) pk = Math.max(pk, Math.abs(x[i]));
      for (let i = 0; i < x.length; i++) if (Math.abs(x[i]) > 0.1 * pk) return Math.max(0, i - 44);
      return 0;
    };
    const pairs = (same) => {
      const out = [];
      for (const A of plain) for (const B of plain) {
        const d = Math.abs(A.note.pitch - B.note.pitch);
        if (A !== B && d >= 1 && d <= 7 && (A.string === B.string) === same) out.push([A, B]);
      }
      return out;
    };
    const made = [];
    for (const [kind, n] of [['join', nJoins], ['mix', nMixes]]) {
      const all = pairs(kind === 'join');
      for (let k = 0; k < Math.round(n / nShards) && all.length; k++) {
        const [A, B] = all[Math.floor(rand() * all.length)];
        const r1 = rand(), r2 = rand();
        const tCut = A.note.t0 + 0.15 + 0.45 * r1;
        const iCut = Math.round(tCut * 44100), bStart = attack(B.x), fade = 88;
        const gain = Math.pow(10, (-6 + 9 * r2) / 20);
        const len = kind === 'join' ? iCut + (B.x.length - bStart) : Math.max(A.x.length, iCut + B.x.length - bStart);
        const y = new Float32Array(len);
        for (let i = 0; i < Math.min(len, A.x.length); i++) {
          if (kind === 'join' && i >= iCut + fade) break;
          const w = kind === 'join' && i >= iCut ? 1 - (i - iCut) / fade : 1;
          y[i] = A.x[i] * w;
        }
        for (let i = bStart; i < B.x.length && iCut + i - bStart < len; i++) y[iCut + i - bStart] += gain * B.x[i];
        made.push({ kind, A, B, tCut, y });
      }
    }
    for (let k0 = 0; k0 < made.length; k0 += BATCH) {
      const batch = made.slice(k0, k0 + BATCH);
      const found = await notesOfAll(batch.map((m) => m.y));
      for (const [bi, { kind, A, B, tCut, y }] of batch.entries()) {
        const notes = found[bi];
        const b = notes.find((n) => n.pitch === B.note.pitch && Math.abs(n.t0 - tCut) < 0.06);
        const a = b && onsetSources(notes).get(b);
        if (!a || a.pitch !== A.note.pitch) continue;
        const others = notes.filter((o) => o !== a && o !== b && o.t0 < b.t0 + 0.03 && o.t1 > b.t0);
        rows.push([`${path.relative(dir, A.file)}+${path.basename(B.file)}`, pu, src, 'pick', kind, ...onsetFeatures(y, 44100, a, b, others, notes.cents)]);
      }
    }
    process.stderr.write(`\r${pu} ${src}: ${nFiles} files, ${rows.length} moves          \n`);
  }
}
fs.writeFileSync(out, JSON.stringify({ features: ONSET_FEATURES, rows }));
const count = {};
for (const r of rows) count[`${r[3]}/${r[4]}`] = (count[`${r[3]}/${r[4]}`] || 0) + 1;
console.error(out, rows.length, 'moves', JSON.stringify(count));
