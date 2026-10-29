// Playing techniques on real recordings: EG-IPT (Fiorini, Brochec, Borg, Pasini, NIME 2025;
// zenodo.org/records/15205644): one professional guitarist, a Gibson SG, every file one gesture
// played with one technique - a picked note and then a hammer-on, a pull-off or a slide; a bend;
// vibrato; a trill; a plain note; tremolo picking. The file names give the string. Runs the whole
// engine (isolation off, standard tuning) on each file and checks the tab against the gesture:
// how many notes, on the string it was played on, and the technique marks.
// Files resampled to 44.1 kHz in <dir>/<pickup>/<source>/<technique>/ (EG-IPT's own layout).
// usage: node tests/eval_egipt.mjs <dir> [--pickups HB-bridge] [--sources DI,dyn] [--per 30] [--only hammer-on,pull-off] [--json out.json]
// env: MODELS = dir with basic_pitch.onnx, beat_this_small.f16.onnx, hrgt.chain, hrgt.weights.bin; ORT
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { transcribe } from '../engine/pipeline.js';
import { parseChain, loadParts } from '../engine/demucs.js';
import { hrModel } from '../engine/hrgt.js';
import { STANDARD } from '../engine/tunings.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const dir = args[0];
const pickups = opt('--pickups', 'HB-bridge').split(',');
const sources = opt('--sources', 'DI').split(',');
const per = Number(opt('--per', 30));
const only = opt('--only') ? opt('--only').split(',') : null;
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

// what the tab of each gesture should show. notes: how many; move: +1 up / -1 down / 0 either
// (two-note gestures); mark: on the first note of a pair (the app's h / p / slide go there), or
// on the single note
const EXPECT = {
  'hammer-on': { notes: 2, move: 1, mark: 'legato' },
  'pull-off': { notes: 2, move: -1, mark: 'legato' },
  'slide-halftone': { notes: 2, move: 0, mark: 'slide' },
  'slide-wholetone': { notes: 2, move: 0, mark: 'slide' },
  'bend-up-halftone': { notes: 1, mark: 'bend', amount: 1 },
  'bend-up-wholetone': { notes: 1, mark: 'bend', amount: 2 },
  'bend-up-1-5-tone': { notes: 1, mark: 'bend', amount: 3 },
  'bend-down-halftone': { notes: 1, mark: 'bend' }, // played bent, released: still one note
  'bend-down-wholetone': { notes: 1, mark: 'bend' },
  'vibrato-fast': { notes: 1, mark: 'vibrato' },
  'vibrato-slow': { notes: 1, mark: 'vibrato' },
  'trill-major': { notes: 'trill', mark: 'legato' },
  'trill-minor': { notes: 'trill', mark: 'legato' },
  ordinario: { notes: 1 },
  staccato: { notes: 1 },
  tremolo: { notes: 'repeat' },
  'art-harmonic': { notes: 1, mark: 'harmonic' },
  'nat-harmonic': { notes: 1, mark: 'harmonic' },
  muted: { mark: 'palmMute' },
};
const MARKS = ['legato', 'slide', 'bend', 'vibrato', 'harmonic', 'palmMute'];

const files = [];
for (const pu of pickups) {
  for (const src of sources) {
    const base = path.join(dir, pu, src);
    if (!fs.existsSync(base)) continue;
    for (const folder of fs.readdirSync(base).sort()) {
      const byVariant = new Map();
      for (const f of fs.readdirSync(path.join(base, folder)).sort()) {
        const m = f.match(/^[^_]+_([^_]+)_.*?(?:_(\d)s)?_[^_]+\.wav$/);
        if (!m || !EXPECT[m[1]] || (only && !only.includes(m[1]) && !only.includes(folder))) continue;
        if (!byVariant.has(m[1])) byVariant.set(m[1], []);
        byVariant.get(m[1]).push({ file: path.join(base, folder, f), variant: m[1], string: m[2] ? 6 - Number(m[2]) : null, pickup: pu, source: src });
      }
      for (const list of byVariant.values()) {
        const step = Math.max(1, list.length / per);
        for (let i = 0; i < Math.min(per, list.length); i++) files.push(list[Math.floor(i * step)]);
      }
    }
  }
}
console.error(`${files.length} files`);

function judge(f, notes, tech) {
  const e = EXPECT[f.variant];
  const marks = notes.map((n) => {
    const t = tech.get(n.id) || {};
    return Object.fromEntries(MARKS.map((k) => [k, t[k] || n[k] || null]).filter(([, v]) => v));
  });
  const has = (i, k) => !!(marks[i] && marks[i][k]);
  const pitches = notes.map((n) => n.pitch);
  let notesOk;
  if (e.notes === 'trill') notesOk = notes.length >= 3 && new Set(pitches).size === 2 && pitches.every((p, i) => i === 0 || p !== pitches[i - 1]);
  else if (e.notes === 'repeat') notesOk = notes.length >= 2 && new Set(pitches).size === 1;
  else if (e.notes) notesOk = notes.length === e.notes && (e.notes !== 2 || (pitches[1] !== pitches[0] && (!e.move || Math.sign(pitches[1] - pitches[0]) === e.move)));
  else notesOk = notes.length >= 1;
  const stringOk = f.string == null ? null : notes.length > 0 && notes.every((n) => n.string === f.string);
  // the expected mark where it belongs, and no other mark anywhere
  let markOk, extra = 0;
  const want = e.mark;
  if (!want) {
    markOk = marks.every((m) => !Object.keys(m).length);
  } else if (e.notes === 2) {
    markOk = notes.length === 2 && has(0, want) && (want !== 'legato' || (marks[0].legato && (tech.get(notes[0].id) || {}).legatoUp === (pitches[1] > pitches[0])));
  } else if (e.notes === 'trill') {
    const pairs = Math.max(1, notes.length - 1);
    markOk = notes.length >= 3 && notes.slice(0, -1).filter((_, i) => has(i, 'legato')).length >= 0.8 * pairs;
  } else {
    markOk = marks.some((m) => m[want]) && (!e.amount || marks.some((m) => m.bend && Math.abs(m.bend - e.amount) <= 0.5));
  }
  for (const m of marks) for (const k of Object.keys(m)) if (k !== want) extra++;
  if (extra) markOk = false;
  return { notesOk, stringOk, markOk, exact: notesOk && stringOk !== false && markOk, extra, marks, pitches, strings: notes.map((n) => n.string) };
}

const rows = [];
const t0 = Date.now();
for (const [i, f] of files.entries()) {
  const w = decodeWav(fs.readFileSync(f.file));
  const x = w.channels[0];
  let notes = [], tech = new Map();
  try {
    const res = await transcribe({ left: x, right: null, sr: w.sr, name: path.basename(f.file) }, { isolate: 'off', split: 'off', tuning: STANDARD, capo: 0 }, env);
    const part = res.parts[0];
    notes = part.notes.filter((n) => !n.dropped && !n.deleted).sort((a, b) => a.t0 - b.t0);
    tech = part.tech || new Map();
  } catch (err) {
    if (!/No guitar notes/.test(err.message)) throw err;
  }
  rows.push({ ...f, file: path.relative(dir, f.file), ...judge(f, notes, tech) });
  if ((i + 1) % 50 === 0) process.stderr.write(`\r${i + 1}/${files.length} ${((Date.now() - t0) / 1000 / (i + 1)).toFixed(2)} s/file`);
}
process.stderr.write('\n');

const pct = (a, b) => (b ? ((100 * a) / b).toFixed(0).padStart(4) + '%' : '    -');
console.log('technique'.padEnd(22), '  n', '  exact', ' notes', 'string', '  mark', ' extra marks');
const groups = new Map();
for (const r of rows) {
  const k = r.variant;
  if (!groups.has(k)) groups.set(k, []);
  groups.get(k).push(r);
}
for (const [k, rs] of [...groups].sort()) {
  const n = rs.length, ws = rs.filter((r) => r.stringOk != null);
  console.log(k.padEnd(22), String(n).padStart(3), ' ', pct(rs.filter((r) => r.exact).length, n), pct(rs.filter((r) => r.notesOk).length, n), pct(ws.filter((r) => r.stringOk).length, ws.length), pct(rs.filter((r) => r.markOk).length, n), pct(rs.filter((r) => r.extra).length, n));
}
const all = rows.length;
console.log('all'.padEnd(22), String(all).padStart(3), ' ', pct(rows.filter((r) => r.exact).length, all));
if (opt('--json')) fs.writeFileSync(opt('--json'), JSON.stringify(rows));
