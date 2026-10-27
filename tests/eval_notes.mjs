// Notes found by the guitar note model (onset F1: same pitch, onset within 50 ms), through the
// app's JS code (engine/hrgt.js, cleanNotes as the pipeline calls them), on clean or
// distorted recordings. The model's frame outputs are cached per take and effect.
//
// usage: node tests/eval_notes.mjs <models dir: hrgt.chain + hrgt.weights.bin> <annotations.json> <audio dir>
//          [--suffix _mic] [--every k] [--offset o] [--limit n] [--names a,b,c] [--fx tanh:30]
//          [--hop 500] [--cache dir] [--per-file] [--json out.json] [--write-wav dir] [--onset 0.4]
// annotations: tools/guitarset_json.py output ({take: {notes: [[t, dur, midi, ...]]}})
// audio: <audio dir>/<take><suffix>.wav
// --fx  tanh:G or amp:G, see tests/evallib.mjs applyFx
// --write-wav keeps the processed audio (16-bit WAV at the original rate), e.g. for tests/eval_strings.mjs
// --onset decodes with another onset threshold than the app's (the cached outputs are reused)
// env: ORT = path of onnxruntime-node's dist/index.js
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav, encodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { hrActivations } from '../engine/hrgt.js';
import { applyFx, loadNoteModel, readActs, writeActs, decodeNotes, matchNotes, f1 } from './evallib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--') && !['--per-file'].includes(args[i - 1])));
const [models, annPath, audioDir] = pos;
const suffix = opt('--suffix', '');
const every = Number(opt('--every', 1)), offset = Number(opt('--offset', 0)), limit = Number(opt('--limit', 1e9));
const fx = opt('--fx', '');
const hop = Number(opt('--hop', 500));
const onset = opt('--onset') ? Number(opt('--onset')) : undefined;
const ann = JSON.parse(fs.readFileSync(annPath, 'utf8'));
const names = opt('--names') ? opt('--names').split(',') : Object.keys(ann).sort().filter((_, i) => i % every === offset).slice(0, limit);
const cacheDir = opt('--cache', path.join(models, 'acts', `${path.basename(annPath, '.json')}_${fx.replace(':', '') || 'clean'}_h${hop}`));
fs.mkdirSync(cacheDir, { recursive: true });

let ort = null, model = null;
async function activations(name) {
  const p = path.join(cacheDir, name + '.f32');
  if (fs.existsSync(p)) return readActs(p);
  if (!model) {
    const ortMod = await import(process.env.ORT || 'onnxruntime-node');
    ort = ortMod.default || ortMod;
    model = await loadNoteModel(ort, models);
  }
  const w = decodeWav(fs.readFileSync(path.join(audioDir, `${name}${suffix}.wav`)));
  const mono = w.channels.length > 1 ? Float32Array.from(w.channels[0], (v, i) => 0.5 * (v + w.channels[1][i])) : w.channels[0];
  const y = applyFx(mono, w.sr, fx);
  if (opt('--write-wav')) { fs.mkdirSync(opt('--write-wav'), { recursive: true }); fs.writeFileSync(path.join(opt('--write-wav'), `${name}.wav`), Buffer.from(encodeWav([y], w.sr))); }
  const act = await hrActivations(ort, model, resample(y, w.sr, 16000, 64, 0.958), { hop });
  writeActs(p, act);
  return act;
}

let TP = 0, NE = 0, NR = 0;
const perFile = {};
const t0 = Date.now();
for (const [i, name] of names.entries()) {
  const est = decodeNotes(await activations(name), { onset });
  const ref = ann[name].notes.map(([t, d, v]) => ({ t0: t, t1: t + d, pitch: Math.round(v) }));
  const { ri } = matchNotes(est, ref);
  let tp = 0;
  for (let k = 0; k < ref.length; k++) if (ri[k] >= 0) tp++;
  TP += tp; NE += est.length; NR += ref.length;
  perFile[name] = { tp, est: est.length, ref: ref.length, F: f1(tp, est.length, ref.length).F };
  if (args.includes('--per-file')) console.log(name, (100 * perFile[name].F).toFixed(1));
  if (i % 20 === 19) console.error(`${i + 1}/${names.length} ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
const { P, R, F } = f1(TP, NE, NR);
console.log(`${path.basename(annPath, '.json')} ${fx || 'clean'} (${names.length} takes): notes found F1 ${(100 * F).toFixed(1)}  P ${(100 * P).toFixed(1)} R ${(100 * R).toFixed(1)}  refs ${NR} est ${NE}`);
if (opt('--json')) fs.writeFileSync(opt('--json'), JSON.stringify({ fx: fx || 'clean', takes: names.length, P, R, F, perFile }, null, 1));
if (model) await model.release();
