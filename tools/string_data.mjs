// Training and test data for the string classifier (engine/strings.js), from recordings where
// the string of every note is known. Writes <out>.f32 (stringFeatures() rows) and <out>.json
// ({nfeat, names, group, meta: [[take, pitch, string, fret, copy]]}) for tools/train_strings.py.
//
//   GuitarSet (acoustic, hexaphonic pickup; https://zenodo.org/records/3371780):
//     node tools/string_data.mjs guitarset <annotations.json> <audio dir> <out> [--version mic|mix] [--copies 2] [--augment n]
//     (annotations.json from tools/guitarset_json.py; audio dir holds <take>_mic.wav / <take>_mix.wav)
//   Guitar-TECHS (electric, one MIDI track per string; https://zenodo.org/records/14963133):
//     node tools/string_data.mjs gtechs-lags <dataset dir>          (once: MIDI pickup lag per take)
//     node tools/string_data.mjs gtechs <dataset dir> <out> [--kind micamp|directinput] [--copies 2] [--augment n] [--groups P1,P2]
//   EGSet12 (electric, test only; https://zenodo.org/records/11406378):
//     node tools/string_data.mjs egset <annotations.json> <audio dir> <out>
//   EGDB (electric through a hexaphonic pickup, as recorded and through amps; labels from
//   tools/egdb_json.py, split by clip) the same way, under its own group name:
//     node tools/string_data.mjs egset egdb_train.json EGDB/audio_DI <out> --copies 2 --group EGDB
//
// --copies n: n passes per take, every pass after the first with jittered note onsets (as the
// note detector's are). --augment n: n more passes through random tone changes (EQ, amp-style
// clipping, speaker roll-off), so the classifier doesn't lean on one guitar's tone.
import fs from 'node:fs';
import path from 'node:path';
import { decodeWav } from '../engine/wav.js';
import { resample } from '../engine/dsp.js';
import { stringFeatures, NFEAT } from '../engine/strings.js';
import { onsetEnvelope } from '../engine/beats.js';

const STD = [40, 45, 50, 55, 59, 64];
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));

function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

function loadMono44(file) {
  const w = decodeWav(fs.readFileSync(file));
  const x = w.channels.length > 1 ? Float32Array.from(w.channels[0], (v, i) => 0.5 * (v + w.channels[1][i])) : w.channels[0];
  return w.sr === 44100 ? x : resample(x, w.sr, 44100);
}

// ---- tone augmentation ------------------------------------------------------------------
function biquad(x, b0, b1, b2, a1, a2) {
  const y = new Float32Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
  }
  return y;
}
function peaking(x, sr, f, q, db) {
  const A = 10 ** (db / 40), w = (2 * Math.PI * f) / sr, al = Math.sin(w) / (2 * q), c = Math.cos(w), a0 = 1 + al / A;
  return biquad(x, (1 + al * A) / a0, (-2 * c) / a0, (1 - al * A) / a0, (-2 * c) / a0, (1 - al / A) / a0);
}
function shelf(x, sr, f, db, high) {
  const A = 10 ** (db / 40), w = (2 * Math.PI * f) / sr, c = Math.cos(w), sq = 2 * Math.sqrt(A) * (Math.sin(w) / 2) * Math.SQRT2;
  const k = high ? 1 : -1;
  const b0 = A * ((A + 1) + k * (A - 1) * c + sq), b1 = -2 * k * A * ((A - 1) + k * (A + 1) * c), b2 = A * ((A + 1) + k * (A - 1) * c - sq);
  const a0 = (A + 1) - k * (A - 1) * c + sq, a1 = 2 * k * ((A - 1) - k * (A + 1) * c), a2 = (A + 1) - k * (A - 1) * c - sq;
  return biquad(x, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}
function lowpass(x, sr, f) {
  const w = (2 * Math.PI * f) / sr, al = Math.sin(w) / Math.SQRT2, c = Math.cos(w), a0 = 1 + al;
  return biquad(x, (1 - c) / 2 / a0, (1 - c) / a0, (1 - c) / 2 / a0, (-2 * c) / a0, (1 - al) / a0);
}
function augment(x, sr, r) {
  const u = (a, b) => a + (b - a) * r();
  let y = x;
  for (let k = 0; k < 3; k++) y = peaking(y, sr, Math.exp(u(Math.log(120), Math.log(6000))), u(0.6, 2.5), u(-9, 9));
  y = shelf(y, sr, u(80, 300), u(-6, 6), false);
  y = shelf(y, sr, u(2000, 6000), u(-8, 6), true);
  let pk = 1e-9;
  for (let i = 0; i < y.length; i++) pk = Math.max(pk, Math.abs(y[i]));
  if (r() < 0.4) {
    const drive = Math.exp(u(Math.log(1.5), Math.log(12)));
    const z = new Float32Array(y.length);
    for (let i = 0; i < y.length; i++) z[i] = Math.tanh((drive * y[i]) / pk) / Math.tanh(drive);
    y = lowpass(lowpass(z, sr, u(3500, 7000)), sr, u(4000, 8000));
    pk = 1e-9;
    for (let i = 0; i < y.length; i++) pk = Math.max(pk, Math.abs(y[i]));
  }
  const g = u(0.3, 1) / pk;
  return Float32Array.from(y, (v) => v * g);
}

// ---- Guitar-TECHS -----------------------------------------------------------------------
function readMidi(p) {
  const b = fs.readFileSync(p);
  if (b.toString('latin1', 0, 4) !== 'MThd') throw new Error('not a MIDI file: ' + p);
  const ntrk = b.readUInt16BE(10), div = b.readUInt16BE(12);
  const tracks = [], tempos = [];
  let i = 14;
  for (let k = 0; k < ntrk; k++) {
    const len = b.readUInt32BE(i + 4), d = b.subarray(i + 8, i + 8 + len);
    i += 8 + len;
    let j = 0, t = 0, run = 0, name = null;
    const ev = [];
    const vlq = () => { let v = 0; for (;;) { const c = d[j++]; v = (v << 7) | (c & 0x7f); if (c < 0x80) return v; } };
    while (j < d.length) {
      t += vlq();
      let st = d[j];
      if (st === 0xff) {
        const typ = d[j + 1]; j += 2;
        const L = vlq();
        if (typ === 0x03) name = d.toString('latin1', j, j + L);
        if (typ === 0x51) tempos.push([t, (d[j] << 16) | (d[j + 1] << 8) | d[j + 2]]);
        j += L;
        continue;
      }
      if (st === 0xf0 || st === 0xf7) { j++; j += vlq(); continue; }
      if (st & 0x80) { run = st; j++; } else st = run;
      const hi = st & 0xf0, n = hi === 0xc0 || hi === 0xd0 ? 1 : 2;
      ev.push({ t, hi, ch: st & 15, a: d[j], b: n > 1 ? d[j + 1] : 0 });
      j += n;
    }
    tracks.push({ name, ev });
  }
  tempos.sort((x, y) => x[0] - y[0]);
  if (!tempos.length || tempos[0][0] > 0) tempos.unshift([0, 500000]);
  const sec = (tick) => {
    let s = 0, last = 0, us = tempos[0][1];
    for (const [tt, u] of tempos) { if (tt >= tick) break; s += ((tt - last) * us) / div / 1e6; last = tt; us = u; }
    return s + ((tick - last) * us) / div / 1e6;
  };
  const notes = [];
  for (const tr of tracks) {
    const open = new Map();
    for (const e of tr.ev) {
      const key = e.a + ':' + e.ch;
      if (e.hi === 0x90 && e.b > 0) open.set(key, e);
      else if (e.hi === 0x80 || e.hi === 0x90) {
        const s = open.get(key);
        if (s) { notes.push({ track: tr.name, pitch: e.a, t0: sec(s.t), t1: sec(e.t) }); open.delete(key); }
      }
    }
  }
  return notes.sort((x, y) => x.t0 - y.t0 || x.pitch - y.pitch);
}

const GT_STRING = { E: 0, A: 1, D: 2, G: 3, B: 4, e: 5 };
// bends and harmonics: the written pitch doesn't describe the sound
const GT_SKIP = /Bendings|Harmonics/;
function gtechsTakes(dir) {
  const lags = fs.existsSync(path.join(dir, 'lags.json')) ? JSON.parse(fs.readFileSync(path.join(dir, 'lags.json'), 'utf8')) : {};
  const takes = [];
  for (const set of fs.readdirSync(dir).filter((d) => /^P\d_/.test(d)).sort()) {
    const midiDir = path.join(dir, set, 'midi');
    if (!fs.existsSync(midiDir)) continue;
    for (const mf of fs.readdirSync(midiDir).filter((f) => f.endsWith('.mid')).sort()) {
      const key = mf.replace(/^midi_/, '').replace(/\.mid$/, '');
      const audio = {};
      for (const kind of ['micamp', 'directinput']) {
        const p = path.join(dir, set, 'audio', kind, `${kind}_${key}.wav`);
        if (fs.existsSync(p)) audio[kind] = p;
      }
      const name = `${set}_${key}`;
      takes.push({ name, group: set.slice(0, 2), midi: path.join(midiDir, mf), audio, lag: lags[name] });
    }
  }
  return takes;
}
function gtechsNotes(take) {
  const lag = take.lag ? take.lag.lag : 0.05;
  return readMidi(take.midi).filter((n) => n.track in GT_STRING)
    .map((n) => ({ t0: Math.max(0, n.t0 - lag), t1: Math.max(0.02, n.t1 - lag), pitch: n.pitch, string: GT_STRING[n.track] }))
    .filter((n) => n.pitch - STD[n.string] >= 0 && n.pitch - STD[n.string] <= 24)
    .sort((a, b) => a.t0 - b.t0 || a.pitch - b.pitch);
}
// The MIDI comes from a pickup with some lag: find it per take from the audio's onsets.
function measureLag(take) {
  const x = resample(loadMono44(take.audio.directinput || take.audio.micamp), 44100, 22050);
  const { env, fps } = onsetEnvelope(x, 22050, { hop: 256 });
  let mx = 0;
  for (const v of env) mx = Math.max(mx, v);
  const peaks = [];
  for (let i = 1; i < env.length - 1; i++) if (env[i] > 0.1 * mx && env[i] >= env[i - 1] && env[i] > env[i + 1]) peaks.push(i / fps);
  const ons = [...new Set(readMidi(take.midi).map((n) => Math.round(n.t0 * 200) / 200))];
  let best = { lag: 0.05, match: 0 };
  for (let off = -0.12; off <= 0.12; off += 0.0025) {
    let hit = 0, j = 0;
    for (const o of ons) {
      const tt = o + off;
      while (j < peaks.length && peaks[j] < tt - 0.02) j++;
      if (j < peaks.length && Math.abs(peaks[j] - tt) <= 0.02) hit++;
    }
    if (hit / ons.length > best.match) best = { lag: -off, match: hit / ons.length };
  }
  return best;
}

// ---- writing ----------------------------------------------------------------------------
function write(out, takes, copies, nAug, seed) {
  const r = rng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  const feats = [], meta = [];
  takes.forEach((t, ti) => {
    const x = t.audio();
    const notes = t.notes;
    for (let c = 0; c < copies + nAug; c++) {
      // pass 0 is the take as it is (what tests use); the others have copy numbers from 1 up
      const aug = c >= copies;
      const jitter = c === 0 && !aug ? null : Float64Array.from(notes, () => Math.max(-0.03, Math.min(0.03, 0.012 * gauss())));
      feats.push(stringFeatures(aug ? augment(x, 44100, r) : x, notes, { jitter }));
      for (const n of notes) meta.push([ti, n.pitch, n.string, n.pitch - STD[n.string], aug ? 1 + c : c]);
    }
    process.stderr.write(`\r${ti + 1}/${takes.length} ${t.name}            `);
  });
  const all = new Float32Array(meta.length * NFEAT);
  let o = 0;
  for (const f of feats) { all.set(f, o); o += f.length; }
  fs.writeFileSync(out + '.f32', Buffer.from(all.buffer));
  fs.writeFileSync(out + '.json', JSON.stringify({ nfeat: NFEAT, names: takes.map((t) => t.name), group: takes.map((t) => t.group), meta }));
  console.error(`\n${out}: ${meta.length} rows`);
}

const cmd = pos[0];
const copies = Number(opt('--copies', cmd === 'egset' ? 1 : 2)), nAug = Number(opt('--augment', 0));
if (cmd === 'guitarset' || cmd === 'egset') {
  const [, annPath, audioDir, out] = pos;
  const ann = JSON.parse(fs.readFileSync(annPath, 'utf8'));
  const version = opt('--version', 'mic');
  const takes = Object.keys(ann).sort().map((name) => ({
    name,
    group: cmd === 'egset' ? opt('--group', 'EG') : name.slice(0, 2),
    audio: () => loadMono44(path.join(audioDir, cmd === 'egset' ? `${name}.wav` : `${name}_${version}.wav`)),
    notes: ann[name].notes.map(([t, d, v, s]) => ({ t0: t, t1: t + d, pitch: Math.round(v), string: s })),
  }));
  write(out, takes, copies, nAug, 12345 + nAug);
} else if (cmd === 'gtechs-lags') {
  const dir = pos[1];
  const lags = {};
  for (const t of gtechsTakes(dir)) {
    lags[t.name] = measureLag(t);
    console.log(t.name.padEnd(34), 'lag', lags[t.name].lag.toFixed(4), 'onsets matched', (100 * lags[t.name].match).toFixed(0) + '%');
  }
  fs.writeFileSync(path.join(dir, 'lags.json'), JSON.stringify(lags, null, 1));
} else if (cmd === 'gtechs') {
  const [, dir, out] = pos;
  const kind = opt('--kind', 'micamp');
  const groups = opt('--groups', null);
  // takes whose MIDI matches the audio's onsets poorly are left out
  const takes = gtechsTakes(dir).filter((t) => t.audio[kind] && !GT_SKIP.test(t.name) && (!t.lag || t.lag.match >= 0.6) && (!groups || groups.split(',').includes(t.group)))
    .map((t) => ({ name: t.name, group: t.group, audio: () => loadMono44(t.audio[kind]), notes: gtechsNotes(t) }));
  write(out, takes, copies, nAug, 777 + nAug);
} else {
  console.error('usage: see the comment at the top of tools/string_data.mjs');
  process.exit(1);
}
