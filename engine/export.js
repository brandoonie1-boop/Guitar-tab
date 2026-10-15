// Exporters: alphaTex (for rendering / Guitar Pro export via alphaTab), plain-text ASCII tab, MIDI.
import { midiName } from './dsp.js';
import { tuningName } from './tunings.js';

const TPQ = 960;

function stringNo(sIdx) { return 6 - sIdx; } // 0 = low string -> alphaTex 6

function durToken(grid, len) {
  // returns list of [durationValue, dotted, tuplet] pieces summing to len slots
  if (grid === 4) return ({ 1: [[16]], 2: [[8]], 3: [[8, 1]], 4: [[4]] })[len] || [[4]];
  if (grid === 3) return ({ 1: [[8, 0, 3]], 2: [[4, 0, 3]], 3: [[4]] })[len] || [[4]];
  if (grid === 8) return ({ 1: [[32]], 2: [[16]], 3: [[16, 1]], 4: [[8]], 5: [[8], [32]], 6: [[8, 1]], 7: [[8, 1], [32]], 8: [[4]] })[len] || [[4]];
  if (grid === 2) return ({ 1: [[8]], 2: [[4]] })[len] || [[4]];
  return [[4]];
}

function noteTex(n, tie, techMap) {
  const s = stringNo(n.string);
  if (tie) return `-.${s}`;
  const eff = [];
  const t = techMap.get(n.id);
  if (t) {
    if (t.legato) eff.push('h');
    if (t.slide) eff.push('sl');
    if (t.bend) eff.push(`b (0 ${Math.max(1, Math.round(t.bend * 2))})`);
    if (t.vibrato) eff.push('v');
  }
  if (n.ghost) eff.push('g');
  return `${n.fret}.${s}${eff.length ? '{' + eff.join(' ') + '}' : ''}`;
}

/** Merge whole-beat segments of the same sustained notes into half / dotted half / whole notes. */
function beatTokens(measure, meter) {
  const out = [];
  const B = measure.beats;
  let j = 0;
  const fullSame = (a, b) => a.segs.length === 1 && b.segs.length === 1 && a.segs[0].len === a.grid && b.segs[0].len === b.grid
    && b.segs[0].tie !== false && (a.segs[0].notes === b.segs[0].notes || (!a.segs[0].notes && !b.segs[0].notes));
  while (j < B.length) {
    const beat = B[j];
    let run = 1;
    while (j + run < B.length && fullSame(B[j + run - 1], B[j + run]) && (B[j + run].segs[0].tie || !B[j + run].segs[0].notes)) run++;
    if (beat.segs.length === 1 && beat.segs[0].len === beat.grid && run > 1) {
      let take = 1;
      if (run >= 4 && j === 0 && meter === 4) take = 4;
      else if (run >= 3 && (j === 0 || (meter === 4 && j === 1))) take = 3;
      else if (run >= 2 && j % 2 === 0) take = 2;
      if (take > 1) {
        const seg = beat.segs[0];
        const val = take === 4 ? [[1]] : take === 3 ? [[2, 1]] : [[2]];
        out.push({ seg, pieces: val });
        j += take;
        continue;
      }
    }
    for (const seg of beat.segs) out.push({ seg, pieces: durToken(beat.grid, seg.len) });
    j++;
  }
  return out;
}

export function toAlphaTex(doc) {
  // doc: {title, bpm, meter, parts:[{name, tuning, capo, measures, tech:Map}]}
  // returns {tex, maps}: maps[part][bar][beat] = {stringIndex(0=low): noteId}
  const lines = [];
  const maps = [];
  const esc = (s) => String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  lines.push(`\\title "${esc(doc.title || 'Transcription')}"`);
  if (doc.subtitle) lines.push(`\\subtitle "${esc(doc.subtitle)}"`);
  lines.push(`\\tempo ${Math.max(20, Math.round(doc.bpm || 120))}`);
  for (const part of doc.parts) {
    const pmap = [];
    maps.push(pmap);
    lines.push(`\\track "${esc(part.name)}"`);
    lines.push('\\staff {tabs}');
    const tun = part.tuning.slice().reverse().map((m) => midiName(m, true)).join(' ');
    lines.push(`\\tuning (${tun}) { label "${esc(tuningName(part.tuning))}" }`);
    if (part.capo) lines.push(`\\capo ${part.capo}`);
    const techMap = part.tech || new Map();
    let prevLen = 0;
    part.measures.forEach((m, mi) => {
      // time signature on the first bar and wherever the bar length changes
      const len = m.beats.length;
      const head = len !== prevLen ? `\\ts (${len} 4) ` : '';
      prevLen = len;
      const toks = [];
      const bmap = [];
      for (const { seg, pieces } of beatTokens(m, len)) {
        pieces.forEach(([dv, dotted, tup], pi) => {
          const eff = [];
          if (dotted) eff.push('d');
          if (tup) eff.push(`tu ${tup}`);
          const effStr = eff.length ? ` {${eff.join(' ')}}` : '';
          let content;
          const nm = {};
          if (!seg.notes) content = 'r';
          else {
            const tie = seg.tie || pi > 0;
            const ns = seg.notes.map((n) => noteTex(n, tie, techMap));
            for (const n of seg.notes) nm[n.string] = n.id;
            content = ns.length === 1 ? ns[0] : `(${ns.join(' ')})`;
          }
          bmap.push(nm);
          toks.push(`${content}.${dv}${effStr}`);
        });
      }
      pmap.push(bmap);
      lines.push(`${head}${toks.join(' ')} |`);
    });
  }
  return { tex: lines.join('\n'), maps };
}

/** Plain text tab (the kind you'd post on Ultimate Guitar). */
export function toAscii(doc, opts = {}) {
  const width = opts.width || 78;
  const out = [];
  out.push(doc.title || 'Transcription');
  for (const part of doc.parts) {
    out.push('');
    out.push(`${part.name}`);
    out.push(`Tuning: ${part.tuning.map((m) => midiName(m, false)).join(' ')} (${tuningName(part.tuning)}), low to high`);
    out.push(part.capo ? `Capo: fret ${part.capo} (fret numbers below are counted from the capo, 0 = capo)` : 'Capo: none');
    const lens = new Set(part.measures.map((m) => m.beats.length));
    out.push(`Tempo: ~${Math.round(doc.bpm)} BPM, ${lens.size > 1 ? `mostly ${doc.meter}/4 (the meter changes; bars are marked |)` : `${doc.meter}/4`}`);
    out.push('');
    const labels = part.tuning.map((m, i) => midiName(m, false)).map((s, i, arr) => (i === arr.length - 1 ? s.toLowerCase() : s));
    const labW = Math.max(...labels.map((l) => l.length));
    const techMap = part.tech || new Map();
    const measureStrs = part.measures.map((m) => {
      const cols = [];
      for (const beat of m.beats) {
        const slotW = beat.grid === 8 ? 1 : beat.grid === 3 ? 2 : 2;
        const slots = Array.from({ length: beat.grid }, () => new Array(6).fill(''));
        for (const seg of beat.segs) {
          if (!seg.notes || seg.tie) continue;
          for (const n of seg.notes) {
            const t = techMap.get(n.id);
            let txt = String(n.fret);
            if (t && t.bend) txt += 'b';
            if (t && t.vibrato) txt += '~';
            if (t && t.legato) txt += t.legatoUp === false ? 'p' : 'h';
            if (t && t.slide) txt += t.slideUp === false ? '\\' : '/';
            slots[seg.s][n.string] = txt;
          }
        }
        for (const sl of slots) {
          const w = Math.max(slotW, ...sl.map((x) => x.length + 1));
          cols.push(sl.map((x) => (x + '-'.repeat(w)).slice(0, w)));
        }
      }
      // rows high string first
      const rows = [];
      for (let s = 5; s >= 0; s--) rows.push('-' + cols.map((c) => c[s]).join('') + '|');
      return rows;
    });
    // lay measures out in lines
    let line = null;
    const flush = () => {
      if (!line) return;
      for (let r = 0; r < 6; r++) out.push(line[r]);
      out.push('');
      line = null;
    };
    for (const ms of measureStrs) {
      if (!line) line = labels.slice().reverse().map((l) => l.padEnd(labW) + '|');
      if (line[0].length + ms[0].length > width && line[0].length > labW + 1) {
        flush();
        line = labels.slice().reverse().map((l) => l.padEnd(labW) + '|');
      }
      for (let r = 0; r < 6; r++) line[r] += ms[r];
    }
    flush();
  }
  out.push('Made with Guitar Tab (auto transcription). Check it by ear: tuning, capo and some notes can be off.');
  return out.join('\n');
}

/** Standard MIDI file (type 1) with one track per part, real (unquantized) timing. */
export function toMidi(doc) {
  const tpq = 480;
  const bpm = doc.bpm || 120;
  const secToTick = (s) => Math.max(0, Math.round((s * bpm * tpq) / 60));
  const tracks = [];
  // tempo track
  {
    const usPerQ = Math.round(60000000 / bpm);
    const ev = [[0, [0xff, 0x51, 0x03, (usPerQ >> 16) & 255, (usPerQ >> 8) & 255, usPerQ & 255]], [0, [0xff, 0x58, 0x04, doc.meter || 4, 2, 24, 8]]];
    tracks.push(ev);
  }
  doc.parts.forEach((part, pi) => {
    const ch = pi % 15;
    const ev = [[0, [0xc0 | ch, 25]]]; // steel guitar
    for (const n of part.notes) {
      if (n.string == null) continue;
      const vel = Math.max(20, Math.min(127, Math.round((n.amp || 0.5) * 160)));
      ev.push([secToTick(n.t0), [0x90 | ch, n.pitch, vel]]);
      ev.push([secToTick(Math.max(n.t1, n.t0 + 0.03)), [0x80 | ch, n.pitch, 0]]);
    }
    ev.sort((a, b) => a[0] - b[0] || (a[1][0] & 0xf0) - (b[1][0] & 0xf0));
    tracks.push(ev);
  });
  const bytes = [];
  const push32 = (v) => bytes.push((v >>> 24) & 255, (v >>> 16) & 255, (v >>> 8) & 255, v & 255);
  const push16 = (v) => bytes.push((v >> 8) & 255, v & 255);
  const str = (s) => { for (const c of s) bytes.push(c.charCodeAt(0)); };
  str('MThd'); push32(6); push16(1); push16(tracks.length); push16(tpq);
  for (const ev of tracks) {
    const data = [];
    let last = 0;
    const vlq = (v) => {
      const st = [v & 0x7f];
      while ((v >>= 7)) st.unshift((v & 0x7f) | 0x80);
      data.push(...st);
    };
    for (const [t, msg] of ev) { vlq(t - last); last = t; data.push(...msg); }
    data.push(0, 0xff, 0x2f, 0);
    str('MTrk'); push32(data.length); bytes.push(...data);
  }
  return new Uint8Array(bytes);
}
