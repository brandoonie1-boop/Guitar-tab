// Tab playback with recorded guitar notes. Every note plays a recording of that string at that fret
// (tools/build_guitar_sounds.py), and the parts behave like a guitar does:
//  - a string plays one note at a time: a new note on it stops the one before (unless let ring);
//  - hammer-ons and pull-offs are not picked again (the new note starts after the pick attack);
//  - slides and bends bend the recorded note itself; vibrato wobbles it;
//  - palm mutes are damped, ghost notes are soft, chords are strummed (a few ms between strings);
//  - electric tones go through an amp after the strings are mixed, as in a real rig: drive, tone
//    and a speaker cabinet response measured from a real amp (Guitar-TECHS, DI vs. amp mic).
// It looks like an <audio> element to its user (currentTime, playbackRate, play/pause, events),
// so the tab view follows it with the same code that follows the recording.

import { fetchAsset } from './assets.js';

const OPEN = [40, 45, 50, 55, 59, 64];
const BANK_ASSET = { electric: 'soundElectric', nylon: 'soundNylon', acoustic: 'soundAcoustic' };

// level: output trim so every sound plays about equally loud (measured on rendered tabs)
export const TONES = {
  acoustic: { name: 'Acoustic', bank: 'acoustic', fallback: 'nylon', reverb: 0.16, level: 0.6 },
  nylon: { name: 'Nylon', bank: 'nylon', reverb: 0.18, level: 0.6 },
  clean: { name: 'Clean electric', bank: 'electric', reverb: 0.13, level: 1, amp: { drive: 0, low: 1, mid: 0, high: 1.5, cab: 'bright', level: 6.5 } },
  crunch: { name: 'Crunch', bank: 'electric', reverb: 0.1, level: 1, amp: { drive: 7, low: 2, mid: 2, high: 2, cab: 'bright', level: 0.75 } },
  lead: { name: 'Distortion', bank: 'electric', reverb: 0.1, level: 1, amp: { drive: 40, low: 3, mid: -3, high: 3, cab: 'bright', level: 0.65, tight: 140 } },
};

const banks = new Map(); // `${ctx rate}:${name}` -> Promise<bank>
const irs = new Map();

/** A packed sound file (tools/build_guitar_sounds.py pack): name -> bytes. */
const packs = new Map();
function soundPack(key) {
  if (!packs.has(key)) {
    packs.set(key, fetchAsset(key).then((b) => {
      if (String.fromCharCode(b[0], b[1], b[2], b[3]) !== 'GTSB') throw new Error('not a sound pack');
      const n = new DataView(b.buffer, b.byteOffset).getUint32(4, true);
      const head = JSON.parse(new TextDecoder().decode(b.subarray(8, 8 + n)));
      const out = {};
      for (const [name, [at, len]] of Object.entries(head)) out[name] = b.slice(8 + n + at, 8 + n + at + len);
      return out;
    }));
    packs.get(key).catch(() => packs.delete(key));
  }
  return packs.get(key);
}

/** Decode a bank and find each note's attack near its nominal start (MP3 decoders differ in delay). */
async function loadBank(ctx, name) {
  const key = `${ctx.sampleRate}:${name}`;
  if (!banks.has(key)) {
    banks.set(key, (async () => {
      const pack = await soundPack(BANK_ASSET[name]);
      const meta = JSON.parse(new TextDecoder().decode(pack.json));
      const buffer = await ctx.decodeAudioData(pack.mp3.buffer);
      const x = buffer.getChannelData(0);
      const sr = buffer.sampleRate;
      const notes = new Map();
      for (const n of meta.notes) {
        const a = Math.max(0, Math.floor((n.at - 0.08) * sr)), b = Math.min(x.length, Math.floor((n.at + 0.3) * sr));
        let pk = 0;
        for (let i = a; i < b; i++) pk = Math.max(pk, Math.abs(x[i]));
        let i = a;
        while (i < b && Math.abs(x[i]) < 0.03 * pk) i++;
        const start = Math.max(0, i / sr - 0.004);
        notes.set(`${n.s}:${n.f}`, { s: n.s, f: n.f, start, len: n.len - (start - n.at), cents: n.cents || 0 });
      }
      return { buffer, notes };
    })());
    banks.get(key).catch(() => banks.delete(key));
  }
  return banks.get(key);
}

async function loadIr(ctx, name) {
  const key = `${ctx.sampleRate}:${name}`;
  if (!irs.has(key)) irs.set(key, soundPack('soundCabs').then((p) => ctx.decodeAudioData(p[name].slice().buffer)));
  return irs.get(key);
}

/** A soft clipper curve for the amp's drive stage. */
function driveCurve(drive) {
  const n = 2048, c = new Float32Array(n);
  const k = Math.max(0.001, drive);
  const norm = Math.tanh(k);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1;
    // asymmetric a little, like a tube stage: even harmonics
    c[i] = Math.tanh(k * (x + 0.08 * x * x)) / norm;
  }
  return c;
}

function roomIr(ctx, seconds = 1.6) {
  const sr = ctx.sampleRate, n = Math.floor(seconds * sr);
  const b = ctx.createBuffer(2, n, sr);
  let seed = 7;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 * 2 - 1; };
  for (let ch = 0; ch < 2; ch++) {
    const d = b.getChannelData(ch);
    let lp = 0;
    for (let i = 0; i < n; i++) {
      const t = i / sr;
      // early reflections, then a darkening exponential tail
      const env = t < 0.012 ? 0 : Math.exp(-t * 4.2);
      const a = Math.min(0.9, 0.25 + t * 0.6); // more high cut later in the tail
      lp = a * lp + (1 - a) * rnd();
      d[i] = lp * env * 0.6;
    }
  }
  return b;
}

/**
 * Note events from an alphaTab score: one track, times from the bars' sync points (the same
 * timeline the cursor uses). ampOf(beat, note) -> 0..1 loudness or null.
 */
export function scoreEvents(score, trackIndex, syncPoints, ampOf) {
  const track = score.tracks[trackIndex];
  if (!track) return [];
  const mbs = score.masterBars;
  const ms = new Array(mbs.length + 1);
  const sp = new Map((syncPoints || []).map((p) => [p.barIndex, p.millisecondOffset / 1000]));
  const tempo = score.tempo || 120;
  for (let i = 0; i < mbs.length; i++) {
    if (sp.has(i)) ms[i] = sp.get(i);
    else if (i > 0) ms[i] = ms[i - 1] + (mbs[i - 1].calculateDuration() / 960) * (60 / tempo);
    else ms[i] = 0;
  }
  const last = mbs.length - 1;
  ms[mbs.length] = mbs.length > 1 ? ms[last] + (ms[last] - ms[last - 1]) * (mbs[last].calculateDuration() / Math.max(1, mbs[last - 1].calculateDuration()))
    : ms[last] + (mbs[last].calculateDuration() / 960) * (60 / tempo);
  const toSec = (tick) => {
    let i = 0;
    while (i < last && mbs[i + 1].start <= tick) i++;
    const d = mbs[i].calculateDuration();
    return ms[i] + (ms[i + 1] - ms[i]) * ((tick - mbs[i].start) / d);
  };
  const staff = track.staves[0];
  const events = [];
  const byNote = new Map();
  for (const bar of staff.bars) {
    const mb = mbs[bar.index];
    // strumming: down on the beat, up in between (on the 16ths when the bar has 16th strums)
    let has16 = false;
    for (const voice of bar.voices) for (const beat of voice.beats) if (beat.notes && beat.notes.length > 1 && beat.playbackStart % 480 === 240) has16 = true;
    for (const voice of bar.voices) {
      for (const beat of voice.beats) {
        if (beat.isRest || !beat.notes || !beat.notes.length) continue;
        const t0 = mb.start + beat.playbackStart, t1 = t0 + beat.playbackDuration;
        const ts = toSec(t0), te = toSec(t1);
        const up = has16 ? beat.playbackStart % 480 === 240 : beat.playbackStart % 960 === 480;
        let lo = 6, hi = -1;
        for (const n of beat.notes) { lo = Math.min(lo, n.string - 1); hi = Math.max(hi, n.string - 1); }
        for (const n of beat.notes) {
          if (n.isTieDestination && n.tieOrigin && byNote.has(n.tieOrigin)) {
            const o = byNote.get(n.tieOrigin);
            o.end = te;
            byNote.set(n, o);
            continue;
          }
          const s = n.string - 1; // alphaTab: 1 = lowest string
          const pitch = n.stringTuning + n.fret;
          const e = {
            t: ts, end: te, s, pitch, fret: n.fret,
            amp: ampOf ? ampOf(beat, n) : null,
            legato: !!n.isHammerPullDestination,
            slideTo: null,
            slideIn: n.slideInType || 0,
            slideOut: n.slideOutType || 0,
            bend: n.hasBend && n.bendPoints ? n.bendPoints.map((p) => [p.offset / 60, p.value * 50]) : null,
            vibrato: n.vibrato || 0,
            ghost: !!n.isGhost, dead: !!n.isDead, palmMute: !!n.isPalmMute, letRing: !!n.isLetRing,
            harmonic: n.harmonicType || 0,
            // strings the pick has crossed before this one (from the low side on a down stroke)
            chordIndex: beat.notes.length > 1 ? (up ? hi - s : s - lo) : -1, chordSize: beat.notes.length, up: beat.notes.length > 1 && up,
          };
          byNote.set(n, e);
          events.push(e);
        }
      }
    }
  }
  // slides: the note glides into the next one on its string (which isn't picked again)
  for (const [n, e] of byNote) {
    if (n.slideTarget && byNote.has(n.slideTarget) && byNote.get(n.slideTarget) !== e) {
      const tgt = byNote.get(n.slideTarget);
      e.slideTo = tgt;
      if (e.slideOut !== 1) tgt.slidDest = true; // legato slide: the target isn't picked
    }
  }
  events.sort((a, b) => a.t - b.t || a.s - b.s);
  return events;
}

export class GuitarSynth extends EventTarget {
  constructor() {
    super();
    this.ctx = null;
    this.events = [];
    this.tone = 'clean';
    this.bank = null;
    this._rate = 1;
    this._pos = 0; // song seconds when paused
    this.paused = true;
    this.loop = false;
    this.duration = 0;
    this.metronome = false;
    this.countIn = false;
    this.beats = []; // song seconds of beats, for the metronome / count-in
    this.beatInfo = () => ({ pos: 0, per: 4 }); // beat i: position in its bar, beats in the bar
    this.voices = new Array(6).fill(null);
    this.timer = 0;
    this.tick = 0;
  }

  async _ensureCtx() {
    if (!this.ctx) {
      const AC = self.AudioContext || self.webkitAudioContext;
      this.ctx = new AC({ latencyHint: 'playback' });
    }
    if (!this.master) {
      const c = this.ctx;
      this.master = c.createGain();
      // gentle bus compression, then a limiter so chords never clip
      this.comp = c.createDynamicsCompressor();
      this.comp.threshold.value = -20; this.comp.knee.value = 8; this.comp.ratio.value = 3; this.comp.attack.value = 0.004; this.comp.release.value = 0.18;
      this.limit = c.createDynamicsCompressor();
      this.limit.threshold.value = -4; this.limit.knee.value = 0; this.limit.ratio.value = 20; this.limit.attack.value = 0.001; this.limit.release.value = 0.08;
      this.master.connect(this.comp).connect(this.limit).connect(c.destination);
      this.reverb = c.createConvolver();
      this.reverb.buffer = roomIr(c);
      this.revSend = c.createGain();
      this.revSend.connect(this.reverb).connect(this.master);
      this.click = c.createGain();
      this.click.gain.value = 0.5;
      this.click.connect(c.destination);
    }
    if (this.ctx.state === 'suspended' && !this.offline) { try { await this.ctx.resume(); } catch { /* needs a gesture */ } }
    return this.ctx;
  }

  /** Pick the sound: 'acoustic' | 'nylon' | 'clean' | 'crunch' | 'lead'. Loads its notes. */
  async setTone(tone) {
    const c = await this._ensureCtx();
    this.tone = TONES[tone] ? tone : 'clean';
    const t = TONES[this.tone];
    let bank;
    try { bank = await loadBank(c, t.bank); } catch (err) {
      if (!t.fallback) throw err;
      bank = await loadBank(c, TONES[t.fallback].bank);
    }
    this.bank = bank;
    // the tone's chain: strings -> (amp) -> master, plus a room
    if (this.input) { try { this.input.disconnect(); } catch { /* ignore */ } }
    if (this.chainOut) { try { this.chainOut.disconnect(); } catch { /* ignore */ } }
    this.input = c.createGain();
    let node = this.input;
    if (t.amp) {
      const a = t.amp;
      const hp = c.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = a.tight || 70; hp.Q.value = 0.7;
      const pre = c.createGain(); pre.gain.value = 1 + a.drive * 0.15;
      const sh = c.createWaveShaper(); sh.curve = driveCurve(a.drive > 0 ? a.drive : 0.4); sh.oversample = '4x';
      const low = c.createBiquadFilter(); low.type = 'lowshelf'; low.frequency.value = 160; low.gain.value = a.low;
      const mid = c.createBiquadFilter(); mid.type = 'peaking'; mid.frequency.value = 750; mid.Q.value = 0.9; mid.gain.value = a.mid;
      const high = c.createBiquadFilter(); high.type = 'highshelf'; high.frequency.value = 2800; high.gain.value = a.high;
      const cab = c.createConvolver();
      cab.normalize = true;
      try { cab.buffer = await loadIr(c, a.cab); } catch { /* without the cabinet: a plain low-pass */ }
      const lp = c.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = a.drive > 10 ? 6500 : 9000;
      const out = c.createGain(); out.gain.value = a.level;
      node.connect(hp); hp.connect(pre); pre.connect(sh); sh.connect(low); low.connect(mid); mid.connect(high);
      if (cab.buffer) { high.connect(cab); cab.connect(lp); } else high.connect(lp);
      lp.connect(out);
      node = out;
    } else {
      // acoustic: no sub-bass rumble, a touch of air
      const hp = c.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 75; hp.Q.value = 0.7;
      const eq = c.createBiquadFilter(); eq.type = 'highshelf'; eq.frequency.value = 6000; eq.gain.value = 3;
      node.connect(hp); hp.connect(eq);
      node = eq;
    }
    const trim = c.createGain();
    trim.gain.value = t.level ?? 1;
    node.connect(trim);
    node = trim;
    this.chainOut = node;
    node.connect(this.master);
    node.connect(this.revSend);
    this.revSend.gain.value = t.reverb;
    this.master.gain.value = this.volume;
  }

  setEvents(events, duration) {
    this.events = events;
    this.duration = duration || (events.length ? Math.max(...events.map((e) => e.end)) + 1 : 0);
  }

  // ---- media-element-like API ----
  get volume() { return this._volume ?? 1; }
  set volume(v) { this._volume = v; if (this.master) this.master.gain.value = v; }

  get playbackRate() { return this._rate; }
  set playbackRate(r) {
    r = Math.max(0.1, Math.min(4, r));
    // like a media element: no event (and no restart) when nothing changes
    if (r === this._rate) return;
    const t = this.currentTime;
    this._rate = r;
    if (!this.paused) this._restart(t);
    this.dispatchEvent(new Event('ratechange'));
  }

  get currentTime() {
    if (this.paused || !this.ctx) return this._pos;
    return this._songAt0 + Math.max(0, this.ctx.currentTime - this._ctxAt0) * this._rate;
  }

  set currentTime(t) {
    this._pos = Math.max(0, t);
    if (!this.paused) this._restart(this._pos);
    this.dispatchEvent(new Event('seeked'));
  }

  async play() {
    if (!this.paused) return;
    await this._ensureCtx();
    if (!this.bank) await this.setTone(this.tone);
    if (this._pos >= this.duration - 0.05) this._pos = 0;
    this.paused = false;
    let from = this._pos;
    let lead = 0.06;
    if (this.countIn) lead += this._scheduleCountIn(from, this.ctx.currentTime + lead);
    this._start(from, lead);
    this.dispatchEvent(new Event('play'));
  }

  pause() {
    if (this.paused) return;
    this._pos = this.currentTime;
    this.paused = true;
    clearInterval(this.timer);
    this._silence(0.03);
    this.dispatchEvent(new Event('pause'));
  }

  load() {}

  // ---- scheduling ----
  _start(from, lead = 0.06) {
    clearInterval(this.timer);
    this._songAt0 = from;
    this._ctxAt0 = this.ctx.currentTime + lead;
    this._next = this.events.findIndex((e) => e.end > from);
    if (this._next < 0) this._next = this.events.length;
    this._nextBeat = this.beats.findIndex((b) => b >= from - 1e-6);
    // notes still sounding at the start point are played too (picked again, as the start)
    this.timer = setInterval(() => this._pump(), 25);
    this._pump();
  }

  _restart(t) {
    this._silence(0.02);
    this._start(t, 0.03);
  }

  _ctxTime(songT) { return this._ctxAt0 + (songT - this._songAt0) / this._rate; }

  _pump() {
    if (this.paused) return;
    const now = this.ctx.currentTime;
    const horizon = this._songAt0 + (now + 0.25 - this._ctxAt0) * this._rate;
    // loop / end
    const end = this.duration;
    if (this.currentTime >= end) {
      if (this.loop) { this._restart(0); this.dispatchEvent(new Event('seeked')); return; }
      this._pos = this.duration;
      this.paused = true;
      clearInterval(this.timer);
      this.dispatchEvent(new Event('pause'));
      this.dispatchEvent(new Event('ended'));
      return;
    }
    while (this._next < this.events.length && this.events[this._next].t < Math.min(horizon, end)) {
      const e = this.events[this._next++];
      if (e.end <= this._songAt0) continue;
      this._note(e);
    }
    if (this.metronome && this._nextBeat >= 0) {
      while (this._nextBeat < this.beats.length && this.beats[this._nextBeat] < Math.min(horizon, end)) {
        const b = this._nextBeat++;
        const t = this.beats[b];
        if (t >= this._songAt0) this._clickAt(this._ctxTime(t), this.beatInfo(b).pos === 0);
      }
    }
    this.tick++;
    if (this.tick % 4 === 0) this.dispatchEvent(new Event('timeupdate'));
  }

  _clickAt(when, accent) {
    const c = this.ctx;
    const o = c.createOscillator(), g = c.createGain();
    o.frequency.value = accent ? 1760 : 1320;
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(accent ? 0.9 : 0.55, when + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, when + 0.05);
    o.connect(g).connect(this.click);
    o.start(when); o.stop(when + 0.06);
  }

  /** One bar of clicks before playback; returns its length in seconds (real time). */
  _scheduleCountIn(from, when) {
    let i = this.beats.findIndex((b) => b >= from - 1e-6);
    if (i < 1) i = 1;
    const beatLen = Math.max(0.2, (this.beats[i] - this.beats[i - 1]) || 0.5) / this._rate;
    const n = this.beatInfo(i).per || 4;
    for (let k = 0; k < n; k++) this._clickAt(when + k * beatLen, k === 0);
    return n * beatLen;
  }

  _silence(fade) {
    const now = this.ctx ? this.ctx.currentTime : 0;
    for (let s = 0; s < 6; s++) {
      const v = this.voices[s];
      if (!v) continue;
      this._release(v, now, fade);
      this.voices[s] = null;
    }
  }

  _release(v, when, fade) {
    try {
      v.gain.gain.cancelScheduledValues(when);
      v.gain.gain.setValueAtTime(v.gain.gain.value || 0.0001, when);
      v.gain.gain.setTargetAtTime(0, when, fade / 3);
      v.src.stop(when + fade * 3 + 0.02);
    } catch { /* already stopped */ }
  }

  /** The sample for string s sounding `pitch`: same string, nearest recorded fret. */
  _sample(s, pitch) {
    const notes = this.bank.notes;
    let f = pitch - OPEN[s];
    let best = null;
    for (let d = 0; d <= 24 && !best; d++) {
      for (const ff of [f - d, f + d]) {
        const n = notes.get(`${s}:${ff}`);
        if (n) { best = n; break; }
      }
    }
    if (!best) return null;
    return { n: best, detune: (pitch - (OPEN[s] + best.f)) * 100 - best.cents };
  }

  _note(e) {
    const c = this.ctx;
    const smp = this._sample(e.s, e.pitch);
    if (!smp) return;
    // strum: a few ms between the strings of a chord, the first one on the beat (down strokes
    // from the low string, up strokes from the high one, a little quicker), never quite even
    const strum = e.chordIndex >= 0 ? e.chordIndex * (e.up ? 0.006 : 0.007) + (e.chordIndex ? (Math.random() - 0.5) * 0.002 : 0) : 0;
    const when = Math.max(c.currentTime + 0.005, this._ctxTime(e.t) + strum);
    const endAt = this._ctxTime(e.end);
    const prev = this.voices[e.s];
    // the note this one slides from keeps sounding instead (no new pluck)
    if (e.slidDest && prev && prev.event && prev.event.slideTo === e) {
      prev.event = e;
      prev.endAt = endAt;
      this._scheduleEnd(prev, e, endAt);
      return;
    }
    if (prev && !e.letRing) { this._release(prev, when, e.legato ? 0.006 : 0.012); this.voices[e.s] = null; }
    const src = c.createBufferSource();
    src.buffer = this.bank.buffer;
    // no two notes quite alike: a few cents of finger pressure either way
    const detune = smp.detune + (Math.random() - 0.5) * 6;
    src.detune.value = detune;
    const g = c.createGain();
    // loudness: the note's own level when known, softer for ghost notes and hammer-ons
    let level = e.amp != null ? 0.35 + 0.65 * Math.min(1, Math.max(0, e.amp)) : 0.8;
    if (e.ghost) level *= 0.35;
    if (e.legato || e.slidDest) level *= 0.7;
    if (e.up) level *= 0.85;
    level *= 1 + (Math.random() - 0.5) * 0.12;
    let out = g;
    if (e.palmMute || e.dead) {
      const lp = c.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = e.dead ? 450 : 900;
      lp.Q.value = 0.5;
      g.connect(lp);
      out = lp;
    }
    out.connect(this.input);
    src.connect(g);
    // a hammer-on/pull-off starts after the pick attack
    const offset = smp.n.start + (e.legato || e.slidDest ? 0.035 : 0);
    const len = Math.max(0.05, smp.n.len - (offset - smp.n.start));
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(level, when + (e.legato ? 0.008 : 0.002));
    if (e.dead) g.gain.setTargetAtTime(0, when + 0.01, 0.02);
    else if (e.palmMute) g.gain.setTargetAtTime(0, when + 0.05, 0.12);
    src.start(when, offset, len);
    const v = { src, gain: g, event: e, endAt, level };
    this.voices[e.s] = v;
    // slide in from a few frets away
    if (e.slideIn === 1 || e.slideIn === 2) {
      const from = e.slideIn === 1 ? -300 : 300;
      src.detune.setValueAtTime(detune + from, when);
      src.detune.linearRampToValueAtTime(detune, when + 0.08);
    }
    this._bendAndVibrato(v, e, when, endAt, detune);
    this._scheduleEnd(v, e, endAt);
  }

  _bendAndVibrato(v, e, when, endAt, base) {
    const d = v.src.detune;
    const dur = Math.max(0.05, endAt - when);
    if (e.bend && e.bend.length) {
      d.setValueAtTime(base + e.bend[0][1], when);
      for (const [pos, cents] of e.bend.slice(1)) d.linearRampToValueAtTime(base + cents, when + pos * dur);
    }
    if (e.slideTo && e.slideOut) {
      // glide to the target fret just before it (shift and legato slides); slides out drop away
      const tgt = e.slideTo;
      const startGlide = Math.max(when + 0.02, this._ctxTime(tgt.t) - 0.09);
      const cents = (tgt.pitch - e.pitch) * 100;
      d.setValueAtTime(base, startGlide);
      d.linearRampToValueAtTime(base + cents, this._ctxTime(tgt.t));
    } else if (e.slideOut === 3 || e.slideOut === 4) {
      const t0 = Math.max(when + 0.02, endAt - 0.12);
      d.setValueAtTime(base, t0);
      d.linearRampToValueAtTime(base + (e.slideOut === 3 ? 500 : -500), endAt);
    }
    if (e.vibrato && dur > 0.2) {
      const lfo = this.ctx.createOscillator(), depth = this.ctx.createGain();
      lfo.frequency.value = 5.6;
      depth.gain.setValueAtTime(0, when);
      depth.gain.linearRampToValueAtTime(e.vibrato === 2 ? 45 : 22, when + Math.min(0.25, dur / 2));
      lfo.connect(depth).connect(d);
      lfo.start(when); lfo.stop(endAt + 0.3);
    }
  }

  _scheduleEnd(v, e, endAt) {
    // the string is damped when the note's time is up (let ring: when the next note on it comes)
    if (e.letRing || e.slideTo) return;
    const fade = e.palmMute ? 0.03 : 0.08;
    try {
      v.gain.gain.setTargetAtTime(0, Math.max(endAt, this.ctx.currentTime), fade / 3);
      v.src.stop(endAt + fade * 4 + 0.05);
    } catch { /* ignore */ }
  }
}

/**
 * The tab from `from` to `to` (seconds) rendered to an AudioBuffer (stereo), e.g. to save it as a
 * file or to test the sound.
 */
export async function renderTab(events, tone, from, to, sampleRate = 44100) {
  const OAC = self.OfflineAudioContext || self.webkitOfflineAudioContext;
  const ctx = new OAC(2, Math.ceil((to - from + 2.5) * sampleRate), sampleRate);
  const g = new GuitarSynth();
  g.ctx = ctx;
  g.offline = true;
  await g.setTone(tone);
  g._songAt0 = from; g._ctxAt0 = 0.05; g._rate = 1; g.paused = false;
  for (const e of events) if (e.t >= from && e.t < to) g._note(e);
  return ctx.startRendering();
}
