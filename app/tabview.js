// alphaTab wrapper: renders the tab, follows the recording with a cursor, plays the tab.
// The cursor always follows a media source: the recording (an <audio> element) or the tab
// itself played with recorded guitar notes (app/guitarsynth.js), which acts like one.
import * as alphaTab from '../vendor/alphatab/alphaTab.mjs';
import { GuitarSynth, scoreEvents, TONES } from './guitarsynth.js';

const BASE = new URL('../', import.meta.url).href;

export class TabView {
  constructor(el, { onNoteClick, onPlayState, onTime, ampOf } = {}) {
    this.el = el;
    this.onNoteClick = onNoteClick;
    this.onPlayState = onPlayState;
    this.onTime = onTime;
    this.ampOf = ampOf || null; // (noteId) -> 0..1 loudness of a transcribed note
    this.api = null;
    this.audio = new Audio();
    this.audio.preload = 'auto';
    this.audio.preservesPitch = true;
    this.synth = new GuitarSynth();
    this.media = this.audio;
    this.srcOffset = 0; // seconds of the recording before the source's t=0
    this.mode = 'mix';
    this.timer = 0;
    this.maps = null;
    this.part = 0;
    this.score = null;
    this.ready = false;
    this.speed = 1;
    this._wireMedia(this.audio);
    this._wireMedia(this.synth);
  }

  _settings() {
    const s = new alphaTab.Settings();
    s.core.fontDirectory = BASE + 'vendor/alphatab/font/';
    s.core.includeNoteBounds = true;
    s.core.useWorkers = true;
    s.core.engine = 'svg';
    s.display.staveProfile = alphaTab.StaveProfile.Tab;
    s.display.layoutMode = alphaTab.LayoutMode.Page;
    s.display.scale = this.el.clientWidth < 560 ? 0.82 : 1.0;
    s.display.justifyLastSystem = false;
    s.notation.rhythmMode = alphaTab.TabRhythmMode.ShowWithBars;
    // the page shows title / tuning itself; dynamics are not transcribed
    s.notation.elements.set(alphaTab.NotationElement.ScoreTitle, false);
    s.notation.elements.set(alphaTab.NotationElement.ScoreSubTitle, false);
    s.notation.elements.set(alphaTab.NotationElement.EffectDynamics, false);
    s.notation.elements.set(alphaTab.NotationElement.TrackNames, false);
    s.player.enablePlayer = true;
    s.player.enableCursor = true;
    s.player.enableUserInteraction = true;
    s.player.enableAnimatedBeatCursor = true;
    s.player.playerMode = alphaTab.PlayerMode.EnabledExternalMedia;
    s.player.scrollMode = alphaTab.ScrollMode.Continuous;
    s.player.scrollOffsetY = -120;
    return s;
  }

  async init() {}

  _create() {
    if (this.api) { try { this.api.destroy(); } catch { /* ignore */ } this.el.innerHTML = ''; }
    const api = new alphaTab.AlphaTabApi(this.el, this._settings());
    this.api = api;
    api.noteMouseDown.on((note) => this._noteClicked(note));
    api.error.on((e) => console.warn('alphaTab', e));
    const out = api.player && api.player.output;
    if (out) out.handler = this._handler();
  }

  _handler() {
    const self = this;
    return {
      get backingTrackDuration() { const d = self.media.duration; return Number.isFinite(d) ? (d + self.srcOffset) * 1000 : 0; },
      get playbackRate() { return self.media.playbackRate; },
      set playbackRate(v) { self.media.playbackRate = v; },
      get masterVolume() { return self.media.volume; },
      set masterVolume(v) { self.media.volume = v; },
      seekTo(time) { const t = (time * self.media.playbackRate) / 1000 - self.srcOffset; self.media.currentTime = Math.max(0, t); },
      play() { const p = self.media.play(); if (p && p.catch) p.catch(() => {}); },
      pause() { self.media.pause(); },
    };
  }

  _wireMedia(a) {
    const active = () => this.media === a;
    const update = () => {
      if (!this.api || !active()) return;
      const out = this.api.player && this.api.player.output;
      if (out && out.updatePosition) out.updatePosition(((a.currentTime + this.srcOffset) / a.playbackRate) * 1000);
      if (this.onTime) this.onTime(a.currentTime + this.srcOffset, (a.duration || 0) + this.srcOffset);
    };
    a.addEventListener('timeupdate', update);
    a.addEventListener('seeked', update);
    a.addEventListener('play', () => {
      if (!active()) return;
      clearInterval(this.timer);
      if (this.api) this.api.play();
      this.timer = setInterval(update, 50);
      if (this.onPlayState) this.onPlayState(true);
    });
    const stop = () => {
      if (!active()) return;
      if (this.api) this.api.pause();
      clearInterval(this.timer);
      update();
      if (this.onPlayState) this.onPlayState(false);
    };
    a.addEventListener('pause', stop);
    a.addEventListener('ended', stop);
    a.addEventListener('ratechange', () => { if (this.api && active()) this.api.playbackSpeed = a.playbackRate; });
  }

  /** Load a result. syncPoints: [{barIndex, barOccurence, barPosition, millisecondOffset}] */
  show(tex, maps, part, syncPoints) {
    this.maps = maps;
    this.part = part;
    this.syncPoints = syncPoints;
    if (!this.api) this._create();
    const settings = this.api.settings;
    const score = alphaTab.importer.ScoreLoader.loadAlphaTex(tex, settings);
    if (syncPoints && syncPoints.length) {
      try { score.applyFlatSyncPoints(syncPoints); } catch (e) { console.warn('sync points', e); }
    }
    this.score = score;
    this.api.renderScore(score, [part]);
    try { this.api.updateSyncPoints(); } catch { /* not ready yet */ }
    this._synthEvents();
  }

  // the tab's notes for the guitar player, on the recording's timeline
  _synthEvents() {
    if (!this.score) return;
    const amp = this.ampOf;
    const maps = this.maps && this.maps[this.part];
    const ampOf = amp && maps ? (beat, note) => {
      const bm = maps[beat.voice.bar.index] && maps[beat.voice.bar.index][beat.index];
      const id = bm ? bm[note.string - 1] : undefined;
      return id !== undefined ? amp(id) : null;
    } : null;
    const ev = scoreEvents(this.score, this.part, this.syncPoints, ampOf);
    this.synth.setEvents(ev, ev.length ? ev[ev.length - 1].end + 1.5 : 0);
    // beats for the metronome and the count-in: from the bars' timeline
    const beats = [], pos = [], per = [];
    for (const e of this._beatTimes()) { beats.push(e.t); pos.push(e.pos); per.push(e.per); }
    this.synth.beats = beats;
    this.synth.beatInfo = (i) => ({ pos: pos[Math.min(i, pos.length - 1)] || 0, per: per[Math.min(i, per.length - 1)] || 4 });
  }

  _beatTimes() {
    const out = [];
    const mbs = this.score.masterBars;
    const sp = new Map((this.syncPoints || []).map((p) => [p.barIndex, p.millisecondOffset / 1000]));
    const bpm = this.score.tempo || 120;
    let t = sp.has(0) ? sp.get(0) : 0;
    for (let i = 0; i < mbs.length; i++) {
      const per = mbs[i].timeSignatureNumerator || 4;
      const t0 = sp.has(i) ? sp.get(i) : t;
      const t1 = sp.has(i + 1) ? sp.get(i + 1) : t0 + per * (60 / bpm);
      for (let b = 0; b < per; b++) out.push({ t: t0 + ((t1 - t0) * b) / per, pos: b, per });
      t = t1;
    }
    return out;
  }

  /** The tab player's sound: 'acoustic' | 'nylon' | 'clean' | 'crunch' | 'lead'. */
  async setTone(tone) {
    await this.synth.setTone(tone);
  }

  setSource(mode, url, offset = 0) {
    const wasPlaying = this.isPlaying();
    this.pause();
    this.mode = mode;
    if (mode === 'synth') {
      this.media = this.synth;
      this.srcOffset = 0;
      this.synth.playbackRate = this.speed;
      // start loading the notes now, so pressing play doesn't wait for them
      this.synth.setTone(this.synth.tone).catch((e) => console.warn('guitar sounds', e));
    } else {
      this.media = this.audio;
      if (url && this.audio.src !== url) { this.audio.src = url; this.audio.load(); }
      this.srcOffset = offset;
      this.audio.playbackRate = this.speed;
    }
    if (!this.api) this._create();
    return wasPlaying;
  }

  isPlaying() { return !this.media.paused; }

  play() {
    const p = this.media.play();
    if (p && p.catch) p.catch((e) => console.warn(e));
  }

  pause() { this.media.pause(); }

  toggle() { if (this.isPlaying()) this.pause(); else this.play(); }

  setSpeed(r) {
    this.speed = r;
    this.media.playbackRate = r;
  }

  setLoop(on) {
    if (!this.api) return;
    this.api.isLooping = on;
    // a selected range is looped by alphaTab (it seeks back); the whole song by the source itself
    const whole = on && !this.api.playbackRange;
    this.audio.loop = whole;
    this.synth.loop = whole;
  }

  setMetronome(on) { this.synth.metronome = on; }

  setCountIn(on) { this.synth.countIn = on; }

  seekFraction(f) {
    const d = this.media.duration;
    if (Number.isFinite(d)) this.media.currentTime = f * d;
  }

  _noteClicked(note) {
    if (!this.onNoteClick || !this.maps) return;
    try {
      const barIdx = note.beat.voice.bar.index;
      const beatIdx = note.beat.index;
      const sIdx = note.string - 1; // alphaTab: 1 = lowest string
      const bm = this.maps[this.part] && this.maps[this.part][barIdx] && this.maps[this.part][barIdx][beatIdx];
      const id = bm ? bm[sIdx] : undefined;
      if (id !== undefined) this.onNoteClick(id, note);
    } catch (e) { console.warn(e); }
  }

  exportGp() {
    if (!this.score) return null;
    const exp = new alphaTab.exporter.Gp7Exporter();
    return exp.export(this.score, this.api.settings);
  }
}

export { alphaTab, TONES };
