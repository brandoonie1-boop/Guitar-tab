// alphaTab wrapper: renders the tab, follows the recording with a cursor, plays the tab.
import * as alphaTab from '../vendor/alphatab/alphaTab.mjs';

const BASE = new URL('../', import.meta.url).href;

export class TabView {
  constructor(el, { onNoteClick, onPlayState, onTime } = {}) {
    this.el = el;
    this.onNoteClick = onNoteClick;
    this.onPlayState = onPlayState;
    this.onTime = onTime;
    this.api = null;
    this.audio = new Audio();
    this.audio.preload = 'auto';
    this.audio.preservesPitch = true;
    this.srcOffset = 0; // seconds of the recording before the source's t=0
    this.mode = 'mix';
    this.timer = 0;
    this.maps = null;
    this.part = 0;
    this.score = null;
    this.ready = false;
    this._wireAudio();
  }

  _settings(mode) {
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
    s.player.soundFont = BASE + 'vendor/alphatab/soundfont/sonivox.sf3';
    s.player.playerMode = mode === 'synth' ? alphaTab.PlayerMode.EnabledSynthesizer : alphaTab.PlayerMode.EnabledExternalMedia;
    s.player.scrollMode = alphaTab.ScrollMode.Continuous;
    s.player.scrollOffsetY = -120;
    return s;
  }

  _create(mode) {
    if (this.api) { try { this.api.destroy(); } catch { /* ignore */ } this.el.innerHTML = ''; }
    const api = new alphaTab.AlphaTabApi(this.el, this._settings(mode));
    this.api = api;
    this.apiMode = mode;
    api.noteMouseDown.on((note) => this._noteClicked(note));
    api.playerStateChanged.on((e) => { if (this.onPlayState) this.onPlayState(e.state === 1); });
    api.playerPositionChanged.on((e) => { if (this.mode === 'synth' && this.onTime) this.onTime(e.currentTime / 1000, e.endTime / 1000); });
    api.error.on((e) => console.warn('alphaTab', e));
    if (mode !== 'synth') {
      const out = api.player && api.player.output;
      if (out) out.handler = this._handler();
    }
  }

  _handler() {
    const a = this.audio;
    const self = this;
    return {
      get backingTrackDuration() { const d = a.duration; return Number.isFinite(d) ? (d + self.srcOffset) * 1000 : 0; },
      get playbackRate() { return a.playbackRate; },
      set playbackRate(v) { a.playbackRate = v; },
      get masterVolume() { return a.volume; },
      set masterVolume(v) { a.volume = v; },
      seekTo(time) { const t = (time * a.playbackRate) / 1000 - self.srcOffset; a.currentTime = Math.max(0, t); },
      play() { a.play().catch(() => {}); },
      pause() { a.pause(); },
    };
  }

  _wireAudio() {
    const a = this.audio;
    const update = () => {
      if (!this.api || this.mode === 'synth') return;
      const out = this.api.player && this.api.player.output;
      if (out && out.updatePosition) out.updatePosition(((a.currentTime + this.srcOffset) / a.playbackRate) * 1000);
      if (this.onTime) this.onTime(a.currentTime + this.srcOffset, (a.duration || 0) + this.srcOffset);
    };
    a.addEventListener('timeupdate', update);
    a.addEventListener('seeked', update);
    a.addEventListener('play', () => {
      clearInterval(this.timer);
      if (this.api && this.mode !== 'synth') this.api.play();
      this.timer = setInterval(update, 50);
      if (this.onPlayState) this.onPlayState(true);
    });
    a.addEventListener('pause', () => {
      if (this.api && this.mode !== 'synth') this.api.pause();
      clearInterval(this.timer);
      if (this.onPlayState) this.onPlayState(false);
    });
    a.addEventListener('ended', () => {
      if (this.api && this.mode !== 'synth') this.api.pause();
      clearInterval(this.timer);
      if (this.onPlayState) this.onPlayState(false);
    });
    a.addEventListener('ratechange', () => { if (this.api && this.mode !== 'synth') this.api.playbackSpeed = a.playbackRate; });
  }

  /** Load a result. syncPoints: [{barIndex, barOccurence, barPosition, millisecondOffset}] */
  show(tex, maps, part, syncPoints) {
    this.maps = maps;
    this.part = part;
    this.syncPoints = syncPoints;
    if (!this.api) this._create(this.mode);
    const settings = this.api.settings;
    const score = alphaTab.importer.ScoreLoader.loadAlphaTex(tex, settings);
    if (syncPoints && syncPoints.length) {
      try { score.applyFlatSyncPoints(syncPoints); } catch (e) { console.warn('sync points', e); }
    }
    this.score = score;
    this.api.renderScore(score, [part]);
    try { this.api.updateSyncPoints(); } catch { /* not ready yet */ }
  }

  setSource(mode, url, offset = 0) {
    const wasPlaying = this.isPlaying();
    this.pause();
    const prev = this.mode;
    this.mode = mode;
    if (mode !== 'synth') {
      if (url && this.audio.src !== url) { this.audio.src = url; this.audio.load(); }
      this.srcOffset = offset;
    }
    const needSynth = mode === 'synth';
    if ((prev === 'synth') !== needSynth || !this.api) {
      this._create(mode);
      if (this.score) {
        this.api.renderScore(this.score, [this.part]);
        try { this.api.updateSyncPoints(); } catch { /* ignore */ }
      }
    }
    return wasPlaying;
  }

  isPlaying() {
    if (this.mode === 'synth') return this.api && this.api.playerState === 1;
    return !this.audio.paused;
  }

  play() {
    if (this.mode === 'synth') { if (this.api) this.api.play(); return; }
    this.audio.play().catch((e) => console.warn(e));
  }

  pause() {
    if (this.mode === 'synth') { if (this.api) this.api.pause(); return; }
    this.audio.pause();
  }

  toggle() { if (this.isPlaying()) this.pause(); else this.play(); }

  setSpeed(r) {
    if (this.mode === 'synth') { if (this.api) this.api.playbackSpeed = r; }
    else this.audio.playbackRate = r;
    this.speed = r;
  }

  setLoop(on) {
    if (!this.api) return;
    this.api.isLooping = on;
    if (this.mode !== 'synth') this.audio.loop = on && !this.api.playbackRange;
  }

  seekFraction(f) {
    if (this.mode === 'synth') { if (this.api) this.api.timePosition = f * this.api.endTime; return; }
    const d = this.audio.duration;
    if (Number.isFinite(d)) this.audio.currentTime = f * d;
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

  playNoteSound(midi) {
    // short preview of a pitch with the synth (only when in synth mode)
    if (this.api && this.mode === 'synth' && this.api.playNote) {
      // playNote needs an alphaTab Note; skipped for now
    }
  }
}

export { alphaTab };
