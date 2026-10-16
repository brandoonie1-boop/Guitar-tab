// Guitar Tab - app controller.
import { TUNINGS, tuningName, STANDARD } from '../engine/tunings.js';
import { midiName } from '../engine/dsp.js';
import { applySetup, layout, docOf, STAGES } from '../engine/pipeline.js';
import { toMidi } from '../engine/export.js';
import { decodeFile, wavBlob, makeZip, saveFile, copyText, inArtifact } from './files.js';
import {
  classifyLink, fetchMedia, captureDisplay, captureMic, makeAudioContext, recordStream, trimSilence, youtubePlayer,
  canCaptureTab, canListen, canEmbedPlayer, IS_IOS,
} from './capture.js';
import { resample } from '../engine/dsp.js';
import * as store from './store.js';
import { helperPing, helperDownload, HELPER_INSTALL_CMD, HELPER_SETUP_FILE, IS_WINDOWS } from './helper.js';

const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, '');
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
};
const fmtTime = (s) => {
  if (!Number.isFinite(s)) return '0:00';
  s = Math.max(0, s);
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
};
const parseTime = (str) => {
  str = (str || '').trim();
  if (!str) return null;
  const p = str.split(':').map(Number);
  if (p.some((x) => !Number.isFinite(x))) return null;
  return p.length === 1 ? p[0] : p.length === 2 ? p[0] * 60 + p[1] : p[0] * 3600 + p[1] * 60 + p[2];
};
const STRING_ORD = ['6th', '5th', '4th', '3rd', '2nd', '1st'];

const state = {
  file: null,
  audio: null, // decoded {left,right,sr,duration}: kept only until the next run takes it
  info: null, // {duration, channels} of the picked file
  opts: { isolate: 'guitar', split: 'auto', tuning: null, capo: null, quality: 'best' },
  worker: null,
  project: null, // {id, name, result, ...}
  part: 0,
  view: 'tab',
  undo: [],
  urls: {},
  tabview: null,
  wakeLock: null,
};

/* ---------------- environment ---------------- */
const ARTIFACT = inArtifact();
const caps = {
  threads: self.crossOriginIsolated ? Math.max(1, Math.min(8, navigator.hardwareConcurrency || 4)) : 1,
  gpu: !!navigator.gpu,
};

const IS_PHONE = /iPhone|iPod|Android.+Mobile/.test(navigator.userAgent);
const DEBUG = /[?&]debug\b/.test(location.search);
// tests only: ?forcegpu uses a software graphics adapter too (it is normally skipped as too slow)
const FORCE_GPU = /[?&]forcegpu\b/.test(location.search);
// Phones and small-memory computers default to the light isolation model: on GuitarSet with a
// band mixed in it found the same notes (F1 77.7 vs 77.9) with about 40% less engine memory.
const DEFAULT_MODEL = IS_PHONE || (navigator.deviceMemory && navigator.deviceMemory <= 4) ? 'lite' : 'full';
const APP_URL = 'https://guitar-tab.onrender.com/';
function gpuPref() {
  try { return localStorage.getItem('gt-gpu') || 'on'; } catch { return 'on'; }
}
function setupGpuRow() {
  // the GPU path needs WebGPU; if it fails or gives bad output the app falls back to the processor
  if (!navigator.gpu) return;
  $('gpuRow').hidden = false;
  $('engineRow').hidden = true;
  $('optGpu').value = gpuPref();
  $('optGpu').addEventListener('change', () => { try { localStorage.setItem('gt-gpu', $('optGpu').value); } catch { /* ignore */ } });
}
function engineInfo() {
  const box = $('gpuRow').hidden ? $('engineInfo2') : $('engineInfo');
  const t = caps.threads > 1 ? `${caps.threads} processor threads` : 'one processor thread';
  box.textContent = `Runs on this device using ${t}.` + (caps.threads === 1 && !ARTIFACT ? ' Reload once after the first visit to unlock more threads.' : '');
  if (ARTIFACT) {
    box.append(' For faster processing and offline use, open the installable version: ');
    const a = el('a', { href: APP_URL, target: '_blank', rel: 'noopener', text: 'guitar-tab.onrender.com' });
    box.append(a, '.');
  }
}

if (!ARTIFACT && 'serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js')
    .then(() => navigator.serviceWorker.ready)
    .then(() => {
      // the service worker turns on cross-origin isolation (multi-threaded processing); it applies after one reload
      let reloaded = false;
      try { reloaded = !!sessionStorage.getItem('gt-reloaded'); } catch { reloaded = true; }
      if (!self.crossOriginIsolated && !reloaded && !state.file) {
        try { sessionStorage.setItem('gt-reloaded', '1'); } catch { /* ignore */ }
        location.reload();
      }
    })
    .catch(() => {});
}

/* ---------------- start screen ---------------- */
function fillSelects() {
  const ts = $('optTuning');
  ts.append(el('option', { value: 'auto', text: 'Work it out for me' }));
  for (const t of TUNINGS) ts.append(el('option', { value: t.id, text: `${t.name} (${t.notes.map((m) => midiName(m, false)).join(' ')})` }));
  ts.append(el('option', { value: 'custom', text: 'Something else…' }));
  const cs = $('optCapo');
  cs.append(el('option', { value: 'auto', text: 'Work it out for me' }));
  cs.append(el('option', { value: '0', text: 'No capo' }));
  for (let i = 1; i <= 12; i++) cs.append(el('option', { value: String(i), text: `Fret ${i}` }));
  // custom tuning pickers (low string first)
  const ct = $('customTuning');
  STANDARD.forEach((m, i) => {
    const s = el('select', { 'aria-label': `${STRING_ORD[i]} string`, id: `ct${i}` });
    for (let p = m - 7; p <= m + 2; p++) s.append(el('option', { value: String(p), text: midiName(p, true) }));
    s.value = String(m);
    ct.append(s);
  });
  ts.addEventListener('change', () => {
    ct.hidden = ts.value !== 'custom';
  });
}

function readOpts() {
  const o = { ...state.opts };
  const tv = $('optTuning').value;
  if (tv === 'auto') o.tuning = null;
  else if (tv === 'custom') o.tuning = STANDARD.map((_, i) => Number($(`ct${i}`).value));
  else o.tuning = TUNINGS.find((t) => t.id === tv).notes.slice();
  const cv = $('optCapo').value;
  o.capo = cv === 'auto' ? null : Number(cv);
  o.split = $('optSplit').value;
  o.quality = $('optQuality').value;
  o.model = $('optModel').value;
  o.notes = $('optNotes').value;
  const a = parseTime($('rangeFrom').value), b = parseTime($('rangeTo').value);
  o.range = null;
  if (a != null || b != null) {
    const dur = state.info ? state.info.duration : 1e9;
    const from = Math.max(0, a ?? 0), to = Math.min(dur, b ?? dur);
    if (to - from >= 2) o.range = [from, to];
  }
  return o;
}

document.querySelectorAll('[data-opt="isolate"]').forEach((b) => b.addEventListener('click', () => {
  state.opts.isolate = b.dataset.val;
  document.querySelectorAll('[data-opt="isolate"]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
}));

async function pickFile(file) {
  if (!file) return;
  closeSource();
  $('startError').hidden = true;
  $('startBtn').disabled = true;
  $('picked').hidden = false;
  $('pickedName').textContent = file.name;
  $('pickedInfo').textContent = 'Reading…';
  state.audio = null;
  if (IS_PHONE && file.size > 400e6) {
    state.file = null; state.info = null;
    $('pickedInfo').textContent = '';
    showError(`This file is too big to open on a phone (${Math.round(file.size / 1e6)} MB). Save just the audio, or a shorter clip, and pick that.`);
    return;
  }
  try {
    setPicked(file, await decodeFile(file));
  } catch (e) {
    state.file = null; state.audio = null; state.info = null;
    $('pickedInfo').textContent = '';
    showError("This file couldn't be read. Try an MP3, M4A, WAV or a phone video. (" + (e.message || e) + ')');
  }
}

// a song ready to go: the file (for playback and saving) and its decoded samples
function setPicked(file, a) {
  state.file = file;
  state.audio = a;
  state.info = { duration: a.duration, channels: a.channels || (a.right ? 2 : 1) };
  $('picked').hidden = false;
  $('pickedName').textContent = file.name;
  $('pickedInfo').textContent = `${fmtTime(a.duration)} · ${state.info.channels > 1 ? 'stereo' : 'mono'}`;
  $('startBtn').disabled = false;
  $('startError').hidden = true;
  if (a.duration > 600) showError('This file is over 10 minutes. Pick a section below so it finishes in a reasonable time.');
  else if (IS_PHONE && a.duration > 420) showError('Long song for a phone. If it stops partway, pick a section below.');
}

/* ---------------- links and recording ---------------- */
const panel = $('sourcePanel');
let source = null; // {player, rec, ctx}

function closeSource() {
  if (source) {
    if (source.cancel) source.cancel();
    if (source.player) source.player.destroy();
    if (source.rec) source.rec.cancel();
    if (source.ctx) source.ctx.close().catch(() => {});
  }
  source = null;
  panel.hidden = true;
  panel.innerHTML = '';
}
function openPanel(title, ...kids) {
  closeSource();
  source = {};
  panel.hidden = false;
  panel.append(el('h2', { text: title }), ...kids);
}
const btnRow = (...b) => el('div', { class: 'range-row' }, ...b);
const button = (text, onclick, primary = false) => el('button', { class: primary ? 'btn primary' : 'btn', type: 'button', text, onclick });

$('linkForm').addEventListener('submit', (e) => {
  e.preventDefault();
  useLink($('linkInput').value);
});
$('linkInput').addEventListener('paste', () => setTimeout(() => {
  const v = $('linkInput').value.trim();
  if (/^https?:\/\/\S+$/.test(v)) useLink(v);
}, 0));
$('recordBtn').addEventListener('click', () => recordPanel());

// Inside the Claude app the page can't record or reach other sites: send people to the app.
function artifactPanel(what) {
  const a = el('a', { href: APP_URL, target: '_blank', rel: 'noopener', text: 'guitar-tab.onrender.com' });
  openPanel(what, el('p', {}, document.createTextNode('This works in the installable version of the app: '), a,
    document.createTextNode('. There you can paste YouTube and other links, record what is playing, or listen with the microphone. Here inside Claude, use Choose a file.')), btnRow(button('Close', closeSource)));
}

function useLink(text) {
  $('startError').hidden = true;
  const L = classifyLink(text);
  if (L.kind === 'invalid') { showError("That doesn't look like a link. Copy the whole address, starting with https://"); return; }
  if (ARTIFACT && L.kind !== 'direct') { artifactPanel(L.kind === 'youtube' ? 'YouTube' : (L.service || 'Links')); return; }
  if (L.kind === 'direct') { downloadLink(L); return; }
  // on a computer the link helper fetches the song itself
  if (!IS_PHONE) { helperLink(L); return; }
  recordLinkPanel(L);
}

// the ways to record a linked song while it plays (no helper, or on a phone)
function recordLinkPanel(L) {
  if (L.kind === 'youtube') youtubePanel(L);
  else streamPanel(L);
}

const linkName = (L) => (L.kind === 'youtube' ? 'YouTube' : (L.service || 'Link'));

async function helperLink(L) {
  const st = el('p', { text: 'Looking for the link helper on this computer…' });
  openPanel(linkName(L), st, btnRow(button('Cancel', closeSource)));
  const mine = source;
  const info = await helperPing();
  if (source !== mine) return;
  if (!info) { helperInstallPanel(L); return; }
  let cancelled = false;
  mine.cancel = () => { cancelled = true; };
  st.textContent = 'Getting the song…';
  try {
    const file = await helperDownload(L.url, {
      onStatus: (s) => { st.textContent = `Getting ${s.title ? `"${s.title}"` : 'the song'}… ${Math.round((s.progress || 0) * 100)}%`; },
      isCancelled: () => cancelled || source !== mine,
    });
    closeSource();
    await pickFile(file);
    // straight on to the tab (long songs wait for a section to be picked)
    if (state.file === file && state.info && state.info.duration <= 600) {
      toast('Got it. Making the tab.');
      start();
    }
  } catch (err) {
    if (err && err.code === 'cancelled') return;
    if (source !== mine) return;
    openPanel("Couldn't get that song", el('p', { text: (err && err.message) || String(err) }),
      btnRow(button('Try again', () => helperLink(L), true), button('Record it instead', () => recordLinkPanel(L)), button('Close', closeSource)));
  }
}

function helperInstallPanel(L) {
  if (!IS_WINDOWS) {
    openPanel(linkName(L),
      el('p', { text: 'Links go straight in on a Windows computer that has the Guitar Tab link helper. Here, record the song while it plays instead.' }),
      btnRow(button('Record it', () => recordLinkPanel(L), true), button('Close', closeSource)));
    return;
  }
  const copyBtn = button('Copy setup command', async () => {
    toast((await copyText(HELPER_INSTALL_CMD)) ? 'Copied. Paste it into PowerShell and press Enter.' : 'Select the command below and copy it.');
  }, true);
  const cmd = el('pre', { class: 'ascii', style: 'white-space:pre-wrap;word-break:break-all;margin:0;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--surface)', text: HELPER_INSTALL_CMD });
  const setup = el('a', { class: 'btn', href: HELPER_SETUP_FILE, download: 'GuitarTabHelperSetup.cmd', text: 'Download setup file' });
  openPanel('Paste links straight in',
    el('p', { text: 'This computer needs the free Guitar Tab link helper once. It grabs the song from YouTube, SoundCloud, Bandcamp, Spotify (found on YouTube) and most other sites with yt-dlp, over your own internet connection, and hands it to this page. Setting it up takes about a minute:' }),
    el('ol', {},
      el('li', { text: 'Press Copy setup command.' }),
      el('li', { text: 'Open PowerShell: press the Windows key, type PowerShell, press Enter.' }),
      el('li', { text: 'Right-click in the PowerShell window to paste, then press Enter. It installs the helper and starts it.' }),
      el('li', { text: 'Come back here and press Try again. If the browser asks to let this site use apps on this device, choose Allow.' })),
    cmd,
    btnRow(copyBtn, button('Try again', () => helperLink(L)), setup),
    el('p', { class: 'small muted', text: 'Or download the setup file and double-click it (Windows may ask you to confirm). Next time, start the helper from the Guitar Tab helper shortcut on your desktop. Already running but this keeps showing? Click the icon at the left of the address bar and allow this site to reach apps on this device.' }),
    btnRow(button('Record it instead', () => recordLinkPanel(L)), button('Close', closeSource)));
}

async function downloadLink(L) {
  const st = el('p', { text: 'Downloading…' });
  openPanel('Downloading the file', st);
  try {
    const file = await fetchMedia(L.url, {
      maxBytes: IS_PHONE ? 400e6 : 2e9,
      onProgress: (f, n) => { st.textContent = `Downloading… ${f ? Math.round(f * 100) + '%' : (n / 1e6).toFixed(1) + ' MB'}`; },
    });
    await pickFile(file);
  } catch (err) {
    if (err.code === 'blocked' && !IS_PHONE && !ARTIFACT) { helperLink(L); return; }
    if (err.code === 'blocked') {
      openPanel(`${L.share || 'This site'} won't hand the file to other pages`,
        el('p', { text: 'Download the file to this device first, then use Choose a file. Or play it and record it with Record.' }),
        btnRow(button('Open the link', () => window.open(L.url, '_blank', 'noopener')), button('Close', closeSource)));
    } else {
      openPanel("Couldn't use that link", el('p', { text: err.message }), btnRow(button('Close', closeSource)));
    }
  }
}

function phoneSteps(service) {
  return el('ol', {},
    el('li', { text: `Screen recording (best sound): open Control Center, start Screen Recording, play the song in ${service === 'YouTube' ? 'the YouTube app' : service}, then stop the recording. Come back, tap Choose a file and pick the recording from Photos.` }),
    el('li', { text: 'Or listen: play the song out loud on another device or speaker, hold this phone close to it and tap Listen.' }));
}

function youtubePanel(L) {
  if (canCaptureTab() && canEmbedPlayer()) {
    const frame = el('div', { class: 'player-frame' });
    const note = el('p', { text: "YouTube doesn't let apps download its audio, so the app records the video while it plays here. Press Record the song, choose This tab when your browser asks what to share, and keep Share tab audio on. The video plays from the start and recording stops when it ends." });
    const status = el('div');
    openPanel('YouTube', frame, note, status,
      btnRow(button('Record the song', () => recordYouTube(L, status), true), button('Close', closeSource)));
    source.player = youtubePlayer(frame, L.id, { start: L.start });
    source.player.onError = (code) => {
      if (code === 101 || code === 150 || code === 153) {
        status.textContent = '';
        otherTabPanel(L, 'YouTube', "This video's owner doesn't allow it to play inside other sites.");
      }
    };
    return;
  }
  if (canCaptureTab()) { otherTabPanel(L, 'YouTube'); return; }
  phonePanel(L, 'YouTube');
}

function streamPanel(L) {
  const service = L.service || 'that site';
  const why = L.protected
    ? `${service} protects its audio, so no app can download it.`
    : `${service} doesn't let other pages download its audio.`;
  if (canCaptureTab()) otherTabPanel(L, service, why);
  else phonePanel(L, service, why);
}

function otherTabPanel(L, service, why = '') {
  openPanel(service,
    el('p', { text: `${why} Record it while it plays instead:`.trim() }),
    el('ol', {},
      el('li', { text: 'Open the song (button below) and get it ready to play.' }),
      el('li', { text: 'Come back here, press Record, and choose that tab with Share tab audio on. (For a desktop app like Spotify on Windows, choose Entire screen with Share system audio on.)' }),
      el('li', { text: 'Play the song from the start. Press Stop when it ends.' })),
    btnRow(button('Open the song', () => window.open(L.url, '_blank', 'noopener')), button('Record', () => startCapture('display', null), true), button('Close', closeSource)));
}

function phonePanel(L, service, why = '') {
  const kids = [el('p', { text: `${why || `${service} doesn't let other apps take its audio.`} Two ways to get the song in:` }), phoneSteps(service),
    el('p', { class: 'small muted', text: 'On a Windows computer with the Guitar Tab link helper, links go straight in.' })];
  const buttons = [];
  if (canListen()) buttons.push(button('Listen', () => startCapture('mic', null), true));
  buttons.push(button('Close', closeSource));
  openPanel(service, ...kids, btnRow(...buttons));
}

function recordPanel() {
  if (ARTIFACT) { artifactPanel('Recording'); return; }
  if (canCaptureTab()) {
    openPanel('Record what is playing',
      el('p', { text: 'Press Start, then choose the tab that plays the song with Share tab audio on (or Entire screen with Share system audio on, for a desktop app). Play the song from the start and press Stop when it ends.' }),
      btnRow(button('Start', () => startCapture('display', null), true), canListen() ? button('Use the microphone', () => startCapture('mic', null)) : null, button('Close', closeSource)));
  } else {
    const kids = [el('p', { text: 'Play the song out loud on another device or speaker, hold this phone close to it, and press Listen. A quiet room works best. Press Stop when the song ends.' })];
    if (IS_IOS) kids.push(el('p', { class: 'small muted', text: 'For the best sound on iPhone, use Screen Recording while the song plays in its app instead, then pick the recording with Choose a file.' }));
    openPanel('Listen to a song', ...kids, btnRow(canListen() ? button('Listen', () => startCapture('mic', null), true) : null, button('Close', closeSource)));
  }
}

function recordingView(title, onStop) {
  const time = el('span', { class: 'mono', text: '0:00' });
  const bar = el('i');
  const hint = el('p', { class: 'small muted', text: '' });
  const view = el('div', { style: 'display:grid;gap:8px' },
    el('div', { class: 'rec-status' }, el('span', { class: 'rec-dot', 'aria-hidden': 'true' }), el('strong', { text: title }), time,
      el('div', { class: 'meter', 'aria-hidden': 'true' }, bar)),
    hint,
    btnRow(button('Stop', onStop, true), button('Cancel', () => { closeSource(); toast('Recording cancelled'); })));
  return {
    view,
    level(peak, secs) {
      bar.style.width = `${Math.min(100, Math.round(100 * Math.sqrt(peak)))}%`;
      time.textContent = fmtTime(secs);
    },
    hint(t) { hint.textContent = t; },
  };
}

// Record from the screen-share picker (a tab / the computer's sound) or the microphone.
async function startCapture(kind, title, { keepPanel = false } = {}) {
  const ctx = makeAudioContext(); // inside the tap, for iPhone
  let stream;
  try {
    stream = kind === 'mic' ? await captureMic() : await captureDisplay({ thisTab: false });
  } catch (err) {
    ctx.close().catch(() => {});
    if (err && err.name === 'NotAllowedError') { toast(kind === 'mic' ? 'The microphone was not allowed.' : 'Recording was cancelled.'); return; }
    showError((err && err.message) || String(err));
    return;
  }
  const label = kind === 'mic' ? 'Listening' : 'Recording';
  let stopping = false;
  const finish = async () => {
    if (stopping || !source || !source.rec) return;
    stopping = true;
    const rec = source.rec;
    source.rec = null;
    const r = await rec.stop();
    await useRecording(r, title || (kind === 'mic' ? 'Listened recording' : 'Recording'));
  };
  const ui = recordingView(label, finish);
  if (!keepPanel) openPanel(label, ui.view);
  else panel.append(ui.view);
  source.ctx = ctx;
  source.rec = await recordStream(stream, { ctx, onLevel: ui.level, onEnded: finish });
  source.ctx = null; // owned by the recorder now
  if (kind === 'mic') ui.hint('Play the song now. The level bar should move with the music.');
  else ui.hint('Play the song now. If the level bar stays still, no sound is being shared.');
  return { ui, finish };
}

async function recordYouTube(L, status) {
  const player = source && source.player;
  if (!player) return;
  const ctx = makeAudioContext();
  let stream;
  try {
    stream = await captureDisplay({ thisTab: true });
  } catch (err) {
    ctx.close().catch(() => {});
    if (err && err.name === 'NotAllowedError') { toast('Recording was cancelled.'); return; }
    showError((err && err.message) || String(err));
    return;
  }
  let stopping = false, startedAt = -1;
  const finish = async () => {
    if (stopping || !source || !source.rec) return;
    stopping = true;
    player.pause();
    const rec = source.rec;
    source.rec = null;
    let r = await rec.stop();
    // anything recorded before the video itself started (an ad, a pause) is cut using the video's length
    const dur = player.info.duration;
    if (dur > 5 && r.duration > dur + 2) {
      const keep = Math.round((dur + 1) * r.sr);
      r = { left: r.left.slice(r.left.length - keep), right: r.right.slice(r.right.length - keep), sr: r.sr, duration: keep / r.sr };
    }
    await useRecording(r, player.info.title || 'YouTube video');
  };
  const ui = recordingView('Recording', finish);
  status.replaceChildren(ui.view);
  source.rec = await recordStream(stream, { ctx, onLevel: ui.level, onEnded: finish });
  player.onInfo = (info) => {
    if (info.state === 1 && startedAt < 0) startedAt = source && source.rec ? source.rec.seconds() : 0;
    if (info.state === 0) finish(); // ended
  };
  player.seek(L.start || 0);
  player.unmute();
  player.play();
  setTimeout(() => { if (player.info.state !== 1 && !stopping) ui.hint('If the video has not started, press play on it.'); }, 3500);
}

async function useRecording(r, title) {
  let t = trimSilence(r);
  if (!t) {
    openPanel('Nothing was recorded', el('p', { text: 'The recording was silent. Make sure the song is playing and, when sharing a tab, that Share tab audio is on.' }), btnRow(button('Close', closeSource)));
    return;
  }
  if (t.sr !== 44100) {
    t = { left: resample(t.left, t.sr, 44100), right: resample(t.right, t.sr, 44100), sr: 44100, duration: t.duration };
    t.duration = t.left.length / 44100;
  }
  closeSource();
  const safe = String(title).replace(/[\\/:*?"<>|]+/g, ' ').trim().slice(0, 80) || 'Recording';
  const file = new File([wavBlob([t.left, t.right], 44100)], `${safe}.wav`, { type: 'audio/wav' });
  setPicked(file, { left: t.left, right: t.right, sr: 44100, duration: t.duration, channels: 2 });
  toast(`Recorded ${fmtTime(t.duration)}. Press Make the tab.`);
}

function showError(msg) { const e = $('startError'); e.textContent = msg; e.hidden = false; }

$('fileInput').addEventListener('change', (e) => pickFile(e.target.files[0]));
const drop = $('drop');
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files && e.dataTransfer.files[0]; if (f) pickFile(f); });

/* ---------------- screens ---------------- */
function show(screen) {
  for (const s of ['screenStart', 'screenWork', 'screenResult']) $(s).hidden = s !== screen;
  $('navNew').hidden = screen === 'screenStart';
  $('player').hidden = screen !== 'screenResult';
  closeSheet();
  window.scrollTo(0, 0);
}
$('navNew').addEventListener('click', () => { stopPlayback(); show('screenStart'); refreshLibrary(); });
$('navLibrary').addEventListener('click', () => { stopPlayback(); show('screenStart'); refreshLibrary(); setTimeout(() => $('libTitle').scrollIntoView({ behavior: 'smooth' }), 50); });

/* ---------------- processing ---------------- */
const STEP_ORDER = ['load', 'separate', 'rhythm', 'split', 'notes', 'setup', 'tab'];
let workStart = 0, lastFrac = 0;
const stageClock = {}; // stage -> performance.now() when it started (for the time left)
const STAGE_SPAN = (() => { const o = {}; let a = 0; for (const [k, w] of STAGES) { o[k] = [a, w]; a += w; } return o; })();

function setStep(stage) {
  const idx = STEP_ORDER.indexOf(stage === 'prepare' ? 'load' : stage);
  document.querySelectorAll('#workSteps li').forEach((li, i) => {
    li.classList.toggle('done', i < idx);
    li.classList.toggle('now', i === idx);
  });
}

function onProgress(p) {
  setStep(p.stage);
  const frac = p.stage === 'load' ? 0 : p.frac;
  lastFrac = Math.max(lastFrac, frac);
  $('workBar').style.width = `${(100 * Math.max(0.01, lastFrac)).toFixed(1)}%`;
  let txt = p.label || '';
  if (p.stage === 'load' && p.frac > 0) txt += ` · ${Math.round(p.frac * 100)}%`;
  // the two long stages say how long they still need
  if (p.stage === 'separate' || p.stage === 'notes') {
    if (!stageClock[p.stage]) stageClock[p.stage] = performance.now();
    const el2 = (performance.now() - stageClock[p.stage]) / 1000;
    const [s0, w] = STAGE_SPAN[p.stage];
    const f = (p.frac - s0) / w;
    if (f > 0.08 && f < 1) {
      const remain = (el2 / f) * (1 - f) * 1.15;
      txt += ` · about ${remain > 90 ? Math.round(remain / 60) + ' min' : Math.max(5, Math.round(remain / 5) * 5) + ' s'} left`;
    }
  }
  if (DEBUG && p.wasmMB) txt += ` · engine memory ${p.wasmMB} MB`;
  $('workStatus').textContent = txt;
}

async function keepAwake(on) {
  try {
    if (on && 'wakeLock' in navigator) state.wakeLock = await navigator.wakeLock.request('screen');
    else if (!on && state.wakeLock) { await state.wakeLock.release(); state.wakeLock = null; }
  } catch { state.wakeLock = null; }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && !$('screenWork').hidden) keepAwake(true);
});

$('startBtn').addEventListener('click', () => start());
$('cancelBtn').addEventListener('click', () => {
  // ending the worker stops at once, even in the middle of a long engine call
  if (state.job) state.job.cancel();
});

/*
 * Each run gets a fresh worker that is ended afterwards (the engine's memory can only grow,
 * so ending the worker is the only way to hand it back). If a run fails because the graphics
 * chip didn't work or memory ran out, it is retried automatically with lighter settings:
 *   graphics chip -> processor -> lighter isolation model.
 * A marker in localStorage notices runs that never finished because the browser ended the page.
 */
const RUN_KEY = 'gt-run';
function setRunMarker(v) { try { if (v) localStorage.setItem(RUN_KEY, JSON.stringify(v)); else localStorage.removeItem(RUN_KEY); } catch { /* ignore */ } }
function getRunMarker() { try { return JSON.parse(localStorage.getItem(RUN_KEY) || 'null'); } catch { return null; } }
function modelPref() { try { return localStorage.getItem('gt-model') || DEFAULT_MODEL; } catch { return DEFAULT_MODEL; } }
function setModelPref(v) { try { localStorage.setItem('gt-model', v); } catch { /* ignore */ } $('optModel').value = v; }
function notesPref() { try { return localStorage.getItem('gt-notes') || 'hr'; } catch { return 'hr'; } }
function setNotesPref(v) { try { localStorage.setItem('gt-notes', v); } catch { /* ignore */ } $('optNotes').value = v; }
function setGpuPref(v) { try { localStorage.setItem('gt-gpu', v); } catch { /* ignore */ } $('optGpu').value = v; }
const looksLikeMemory = (msg) => /memory|oom|allocation|array buffer|rangeerror|aborted|out of bounds/i.test(msg || '');

async function takeAudio() {
  // the decoded samples are handed to the worker (not copied); decode again for another run
  if (state.audio) { const a = state.audio; state.audio = null; return a; }
  return decodeFile(state.file);
}

// Run one job ('separate' or 'analyze') in a fresh worker that is ended when the job is done.
// Resolves with the job's final message; rejects with {message, gpuFailed?, crash?}.
function runJob(type, payload, transfer) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    let done = false;
    const end = () => {
      done = true;
      try { w.terminate(); } catch { /* ignore */ }
      if (state.job === job) state.job = null;
    };
    const job = { cancel: () => { if (!done) { end(); reject({ message: 'cancelled' }); } } };
    state.job = job;
    w.addEventListener('error', (ev) => {
      // the worker itself died (for example the engine ran out of memory)
      ev.preventDefault();
      if (done) return;
      end();
      reject({ message: ev.message || 'the processing worker stopped', crash: true });
    });
    w.onmessage = (e) => {
      const m = e.data;
      if (done) return;
      if (m.type === 'progress') onProgress(m);
      else if (m.type === 'engine') state.backend = m;
      else if (m.type === 'error') { end(); reject(m); }
      else if (m.type === 'stems' || m.type === 'result') { end(); resolve(m); }
    };
    w.postMessage({ type, ...payload }, transfer);
  });
}

const buffersOf = (st) => {
  const t = [st.gL.buffer];
  if (st.gR && st.gR !== st.gL) t.push(st.gR.buffer);
  if (st.mix22) t.push(st.mix22.buffer);
  return t;
};

async function start(mode = null) {
  if (!state.file) return;
  const opts = readOpts();
  if (!mode) {
    mode = {
      gpu: !$('gpuRow').hidden && $('optGpu').value === 'on',
      lite: opts.model === 'lite',
      notes: opts.notes,
      // each engine thread costs memory too; phones gain little from more than 4
      threads: IS_PHONE ? 4 : 8,
    };
  }
  show('screenWork');
  $('workTitle').textContent = `Working on ${state.file.name}`;
  $('workBar').style.width = '1%';
  $('workStatus').textContent = 'Reading the file';
  lastFrac = 0; workStart = performance.now();
  for (const k of Object.keys(stageClock)) delete stageClock[k];
  setStep('load');
  keepAwake(true);
  if (state.job) state.job.cancel();
  let a;
  try {
    a = await takeAudio();
  } catch (e) {
    keepAwake(false);
    show('screenStart');
    showError("This file couldn't be read again. (" + (e.message || e) + ')');
    return;
  }
  // keep the recording for playback: the original file if this browser can play it, otherwise a WAV
  const canPlay = state.file.type && new Audio().canPlayType(state.file.type);
  const mixBlob = canPlay ? state.file : wavBlob(a.right ? [a.left, a.right] : [a.left]);
  // only the chosen section goes to the worker
  let left = a.left, right = a.right, offset = 0;
  if (opts.range) {
    const i0 = Math.max(0, Math.floor(opts.range[0] * a.sr)), i1 = Math.min(left.length, Math.ceil(opts.range[1] * a.sr));
    left = left.slice(i0, i1);
    right = right ? right.slice(i0, i1) : null;
    offset = i0 / a.sr;
  }
  const sr = a.sr, name = state.file.name;
  a = null;
  const runOpts = {
    threads: mode.threads, gpu: mode.gpu, lite: mode.lite, notesModel: mode.notes || 'hr',
    isolate: opts.isolate, split: opts.split, tuning: opts.tuning, capo: opts.capo, range: null,
    overlap: opts.quality === 'fast' ? 0.1 : 0.25,
    hrHop: opts.quality === 'fast' ? 750 : 500,
    preferTunings: favoriteTunings(),
    debugMem: DEBUG,
    forceGpu: FORCE_GPU,
  };
  const marker = { t: Date.now(), gpu: mode.gpu, lite: mode.lite, notes: mode.notes || 'hr', name, phase: 'separate', hidden: document.visibilityState === 'hidden' };
  // 1. isolate the guitar (its own worker, ended afterwards so the model's memory is freed)
  let st;
  if (opts.isolate === 'off') {
    st = { gL: left, gR: right, mix22: null, stereo: !!right, sr, offset, duration: left.length / sr, name };
  } else {
    setRunMarker(marker);
    try {
      const m = await runJob('separate', { audio: { left, right, sr, name, offset }, opts: runOpts },
        [left.buffer].concat(right ? [right.buffer] : []));
      st = m.stems;
    } catch (m) {
      setRunMarker(null);
      runFailed(mode, m, 'separate');
      return;
    }
  }
  left = null; right = null;
  // 2. everything else in a fresh worker
  setRunMarker({ ...marker, phase: 'analyze' });
  let res;
  try {
    res = await runJob('analyze', { stems: st, opts: runOpts }, buffersOf(st));
  } catch (m) {
    setRunMarker(null);
    runFailed(mode, m, 'analyze');
    return;
  }
  st = null;
  setRunMarker(null);
  keepAwake(false);
  clearNotice();
  res.ms = performance.now() - workStart;
  res.backend = state.backend ? state.backend.backend : 'none';
  res.lite = opts.isolate !== 'off' && mode.lite;
  try {
    await finish(res, mixBlob, opts);
  } catch (err) {
    console.error(err);
    show('screenStart');
    showError('Something went wrong while showing the tab: ' + (err.message || err));
  }
}

// Decide how to retry a failed run, from lightest change to heaviest.
function runFailed(mode, m, phase) {
  console.warn('run failed in', phase, ':', m.message, m.reason || '');
  if (m.message === 'cancelled') {
    keepAwake(false);
    show('screenStart');
    showError('Stopped.');
    return;
  }
  const memory = m.crash || looksLikeMemory(m.message);
  if (phase === 'separate') {
    if (m.gpuFailed || (mode.gpu && memory)) {
      // remember that the graphics chip didn't work here, so next time goes straight to the processor
      setGpuPref('off');
      toast('The graphics chip did not work here. Using the processor instead.');
      start({ ...mode, gpu: false });
      return;
    }
    if (!mode.lite && memory) {
      setModelPref('lite');
      toast('Ran out of memory. Trying again with the lighter isolation model.');
      start({ ...mode, lite: true });
      return;
    }
    if (m.crash && mode.threads > 1) {
      toast('Trying again with a single thread.');
      start({ ...mode, threads: 1 });
      return;
    }
  }
  if (phase === 'analyze' && memory && (mode.notes || 'hr') !== 'basic') {
    setNotesPref('basic');
    toast('Ran out of memory while finding the notes. Trying again with the light note detection.');
    start({ ...mode, notes: 'basic' });
    return;
  }
  keepAwake(false);
  show('screenStart');
  const tip = memory ? ' This device ran out of memory. Pick a shorter part of the song below and try again.' : '';
  showError(`Something went wrong: ${m.message}.${tip}`);
}

// A run that was still marked as running when the page loaded never finished: the browser
// ended the page (usually for using too much memory) or it was closed. Lighten the next run.
function checkLastRun() {
  const r = getRunMarker();
  setRunMarker(null);
  if (!r || Date.now() - r.t > 12 * 3600e3) return;
  let msg;
  if (r.hidden) {
    msg = `The last run (${r.name}) stopped when the app went into the background. Keep the screen on and the app open until it finishes.`;
  } else if (r.phase === 'analyze' && r.notes !== 'basic') {
    setNotesPref('basic');
    msg = `The last run (${r.name}) stopped while finding the notes, most likely because the browser ran out of memory. The light note detection is now selected below (or pick a shorter part of the song).`;
  } else if (r.phase === 'analyze') {
    msg = `The last run (${r.name}) stopped while writing the tab, most likely because the browser ran out of memory. Pick a shorter part of the song below, or close other apps and tabs, then try again.`;
  } else if (r.gpu) {
    setGpuPref('off');
    msg = `The last run (${r.name}) stopped before it finished. It will use the processor instead of the graphics chip this time.`;
  } else if (!r.lite) {
    setModelPref('lite');
    msg = `The last run (${r.name}) stopped before it finished, most likely because the browser ran out of memory. The lighter isolation model is now selected below.`;
  } else {
    msg = `The last run (${r.name}) stopped before it finished, most likely because the browser ran out of memory. Pick a shorter part of the song below, or close other apps and tabs, then try again.`;
  }
  const n = $('runNotice');
  n.textContent = msg;
  n.hidden = false;
}
function clearNotice() { $('runNotice').hidden = true; }
document.addEventListener('visibilitychange', () => {
  const r = getRunMarker();
  if (r && state.job) setRunMarker({ ...r, hidden: document.visibilityState === 'hidden' });
});

const DEFAULT_FAVS = ['38,45,52,57,59,64']; // DAEABE

function favoriteTunings() {
  try {
    const v = localStorage.getItem('gt-fav-tunings');
    return v ? JSON.parse(v) : DEFAULT_FAVS.slice();
  } catch { return DEFAULT_FAVS.slice(); }
}
function setFavorites(list) {
  try { localStorage.setItem('gt-fav-tunings', JSON.stringify(list.slice(0, 8))); } catch { /* ignore */ }
  renderFavorites();
}
function rememberTuning(t) {
  const key = t.join(',');
  if (key === STANDARD.join(',')) return;
  const f = favoriteTunings().filter((k) => k !== key);
  f.unshift(key);
  setFavorites(f);
}
function renderFavorites() {
  const box = $('favList');
  if (!box) return;
  box.innerHTML = '';
  const favs = favoriteTunings();
  if (!favs.length) box.append(el('span', { class: 'small muted', text: 'None yet.' }));
  for (const k of favs) {
    const notes = k.split(',').map(Number);
    box.append(el('button', { class: 'chip', type: 'button', title: 'Remove', 'aria-label': `Remove ${tuningName(notes)}`,
      onclick: () => setFavorites(favoriteTunings().filter((x) => x !== k)),
      text: `${tuningName(notes)} ✕` }));
  }
}
function fillFavAdd() {
  const box = $('favAdd');
  if (!box) return;
  const sel = el('select', { id: 'favSel', 'aria-label': 'Add a tuning you use' });
  sel.append(el('option', { value: '', text: 'Add a tuning…' }));
  for (const t of TUNINGS) if (t.id !== 'std') sel.append(el('option', { value: t.notes.join(','), text: t.name }));
  sel.addEventListener('change', () => {
    if (!sel.value) return;
    const f = favoriteTunings().filter((k) => k !== sel.value);
    f.push(sel.value);
    setFavorites(f);
    sel.value = '';
  });
  box.append(sel);
}

function restoreTech(result) {
  for (const p of result.parts) if (Array.isArray(p.tech)) p.tech = new Map(p.tech);
}

async function finish(m, mixBlob, opts) {
  console.info(`processed with ${m.backend}${m.lite ? ' (light model)' : ''} in ${(m.ms / 1000).toFixed(1)} s`);
  const result = m.result;
  restoreTech(result);
  const id = `p${Date.now().toString(36)}`;
  const project = {
    id,
    name: result.name,
    created: Date.now(),
    duration: result.duration,
    offset: result.offset || 0,
    summary: summaryOf(result),
    result,
    stats: { ms: m.ms, backend: m.backend, lite: !!m.lite },
  };
  state.project = project;
  state.undo = [];
  // audio for playback
  const urls = {};
  urls.mix = URL.createObjectURL(mixBlob);
  const guitarBlob = new Blob([m.guitarWav], { type: 'audio/wav' });
  urls.guitar = URL.createObjectURL(guitarBlob);
  const partBlobs = {};
  for (const [pid, bytes] of Object.entries(m.partWavs || {})) {
    partBlobs[pid] = new Blob([bytes], { type: 'audio/wav' });
    urls['part:' + pid] = URL.createObjectURL(partBlobs[pid]);
  }
  m.guitarWav = null; m.partWavs = null;
  setUrls(urls);
  // save
  const stored = await store.saveProject(serializeProject(project));
  if (stored) {
    store.persist();
    await store.saveBlob(id, 'mix', mixBlob);
    await store.saveBlob(id, 'guitar', guitarBlob);
    for (const [pid, b] of Object.entries(partBlobs)) await store.saveBlob(id, 'part:' + pid, b);
  }
  openProject(project, true);
}

function setUrls(urls) {
  for (const u of Object.values(state.urls)) { try { URL.revokeObjectURL(u); } catch { /* ignore */ } }
  state.urls = urls;
}

function serializeProject(p) {
  const r = p.result;
  return {
    ...p,
    result: {
      ...r,
      alphaTex: null, ascii: null, texMaps: null,
      parts: r.parts.map((q) => ({ ...q, measures: null, audio: null })),
    },
  };
}

function summaryOf(result) {
  const p = result.parts[0];
  return `${tuningName(p.setup.tuning)}${p.setup.capo ? ', capo ' + p.setup.capo : ''} · ${Math.round(result.rhythm.bpm)} BPM` + (result.parts.length > 1 ? ` · ${result.parts.length} guitars` : '');
}

async function saveCurrent() {
  if (!state.project) return;
  state.project.summary = summaryOf(state.project.result);
  await store.saveProject(serializeProject(state.project));
}

/* ---------------- result screen ---------------- */
let tabviewPromise = null;
async function getTabView() {
  if (state.tabview) return state.tabview;
  if (!tabviewPromise) {
    tabviewPromise = import('./tabview.js').then(async ({ TabView }) => {
      const tv = new TabView($('tab'), {
        onNoteClick: (id) => openSheet(id),
        onPlayState: (playing) => setPlayIcon(playing),
        onTime: (t, d) => {
          $('timeLbl').textContent = `${fmtTime(t)} / ${fmtTime(d)}`;
          if (!seeking && d > 0) $('seek').value = String(Math.round((1000 * t) / d));
        },
      });
      await tv.init();
      state.tabview = tv;
      return tv;
    });
  }
  return tabviewPromise;
}

async function openProject(project, fresh = false) {
  state.project = project;
  state.part = 0;
  const r = project.result;
  restoreTech(r);
  if (!r.alphaTex) layout(r);
  show('screenResult');
  renderHead();
  renderSetup();
  renderParts();
  await renderTab(true);
  if (fresh && r.parts.length > 1) toast(`Found ${r.parts.length} guitars. Switch between them above the tab.`);
}

function part() { return state.project.result.parts[state.part]; }
// "4/4", or "4/4, changes" when the bars found in the song are not all the same length
function meterLabel(rh) {
  const lens = rh.bars ? new Set(rh.bars.slice(0, -1).map((b) => b.len)) : null;
  return `${rh.meter}/4${lens && lens.size > 1 ? ', changes' : ''}`;
}

function renderHead() {
  const r = state.project.result;
  $('songTitle').textContent = r.name.replace(/\.[a-z0-9]+$/i, '');
  const f = $('facts');
  f.innerHTML = '';
  const p = part();
  f.append(
    el('span', { class: 'fact' }, el('span', { class: 'k', text: 'Tuning' }), el('b', { text: tuningName(p.setup.tuning) })),
    el('span', { class: 'fact' }, el('span', { class: 'k', text: 'Capo' }), el('b', { text: p.setup.capo ? `Fret ${p.setup.capo}` : 'None' })),
    el('button', { class: 'fact', type: 'button', onclick: openRhythmDialog, title: 'Change tempo or time signature' },
      el('span', { class: 'k', text: 'Tempo' }), el('b', { class: 'mono', text: `${Math.round(r.rhythm.bpm)} BPM · ${meterLabel(r.rhythm)}` })),
  );
  if (r.tuningCents && Math.abs(r.tuningCents) >= 12) {
    f.append(el('span', { class: 'fact' }, el('span', { class: 'k', text: 'Pitch' }), el('b', { text: `${Math.abs(r.tuningCents)} cents ${r.tuningCents < 0 ? 'flat' : 'sharp'}` })));
  }
  if (r.offset || (r.duration && state.project.duration && Math.abs(r.duration - state.project.duration) > 1)) {
    f.append(el('span', { class: 'fact' }, el('span', { class: 'k', text: 'Section' }), el('b', { class: 'mono', text: `${fmtTime(r.offset || 0)}–${fmtTime((r.offset || 0) + r.duration)}` })));
  }
}

function neckSvg(capo) {
  // frets 0..12, six strings; capo drawn as a bar on its fret
  const W = 360, H = 70, x0 = 18, x1 = W - 6, n = 12;
  const fx = (k) => x0 + ((x1 - x0) * k) / n;
  const parts = [];
  parts.push(`<rect x="${x0 - 4}" y="8" width="4" height="${H - 22}" fill="var(--fg)"/>`);
  for (let k = 1; k <= n; k++) parts.push(`<line x1="${fx(k)}" y1="8" x2="${fx(k)}" y2="${H - 14}" stroke="var(--line)" stroke-width="1.5"/>`);
  for (let s = 0; s < 6; s++) { const y = 10 + (s * (H - 26)) / 5; parts.push(`<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="var(--muted)" stroke-width="${0.6 + (s * 0.25)}"/>`); }
  for (const k of [3, 5, 7, 9]) parts.push(`<circle cx="${(fx(k - 1) + fx(k)) / 2}" cy="${H / 2 - 4}" r="2.6" fill="var(--line)"/>`);
  parts.push(`<circle cx="${(fx(11) + fx(12)) / 2}" cy="${H / 2 - 12}" r="2.6" fill="var(--line)"/><circle cx="${(fx(11) + fx(12)) / 2}" cy="${H / 2 + 4}" r="2.6" fill="var(--line)"/>`);
  if (capo) {
    const cx = fx(capo) - 6;
    parts.push(`<rect x="${cx - 4}" y="4" width="8" height="${H - 14}" rx="3" fill="var(--accent)"/>`);
  }
  for (const k of [3, 5, 7, 9, 12]) parts.push(`<text x="${(fx(k - 1) + fx(k)) / 2}" y="${H - 2}" font-size="10" text-anchor="middle" fill="var(--muted)" font-family="system-ui, sans-serif">${k}</text>`);
  return `<svg class="neck" viewBox="0 0 ${W} ${H}" role="img" aria-label="${capo ? 'Capo on fret ' + capo : 'No capo'}">${parts.join('')}</svg>`;
}

function renderSetup() {
  const p = part();
  const box = $('setupBox');
  box.innerHTML = '';
  const strings = el('div', { class: 'strings', 'aria-label': 'Tuning, low string to high string' });
  p.setup.tuning.forEach((m, i) => {
    const d = el('div', {}, document.createTextNode(midiName(m, false)), el('small', { text: `${6 - i}` }));
    d.title = `${STRING_ORD[i]} string: ${midiName(m, true)}${p.setup.capo ? `, sounds ${midiName(m + p.setup.capo, true)} with the capo` : ''}`;
    strings.append(d);
  });
  const neckWrap = el('div', { style: 'min-width:0;flex:1 1 220px' });
  neckWrap.innerHTML = neckSvg(p.setup.capo);
  const lead = el('div', { style: 'display:grid;gap:6px;min-width:0' },
    el('div', { class: 'section-title', text: p.manual ? 'Your setup' : 'Best match' }),
    el('div', { style: 'font-size:18px;font-weight:700', text: `${tuningName(p.setup.tuning)}${p.setup.capo ? ` · capo on fret ${p.setup.capo}` : ' · no capo'}` }),
    strings,
  );
  box.append(el('div', { class: 'setup-main' }, lead, neckWrap));
  // alternatives
  if (p.ranking && p.ranking.length) {
    const alts = el('div', { class: 'alts' });
    p.ranking.slice(0, 5).forEach((r) => {
      const active = r.tuning.join() === p.setup.tuning.join() && r.capo === p.setup.capo;
      const b = el('button', { class: 'alt', type: 'button', 'aria-pressed': String(active), onclick: () => changeSetup({ tuning: r.tuning, capo: r.capo }) },
        document.createTextNode(`${r.name}${r.capo ? ', capo ' + r.capo : ''}`), el('span', { class: 'pct', text: `${Math.round(r.confidence * 100)}%` }));
      alts.append(b);
    });
    alts.append(el('button', { class: 'alt', type: 'button', onclick: openSetupDialog, text: 'Other…' }));
    box.append(el('div', { class: 'section-title', text: 'Other possibilities' }), alts);
    const eq = p.ranking[0].equivalents || [];
    const same = eq.length ? `Sounds the same: ${eq.map((e) => `${e.name}${e.capo ? ', capo ' + e.capo : ''}`).join('; ')}. The tab numbers are identical for those.` : '';
    box.append(el('p', { class: 'note-line', text: `${same} The tuning and capo are worked out from which fingering is most playable. Check against the recording, and pick another if it sounds off.`.trim() }));
  }
}

function renderParts() {
  const seg = $('partSeg');
  seg.innerHTML = '';
  const parts = state.project.result.parts;
  seg.hidden = parts.length < 2;
  parts.forEach((p, i) => {
    seg.append(el('button', { type: 'button', 'aria-pressed': String(i === state.part), onclick: () => { state.part = i; renderParts(); renderHead(); renderSetup(); renderTab(); }, text: p.name }));
  });
  const src = $('srcSel');
  const gOpt = src.querySelector('option[value="guitar"]');
  gOpt.textContent = parts.length > 1 ? 'This guitar only' : 'Guitar only';
}

async function renderTab(resetSource = false) {
  const r = state.project.result;
  $('ascii').textContent = asciiForPart(state.part);
  const tv = await getTabView();
  if (resetSource) {
    $('srcSel').value = 'mix';
    tv.setSource('mix', state.urls.mix, 0);
    $('speedSel').value = '1';
  }
  tv.show(r.alphaTex, r.texMaps, state.part, r.syncPoints);
  if ($('srcSel').value === 'guitar') applySource();
}

function asciiForPart(i) {
  // text tab for just this part
  const r = state.project.result;
  const blocks = r.ascii.split(/\n(?=Guitar)/);
  if (r.parts.length < 2) return r.ascii;
  const name = r.parts[i].name;
  const head = blocks[0].split('\n')[0];
  const b = blocks.find((x) => x.startsWith(name));
  return b ? `${head}\n\n${b}` : r.ascii;
}

$('viewTab').addEventListener('click', () => setView('tab'));
$('viewText').addEventListener('click', () => setView('text'));
function setView(v) {
  state.view = v;
  $('viewTab').setAttribute('aria-pressed', String(v === 'tab'));
  $('viewText').setAttribute('aria-pressed', String(v === 'text'));
  $('tab').hidden = v !== 'tab';
  $('ascii').hidden = v !== 'text';
  $('editHint').hidden = v !== 'tab';
}

function relayout() {
  layout(state.project.result);
  renderHead();
  renderSetup();
  renderTab();
  saveCurrent();
}

function pushUndo() {
  const p = part();
  state.undo.push({ part: state.part, notes: p.notes.map((n) => ({ ...n })), setup: { ...p.setup }, manual: p.manual, rhythm: { ...state.project.result.rhythm } });
  if (state.undo.length > 40) state.undo.shift();
  $('undoBtn').disabled = false;
}
$('undoBtn').addEventListener('click', () => {
  const u = state.undo.pop();
  if (!u) return;
  const p = state.project.result.parts[u.part];
  p.notes = u.notes;
  p.setup = u.setup;
  p.manual = u.manual;
  state.project.result.rhythm = u.rhythm;
  applyTech(p);
  $('undoBtn').disabled = state.undo.length === 0;
  relayout();
});

function applyTech(p) {
  // keep fingering as is, just refresh technique marks
  import('../engine/techniques.js').then(({ detectTechniques }) => {
    p.tech = detectTechniques(p.notes.filter((n) => !n.dropped && !n.deleted));
  });
}

function changeSetup(setup) {
  pushUndo();
  const p = part();
  for (const n of p.notes) delete n.pin; // a new tuning invalidates hand-picked strings
  applySetup(p, setup);
  p.manual = !(p.ranking && p.ranking[0] && p.ranking[0].tuning.join() === setup.tuning.join() && p.ranking[0].capo === setup.capo);
  rememberTuning(setup.tuning);
  relayout();
  toast(`Re-fingered for ${tuningName(setup.tuning)}${setup.capo ? ', capo ' + setup.capo : ''}`);
}

/* ---------------- dialogs ---------------- */
function dialog(title, body, actions) {
  const back = el('div', { class: 'dialog-backdrop', onclick: (e) => { if (e.target === back) back.remove(); } });
  const d = el('div', { class: 'dialog', role: 'dialog', 'aria-modal': 'true', 'aria-label': title }, el('h2', { text: title }), body, el('div', { class: 'row' }, ...actions(() => back.remove())));
  back.append(d);
  document.body.append(back);
  const f = d.querySelector('select, input, button');
  if (f) f.focus();
  return back;
}

function openSetupDialog() {
  const p = part();
  const tSel = el('select', { id: 'dlgTuning' });
  let found = false;
  for (const t of TUNINGS) {
    const o = el('option', { value: t.notes.join(','), text: `${t.name} (${t.notes.map((m) => midiName(m, false)).join(' ')})` });
    if (t.notes.join() === p.setup.tuning.join()) { o.selected = true; found = true; }
    tSel.append(o);
  }
  if (!found) tSel.prepend(el('option', { value: p.setup.tuning.join(','), text: `${tuningName(p.setup.tuning)} (${p.setup.tuning.map((m) => midiName(m, false)).join(' ')})`, selected: true }));
  const custom = el('div', { class: 'tuning-custom' });
  STANDARD.forEach((m, i) => {
    const s = el('select', { 'aria-label': `${STRING_ORD[i]} string`, 'data-i': String(i) });
    for (let q = m - 7; q <= m + 2; q++) s.append(el('option', { value: String(q), text: midiName(q, true) }));
    s.value = String(p.setup.tuning[i]);
    s.addEventListener('change', () => { tSel.value = ''; });
    custom.append(s);
  });
  tSel.addEventListener('change', () => {
    if (!tSel.value) return;
    tSel.value.split(',').map(Number).forEach((v, i) => { custom.children[i].value = String(v); });
  });
  const cSel = el('select', { id: 'dlgCapo' });
  for (let i = 0; i <= 12; i++) cSel.append(el('option', { value: String(i), text: i ? `Fret ${i}` : 'No capo', selected: i === p.setup.capo }));
  const body = el('div', { style: 'display:grid;gap:12px' },
    el('label', { class: 'small muted', text: 'Tuning' }), tSel,
    el('div', { class: 'small muted', text: 'Or set each string (low to high):' }), custom,
    el('label', { class: 'small muted', text: 'Capo' }), cSel,
  );
  dialog('Tuning and capo', body, (close) => [
    el('button', { class: 'btn primary', type: 'button', text: 'Re-finger the tab', onclick: () => {
      const tuning = Array.from(custom.children).map((s) => Number(s.value));
      close();
      changeSetup({ tuning, capo: Number(cSel.value) });
    } }),
    el('button', { class: 'btn', type: 'button', text: 'Cancel', onclick: close }),
  ]);
}

function openRhythmDialog() {
  const r = state.project.result;
  const bpm = el('input', { type: 'number', id: 'dlgBpm', min: '30', max: '300', step: '0.1', value: String(Math.round(r.rhythm.bpm * 10) / 10), class: 'mono' });
  const meter = el('select', { id: 'dlgMeter' });
  for (const m of [2, 3, 4, 5, 6, 7]) meter.append(el('option', { value: String(m), text: `${m}/4`, selected: m === r.rhythm.meter }));
  const body = el('div', { style: 'display:grid;gap:10px' },
    el('p', { class: 'note-line', text: 'If the bars look off, fix the beat here. Half-time and double-time mix-ups are common.' }),
    el('div', { class: 'row' }, el('label', { class: 'small muted', for: 'dlgBpm', text: 'Beats per minute' }), bpm,
      el('button', { class: 'btn', type: 'button', text: '÷2', onclick: () => { bpm.value = String(Math.round(Number(bpm.value) * 5) / 10); } }),
      el('button', { class: 'btn', type: 'button', text: '×2', onclick: () => { bpm.value = String(Math.round(Number(bpm.value) * 20) / 10); } })),
    el('div', { class: 'row' }, el('label', { class: 'small muted', for: 'dlgMeter', text: 'Beats per bar' }), meter),
    el('div', { class: 'row' }, el('span', { class: 'small muted', text: 'Bar lines start' }),
      el('button', { class: 'btn', type: 'button', text: '1 beat earlier', onclick: () => { shift -= 1; } }),
      el('button', { class: 'btn', type: 'button', text: '1 beat later', onclick: () => { shift += 1; } })),
  );
  let shift = 0;
  dialog('Tempo and bars', body, (close) => [
    el('button', { class: 'btn primary', type: 'button', text: 'Apply', onclick: () => {
      pushUndo();
      const nb = Number(bpm.value);
      const newTempo = Number.isFinite(nb) && nb > 20 && Math.abs(nb - r.rhythm.bpm) > 0.05;
      const changed = newTempo || Number(meter.value) !== r.rhythm.meter || shift !== 0;
      const firstBarTime = r.rhythm.bars && r.rhythm.bars.length ? r.rhythm.beats[r.rhythm.bars[0].beat] : null;
      if (newTempo) rescaleBeats(r, nb);
      // a hand-set tempo, meter or bar shift means one steady meter from here on, starting
      // from the first bar line that was found
      if (changed && r.rhythm.bars) {
        if (firstBarTime != null && r.rhythm.beats.length) {
          let bi = 0;
          r.rhythm.beats.forEach((t, i) => { if (Math.abs(t - firstBarTime) < Math.abs(r.rhythm.beats[bi] - firstBarTime)) bi = i; });
          r.rhythm.downbeatPhase = bi % Number(meter.value);
        }
        r.rhythm.bars = null;
      }
      r.rhythm.meter = Number(meter.value);
      r.rhythm.downbeatPhase = (((r.rhythm.downbeatPhase || 0) + shift) % r.rhythm.meter + r.rhythm.meter) % r.rhythm.meter;
      close();
      relayout();
    } }),
    el('button', { class: 'btn', type: 'button', text: 'Cancel', onclick: close }),
  ]);
}

function rescaleBeats(r, nb) {
  // halve / double the tracked beats when the tempo changes by ~2x, otherwise rebuild a steady grid
  const old = r.rhythm.bpm, beats = r.rhythm.beats;
  const ratio = nb / old;
  if (Math.abs(ratio - 2) < 0.08 && beats.length > 1) {
    const nbts = [];
    for (let i = 0; i < beats.length - 1; i++) nbts.push(beats[i], (beats[i] + beats[i + 1]) / 2);
    nbts.push(beats[beats.length - 1]);
    r.rhythm.beats = nbts;
  } else if (Math.abs(ratio - 0.5) < 0.04 && beats.length > 1) {
    r.rhythm.beats = beats.filter((_, i) => i % 2 === 0);
  } else {
    const start = beats.length ? beats[0] : 0;
    const out = [];
    for (let t = start; t < r.duration; t += 60 / nb) out.push(t);
    r.rhythm.beats = out;
  }
  r.rhythm.bpm = nb;
}

/* ---------------- note editing ---------------- */
let sheet = null;
function closeSheet() { if (sheet) { sheet.remove(); sheet = null; } }
function openSheet(id) {
  closeSheet();
  const p = part();
  const n = p.notes.find((q) => q.id === id);
  if (!n) return;
  const { tuning, capo } = p.setup;
  const head = el('div', { class: 'edit-head' },
    el('strong', { class: 'mono', text: `${midiName(n.pitch, true)}` }),
    el('span', { class: 'muted small', text: `${STRING_ORD[n.string]} string, fret ${n.fret}${capo ? ' (from the capo)' : ''} · at ${fmtTime(n.t0 + (state.project.result.offset || 0))}` }),
    el('span', { class: 'spacer' }),
    el('button', { class: 'btn quiet', type: 'button', text: 'Close', onclick: closeSheet }),
  );
  const chips = el('div', { class: 'chips' });
  for (let s = 0; s < 6; s++) {
    const f = n.pitch - tuning[s] - capo;
    if (f < 0 || f + capo > 24) continue;
    chips.append(el('button', { class: 'chip', type: 'button', 'aria-pressed': String(s === n.string), onclick: () => {
      pushUndo(); n.string = s; n.fret = f; n.pin = s; applyTech(p); closeSheet(); relayout();
    }, text: `${6 - s}: ${f}` }));
  }
  const actions = el('div', { class: 'chips' },
    el('button', { class: 'btn', type: 'button', text: '½ step down', onclick: () => editPitch(n, -1) }),
    el('button', { class: 'btn', type: 'button', text: '½ step up', onclick: () => editPitch(n, 1) }),
    el('button', { class: 'btn', type: 'button', text: 'Octave down', onclick: () => editPitch(n, -12) }),
    el('button', { class: 'btn', type: 'button', text: 'Delete note', onclick: () => { pushUndo(); n.deleted = true; closeSheet(); relayout(); } }),
  );
  sheet = el('div', { class: 'edit-sheet', role: 'dialog', 'aria-label': 'Edit note' },
    el('div', { class: 'edit-inner' }, head, el('div', { class: 'small muted', text: 'Play it on another string (string: fret):' }), chips, actions));
  document.body.append(sheet);
}

function editPitch(n, d) {
  const p = part();
  const { tuning, capo } = p.setup;
  const np = n.pitch + d;
  // keep the same string if possible, otherwise the nearest playable one
  let best = null;
  for (let s = 0; s < 6; s++) {
    const f = np - tuning[s] - capo;
    if (f < 0 || f + capo > 24) continue;
    const score = Math.abs(s - n.string) * 10 + Math.abs(f - n.fret);
    if (!best || score < best.score) best = { s, f, score };
  }
  if (!best) { toast('That note is out of range for this tuning.'); return; }
  pushUndo();
  n.pitch = np; n.string = best.s; n.fret = best.f; n.pin = best.s;
  applyTech(p);
  closeSheet();
  relayout();
}

/* ---------------- export ---------------- */
$('exportBtn').addEventListener('click', () => {
  const r = state.project.result;
  const base = r.name.replace(/\.[a-z0-9]+$/i, '').replace(/[\\/:*?"<>|]+/g, '_');
  const item = (label, sub, fn) => el('button', { class: 'btn', type: 'button', style: 'justify-content:flex-start;text-align:left;height:auto;padding:10px 14px;display:grid', onclick: async () => {
    try {
      const res = await fn();
      if (res === 'saved' || res === 'shared') toast('Saved');
      else if (res === 'copied') toast('Copied');
    } catch (e) { toast(e.message || String(e)); }
  } }, el('span', { text: label }), el('span', { class: 'small muted', style: 'font-weight:400', text: sub }));
  const list = el('div', { style: 'display:grid;gap:8px' },
    item('Copy the text tab', 'Paste it into notes, a message or a tab site', async () => (await copyText(asciiForPart(state.part)) ? 'copied' : (setView('text'), toast('Select the text tab and copy it'), 'none'))),
    item('Text tab (.txt)', 'Plain text, every guitar part', () => saveFile(`${base} - tab.txt`, new Blob([r.ascii], { type: 'text/plain' }))),
    item('Guitar Pro (.gp)', 'Opens in Guitar Pro, TuxGuitar, MuseScore, Songsterr', async () => {
      const tv = await getTabView();
      const bytes = tv.exportGp();
      return saveFile(`${base}.gp`, new Blob([bytes], { type: 'application/octet-stream' }));
    }),
    item('MIDI (.mid)', 'Notes with the real timing, for a DAW', () => saveFile(`${base}.mid`, new Blob([toMidi(docOf(r))], { type: 'audio/midi' }))),
    item('Isolated guitar (.wav)', 'The guitar pulled out of the mix', async () => {
      const b = await (await fetch(state.urls.guitar)).blob();
      return saveFile(`${base} - guitar only.wav`, b);
    }),
    item('Everything (.zip)', 'Text tab, Guitar Pro, MIDI and the guitar audio', async () => {
      const tv = await getTabView();
      const g = new Uint8Array(await (await (await fetch(state.urls.guitar)).blob()).arrayBuffer());
      const zip = makeZip([
        { name: `${base} - tab.txt`, data: r.ascii },
        { name: `${base}.gp`, data: tv.exportGp() },
        { name: `${base}.mid`, data: toMidi(docOf(r)) },
        { name: `${base} - guitar only.wav`, data: g },
      ]);
      return saveFile(`${base} - tab.zip`, zip);
    }),
  );
  dialog('Save or export', list, (close) => [el('button', { class: 'btn', type: 'button', text: 'Done', onclick: close })]);
});

/* ---------------- player ---------------- */
let seeking = false;
function setPlayIcon(playing) {
  $('playIcon').innerHTML = playing ? '<rect x="4" y="3" width="3.5" height="12" fill="currentColor"/><rect x="10.5" y="3" width="3.5" height="12" fill="currentColor"/>' : '<path d="M4 2.5v13l11-6.5z" fill="currentColor"/>';
  $('playBtn').setAttribute('aria-label', playing ? 'Pause' : 'Play');
}
$('playBtn').addEventListener('click', async () => { const tv = await getTabView(); tv.toggle(); });
$('speedSel').addEventListener('change', async () => { const tv = await getTabView(); tv.setSpeed(Number($('speedSel').value)); });
$('loopBtn').addEventListener('click', async () => {
  const tv = await getTabView();
  const on = $('loopBtn').getAttribute('aria-pressed') !== 'true';
  $('loopBtn').setAttribute('aria-pressed', String(on));
  tv.setLoop(on);
  if (on) toast('Looping. Drag across the tab to loop just that part.');
});
$('seek').addEventListener('input', () => { seeking = true; });
$('seek').addEventListener('change', async () => { seeking = false; const tv = await getTabView(); tv.seekFraction(Number($('seek').value) / 1000); });
$('srcSel').addEventListener('change', applySource);
async function applySource() {
  const tv = await getTabView();
  const v = $('srcSel').value;
  const r = state.project.result;
  const off = r.offset || 0;
  if (v === 'mix') tv.setSource('mix', state.urls.mix, 0);
  else if (v === 'guitar') {
    const pid = r.parts[state.part].id;
    const url = state.urls['part:' + pid] || state.urls.guitar;
    tv.setSource('guitar', url, off);
  } else tv.setSource('synth');
  tv.setSpeed(Number($('speedSel').value));
}
function stopPlayback() { if (state.tabview) state.tabview.pause(); }

/* ---------------- library ---------------- */
async function refreshLibrary() {
  const list = await store.listProjects();
  const box = $('library');
  box.innerHTML = '';
  if (!list.length) { box.append(el('div', { class: 'empty', text: 'Tabs you make are saved here on this device.' })); return; }
  for (const p of list) {
    const confirmDel = el('button', { class: 'btn', type: 'button', text: 'Delete', hidden: true, onclick: async () => { await store.deleteProject(p.id); refreshLibrary(); } });
    const del = el('button', { class: 'btn quiet', type: 'button', text: '✕', 'aria-label': `Remove ${p.name}`, onclick: () => { del.hidden = true; confirmDel.hidden = false; } });
    box.append(el('div', { class: 'lib-item' },
      el('div', { class: 'name', text: p.name.replace(/\.[a-z0-9]+$/i, '') }),
      el('div', { class: 'meta', text: `${p.summary || ''} · ${new Date(p.created).toLocaleDateString()}` }),
      el('div', { class: 'actions' }, el('button', { class: 'btn', type: 'button', text: 'Open', onclick: () => loadSaved(p.id) }), del, confirmDel)));
  }
}

async function loadSaved(id) {
  const p = await store.getProject(id);
  if (!p) { toast('Could not open that tab.'); return; }
  const urls = {};
  const mix = await store.getBlob(id, 'mix');
  const guitar = await store.getBlob(id, 'guitar');
  if (mix) urls.mix = URL.createObjectURL(mix);
  if (guitar) urls.guitar = URL.createObjectURL(guitar);
  for (const q of p.result.parts) {
    const b = await store.getBlob(id, 'part:' + q.id);
    if (b) urls['part:' + q.id] = URL.createObjectURL(b);
  }
  setUrls(urls);
  state.undo = [];
  $('undoBtn').disabled = true;
  openProject(p);
}

/* ---------------- misc ---------------- */
let toastTimer = 0;
function toast(msg) {
  let t = document.querySelector('.toast');
  if (!t) { t = el('div', { class: 'toast', role: 'status' }); document.body.append(t); }
  t.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 3200);
}

fillSelects();
setupGpuRow();
$('optModel').value = modelPref();
$('optModel').addEventListener('change', () => setModelPref($('optModel').value));
$('optNotes').value = notesPref();
$('optNotes').addEventListener('change', () => setNotesPref($('optNotes').value));
checkLastRun();
fillFavAdd();
renderFavorites();
engineInfo();
refreshLibrary();
