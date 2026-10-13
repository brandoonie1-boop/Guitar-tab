// Guitar Tab - app controller.
import { TUNINGS, tuningName, STANDARD } from '../engine/tunings.js';
import { midiName } from '../engine/dsp.js';
import { applySetup, layout, docOf } from '../engine/pipeline.js';
import { toMidi } from '../engine/export.js';
import { decodeFile, wavBlob, makeZip, saveFile, copyText, inArtifact } from './files.js';
import * as store from './store.js';

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
  audio: null, // decoded {left,right,sr,duration}
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

function engineInfo() {
  const t = caps.threads > 1 ? `${caps.threads} processor threads` : 'one processor thread';
  const msg = `Runs on this device using ${t}.` + (caps.threads === 1 && !ARTIFACT ? ' Reload once after the first visit to unlock more threads.' : '')
    + (ARTIFACT ? ' The installed web app version is faster.' : '');
  $('engineInfo').textContent = msg;
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
  const a = parseTime($('rangeFrom').value), b = parseTime($('rangeTo').value);
  o.range = null;
  if (a != null || b != null) {
    const dur = state.audio ? state.audio.duration : 1e9;
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
  $('startError').hidden = true;
  $('startBtn').disabled = true;
  $('picked').hidden = false;
  $('pickedName').textContent = file.name;
  $('pickedInfo').textContent = 'Reading…';
  try {
    const a = await decodeFile(file);
    state.file = file;
    state.audio = a;
    $('pickedInfo').textContent = `${fmtTime(a.duration)} · ${a.channels > 1 ? 'stereo' : 'mono'}`;
    $('startBtn').disabled = false;
    if (a.duration > 600) showError('This file is over 10 minutes. Pick a section below so it finishes in a reasonable time.');
  } catch (e) {
    state.file = null; state.audio = null;
    $('pickedInfo').textContent = '';
    showError("This file couldn't be read. Try an MP3, M4A, WAV or a phone video. (" + (e.message || e) + ')');
  }
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
let workStart = 0, sepStart = 0, lastFrac = 0;

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
  if (p.stage === 'separate') {
    if (!sepStart) sepStart = performance.now();
    const el2 = (performance.now() - sepStart) / 1000;
    const sepFrac = (lastFrac - 0.02) / 0.62;
    if (sepFrac > 0.08) {
      const remain = (el2 / sepFrac) * (1 - sepFrac) * 1.25;
      txt += ` · about ${remain > 90 ? Math.round(remain / 60) + ' min' : Math.max(5, Math.round(remain / 5) * 5) + ' s'} left`;
    }
  }
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

$('startBtn').addEventListener('click', () => start(false));
$('cancelBtn').addEventListener('click', () => {
  if (state.worker) { state.worker.postMessage({ type: 'cancel' }); }
});

async function start(retry = false) {
  if (!state.audio) return;
  const opts = readOpts();
  const a = state.audio;
  show('screenWork');
  $('workTitle').textContent = `Working on ${state.file.name}`;
  $('workBar').style.width = '1%';
  $('workStatus').textContent = 'Starting';
  lastFrac = 0; sepStart = 0; workStart = performance.now();
  setStep('load');
  keepAwake(true);
  if (!state.worker) {
    state.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    state.worker.addEventListener('error', (ev) => {
      // a crash inside the worker (e.g. out of memory with many threads): retry once on one thread
      ev.preventDefault();
      workerFailed(ev.message || 'the processing worker stopped');
    });
  }
  const w = state.worker;
  state.retrying = retry;
  // keep the recording for playback: the original file if this browser can play it, otherwise a WAV
  const canPlay = state.file.type && new Audio().canPlayType(state.file.type);
  const mixBlob = canPlay ? state.file : wavBlob(a.right ? [a.left, a.right] : [a.left]);
  const left = a.left.slice(), right = a.right ? a.right.slice() : null;
  const runOpts = {
    threads: retry ? 1 : 8,
    isolate: opts.isolate, split: opts.split, tuning: opts.tuning, capo: opts.capo, range: opts.range,
    overlap: opts.quality === 'fast' ? 0.1 : 0.25,
    gpu: false,
    preferTunings: favoriteTunings(),
  };
  w.onmessage = async (e) => {
    const m = e.data;
    if (m.type === 'progress') onProgress(m);
    else if (m.type === 'error') {
      if (m.message !== 'cancelled' && !state.retrying && caps.threads > 1) { workerFailed(m.message); return; }
      keepAwake(false);
      show('screenStart');
      showError(m.message === 'cancelled' ? 'Stopped.' : `Something went wrong: ${m.message}`);
    } else if (m.type === 'result') {
      keepAwake(false);
      try {
        await finish(m, mixBlob, opts);
      } catch (err) {
        console.error(err);
        show('screenStart');
        showError('Something went wrong while showing the tab: ' + (err.message || err));
      }
    }
  };
  w.postMessage({ type: 'run', audio: { left, right, sr: a.sr, name: state.file.name }, opts: runOpts }, [left.buffer].concat(right ? [right.buffer] : []));
}

const DEFAULT_FAVS = ['38,45,52,57,59,64']; // DAEABE
function workerFailed(msg) {
  console.warn('worker failed:', msg);
  try { state.worker.terminate(); } catch { /* ignore */ }
  state.worker = null;
  if (!state.retrying && state.audio) {
    toast('Retrying with a single thread');
    start(true);
  } else {
    keepAwake(false);
    show('screenStart');
    showError(`Something went wrong: ${msg}. Try a shorter section of the song.`);
  }
}

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
    stats: { ms: m.ms, backend: m.backend },
  };
  state.project = project;
  state.undo = [];
  // audio for playback
  const urls = {};
  urls.mix = URL.createObjectURL(mixBlob);
  const g = m.stems.guitarR ? [m.stems.guitarL, m.stems.guitarR] : [m.stems.guitarL];
  const guitarBlob = wavBlob(g);
  urls.guitar = URL.createObjectURL(guitarBlob);
  const partBlobs = {};
  for (const [pid, arr] of Object.entries(m.partAudio)) {
    if (result.parts.length > 1) { partBlobs[pid] = wavBlob([arr]); urls['part:' + pid] = URL.createObjectURL(partBlobs[pid]); }
  }
  setUrls(urls);
  state.stems = m.stems;
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
    tabviewPromise = import('./tabview.js').then(({ TabView }) => {
      state.tabview = new TabView($('tab'), {
        onNoteClick: (id) => openSheet(id),
        onPlayState: (playing) => setPlayIcon(playing),
        onTime: (t, d) => {
          $('timeLbl').textContent = `${fmtTime(t)} / ${fmtTime(d)}`;
          if (!seeking && d > 0) $('seek').value = String(Math.round((1000 * t) / d));
        },
      });
      return state.tabview;
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
      el('span', { class: 'k', text: 'Tempo' }), el('b', { class: 'mono', text: `${Math.round(r.rhythm.bpm)} BPM · ${r.rhythm.meter}/4` })),
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
      if (Number.isFinite(nb) && nb > 20 && Math.abs(nb - r.rhythm.bpm) > 0.05) rescaleBeats(r, nb);
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
fillFavAdd();
renderFavorites();
engineInfo();
refreshLibrary();
