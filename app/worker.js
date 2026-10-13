// Processing worker: loads the models and runs the transcription pipeline off the main thread.
import { transcribe } from '../engine/pipeline.js';

const BASE = new URL('../', import.meta.url).href;
let ort = null;
let ortKind = null;
let demucs = null, basicPitch = null;
let cancelled = false;
let manifest = null;

function post(type, data, transfer) { self.postMessage({ type, ...data }, transfer || []); }

async function getManifest() {
  if (manifest) return manifest;
  const r = await fetch(BASE + 'assets.json');
  manifest = await r.json();
  return manifest;
}

// fetch one asset (possibly split into chunks), with optional Cache Storage caching
async function fetchAsset(key, onProgress) {
  const m = await getManifest();
  const a = m.assets[key];
  if (!a) throw new Error('missing asset ' + key);
  const urls = a.parts.map((p) => BASE + p);
  let cache = null;
  try { cache = self.caches ? await caches.open('guitar-tab-assets-' + m.version) : null; } catch { cache = null; }
  const total = a.size;
  let loaded = 0;
  const chunks = [];
  for (const u of urls) {
    let resp = null;
    if (cache) { try { resp = await cache.match(u); } catch { resp = null; } }
    let buf;
    if (resp) {
      buf = new Uint8Array(await resp.arrayBuffer());
      loaded += buf.length;
      if (onProgress) onProgress(loaded / total);
    } else {
      const r = await fetch(u);
      if (!r.ok) throw new Error(`Could not download ${key} (${r.status})`);
      const reader = r.body ? r.body.getReader() : null;
      if (reader) {
        const parts = [];
        let n = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          parts.push(value); n += value.length; loaded += value.length;
          if (onProgress) onProgress(Math.min(1, loaded / total));
        }
        buf = new Uint8Array(n);
        let o = 0;
        for (const p of parts) { buf.set(p, o); o += p.length; }
      } else {
        buf = new Uint8Array(await r.arrayBuffer());
        loaded += buf.length;
      }
      if (cache) { try { await cache.put(u, new Response(buf.slice(0))); } catch { /* storage full or blocked */ } }
    }
    chunks.push(buf);
  }
  if (chunks.length === 1) return chunks[0];
  const out = new Uint8Array(chunks.reduce((s, c) => s + c.length, 0));
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

let threadLimit = 8;
async function loadOrt(useGpu) {
  const want = useGpu ? 'webgpu' : 'wasm';
  if (ort && ortKind === want) return ort;
  const m = await getManifest();
  const mod = await import(BASE + (useGpu ? m.ort.webgpu : m.ort.wasm));
  const o = mod.default && mod.default.InferenceSession ? mod.default : mod;
  const wasmKey = useGpu ? 'ortWasmGpu' : 'ortWasm';
  o.env.wasm.wasmBinary = await fetchAsset(wasmKey, (f) => post('progress', { stage: 'load', frac: f * 0.3, label: 'Loading the audio engine' }));
  const threads = self.crossOriginIsolated ? Math.max(1, Math.min(threadLimit, navigator.hardwareConcurrency || 4)) : 1;
  o.env.wasm.numThreads = threads;
  o.env.wasm.proxy = false;
  ort = o; ortKind = want;
  demucs = null; basicPitch = null;
  return o;
}

async function createSession(bytes, useGpu) {
  const opts = { graphOptimizationLevel: 'all' };
  if (useGpu) opts.executionProviders = ['webgpu', 'wasm'];
  else opts.executionProviders = ['wasm'];
  return ort.InferenceSession.create(bytes, opts);
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'cancel') { cancelled = true; return; }
  if (msg.type !== 'run') return;
  cancelled = false;
  if (msg.opts.threads) threadLimit = msg.opts.threads;
  const t0 = performance.now();
  try {
    const wantGpu = !!msg.opts.gpu && !!self.navigator.gpu;
    let useGpu = wantGpu;
    try {
      await loadOrt(useGpu);
    } catch (err) {
      if (!useGpu) throw err;
      useGpu = false;
      await loadOrt(false);
    }
    const env = {
      ort,
      cancelled: () => cancelled,
      progress: (p) => post('progress', p),
      getDemucs: async (onProg) => {
        if (demucs) return demucs;
        const bytes = await fetchAsset('demucs', (f) => post('progress', { stage: 'load', frac: 0.3 + 0.7 * f, label: 'Downloading the guitar isolation model (once)' }));
        try {
          demucs = await createSession(bytes, useGpu);
        } catch (err) {
          if (!useGpu) throw err;
          // GPU path failed: fall back to CPU
          useGpu = false;
          await loadOrt(false);
          env.ort = ort;
          demucs = await createSession(bytes, false);
        }
        return demucs;
      },
      getBasicPitch: async () => {
        if (basicPitch) return basicPitch;
        const bytes = await fetchAsset('basicpitch');
        basicPitch = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
        return basicPitch;
      },
    };
    const audio = { left: msg.audio.left, right: msg.audio.right, sr: msg.audio.sr, name: msg.audio.name };
    const result = await transcribe(audio, msg.opts, env);
    // send arrays separately (transferable); keep the rest as plain data
    const transfer = [];
    const stems = {};
    const add = (k, arr) => { if (arr) { stems[k] = arr; transfer.push(arr.buffer); } };
    add('guitarL', result.stems.guitar[0]);
    add('guitarR', result.stems.guitar[1] !== result.stems.guitar[0] ? result.stems.guitar[1] : null);
    if (result.stems.drums) add('drums', result.stems.drums);
    result.stems = null;
    const partAudio = {};
    for (const p of result.parts) { partAudio[p.id] = p.audio; transfer.push(p.audio.buffer); p.audio = null; }
    // Maps don't survive structured clone well in all browsers: convert tech to arrays
    for (const p of result.parts) p.tech = Array.from(p.tech.entries());
    post('result', { result, stems, partAudio, ms: performance.now() - t0, backend: useGpu ? 'webgpu' : `wasm x${ort.env.wasm.numThreads}` }, transfer);
  } catch (err) {
    post('error', { message: err && err.message === 'cancelled' ? 'cancelled' : String(err && err.message || err) });
  }
};
