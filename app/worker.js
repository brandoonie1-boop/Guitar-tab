// Processing worker: loads the models and runs the transcription pipeline off the main thread.
import { transcribe } from '../engine/pipeline.js';

import { fetchAsset, getManifest, baseUrl } from './assets.js';

const BASE = baseUrl();
let ort = null;
let ortKind = null;
let demucs = null, basicPitch = null;
let cancelled = false;

function post(type, data, transfer) { self.postMessage({ type, ...data }, transfer || []); }

// A real graphics chip, not a software emulation (which would be far slower than the processor).
let gpuCheck = null;
function hardwareGpu() {
  if (gpuCheck) return gpuCheck;
  gpuCheck = (async () => {
    try {
      if (!self.navigator.gpu) return false;
      const adapter = await self.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return false;
      const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {}) || {};
      if (adapter.isFallbackAdapter || info.isFallbackAdapter) return false;
      const desc = `${info.vendor || ''} ${info.architecture || ''} ${info.device || ''} ${info.description || ''}`.toLowerCase();
      if (/swiftshader|llvmpipe|lavapipe|software|microsoft basic render|warp/.test(desc)) return false;
      return true;
    } catch { return false; }
  })();
  return gpuCheck;
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
  if (useGpu) opts.executionProviders = ['webgpu'];
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
    const mf = await getManifest();
    const wantGpu = !!msg.opts.gpu && !!mf.ort.webgpu && !!mf.assets.ortWasmGpu && (await hardwareGpu());
    let useGpu = wantGpu;
    try {
      await loadOrt(useGpu);
    } catch (err) {
      if (!useGpu) throw err;
      useGpu = false;
      await loadOrt(false);
    }
    let demucsBytes = null;
    const env = {
      ort,
      cancelled: () => cancelled,
      progress: (p) => post('progress', p),
      checkSeparation: useGpu,
      getDemucs: async () => {
        if (demucs) return demucs;
        demucsBytes = await fetchAsset('demucs', (f) => post('progress', { stage: 'load', frac: 0.3 + 0.7 * f, label: 'Downloading the guitar isolation model (once)' }));
        try {
          demucs = await createSession(demucsBytes, useGpu);
        } catch (err) {
          if (!useGpu) throw err;
          console.warn('GPU session failed, using the processor', err);
          post('gpu', { ok: false, reason: String(err && err.message || err) });
          useGpu = false;
          env.checkSeparation = false;
          await loadOrt(false);
          env.ort = ort;
          demucs = await createSession(demucsBytes, false);
        }
        return demucs;
      },
      fallbackToCpu: async (err) => {
        if (!useGpu) throw err;
        console.warn('GPU separation failed, using the processor', err);
        post('gpu', { ok: false, reason: String(err && err.message || err) });
        useGpu = false;
        env.checkSeparation = false;
        try { if (demucs && demucs.release) await demucs.release(); } catch { /* ignore */ }
        demucs = null; basicPitch = null;
        await loadOrt(false);
        env.ort = ort;
        if (!demucsBytes) demucsBytes = await fetchAsset('demucs');
        demucs = await createSession(demucsBytes, false);
        return { ort, session: demucs };
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
