// Processing worker: loads the models and runs the transcription pipeline off the main thread.
// A run uses two workers one after the other: one isolates the guitar ('separate'), the next
// does everything else ('analyze'). The page ends each worker when its part is done, so all
// the memory it used (the engine's WebAssembly memory can only grow, never shrink) is given
// back before the next part starts.
import { separateStage, analyzeStage } from '../engine/pipeline.js';
import { parseChain, loadChain } from '../engine/demucs.js';
import { encodeWav } from '../engine/wav.js';
import { fetchAsset, getManifest, baseUrl } from './assets.js';

const BASE = baseUrl();
let cancelled = false;

// diagnostics (?debug in the page URL): report the engine's WebAssembly memory with progress
const wasmMemories = [];
let debugMem = false;
function watchWasmMemory() {
  const Orig = WebAssembly.Memory;
  WebAssembly.Memory = function Memory(desc) { const m = new Orig(desc); if (wasmMemories.length < 8) wasmMemories.push(m); return m; };
  WebAssembly.Memory.prototype = Orig.prototype;
}
const wasmMB = () => Math.round(wasmMemories.reduce((s, m) => s + m.buffer.byteLength, 0) / 1e6);

function post(type, data, transfer) {
  if (debugMem && type === 'progress') data = { ...data, wasmMB: wasmMB() };
  self.postMessage({ type, ...data }, transfer || []);
}

// A real graphics chip, not a software emulation (which would be far slower than the processor).
async function hardwareGpu() {
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
}

async function loadOrt(useGpu, threads) {
  const m = await getManifest();
  const mod = await import(BASE + (useGpu ? m.ort.webgpu : m.ort.wasm));
  const o = mod.default && mod.default.InferenceSession ? mod.default : mod;
  o.env.wasm.wasmBinary = await fetchAsset(useGpu ? 'ortWasmGpu' : 'ortWasm', (f) => post('progress', { stage: 'load', frac: f * 0.3, label: 'Loading the audio engine' }));
  o.env.wasm.numThreads = threads;
  o.env.wasm.proxy = false;
  return o;
}

// The page restarts the run on the processor when the graphics chip fails.
function gpuFailed(err) {
  const e = new Error('gpu-failed');
  e.gpuFailed = true;
  e.reason = String((err && err.message) || err);
  return e;
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'cancel') { cancelled = true; return; }
  if (msg.type !== 'separate' && msg.type !== 'analyze') return;
  cancelled = false;
  const opts = msg.opts;
  debugMem = !!opts.debugMem;
  if (debugMem) watchWasmMemory();
  const t0 = performance.now();
  let useGpu = false;
  try {
    const mf = await getManifest();
    useGpu = msg.type === 'separate' && !!opts.gpu && !!mf.ort.webgpu && !!mf.assets.ortWasmGpu && (await hardwareGpu());
    const threads = self.crossOriginIsolated ? Math.max(1, Math.min(opts.threads || 8, navigator.hardwareConcurrency || 4)) : 1;
    let ort;
    try {
      ort = await loadOrt(useGpu, threads);
    } catch (err) {
      if (useGpu) throw gpuFailed(err);
      throw err;
    }
    const backend = useGpu ? 'webgpu' : `wasm x${threads}`;
    if (msg.type === 'separate') {
      const lite = !!opts.lite && !!mf.assets.demucsLite;
      post('engine', { backend, lite });
      let demucs = null;
      const env = {
        ort,
        cancelled: () => cancelled,
        progress: (p) => post('progress', p),
        checkSeparation: useGpu,
        getDemucs: async () => {
          let weights = await fetchAsset('demucsWeights', (f) => post('progress', { stage: 'load', frac: 0.3 + 0.65 * f, label: 'Downloading the guitar isolation model (once)' }));
          const chain = parseChain(await fetchAsset(lite ? 'demucsLite' : 'demucsFull'));
          post('progress', { stage: 'load', frac: 0.97, label: 'Starting the guitar isolation model' });
          try {
            demucs = await loadChain(ort, chain, weights, { graphOptimizationLevel: 'all', executionProviders: [useGpu ? 'webgpu' : 'wasm'] });
          } catch (err) {
            if (useGpu) throw gpuFailed(err);
            throw err;
          }
          weights = null;
          return demucs;
        },
        releaseDemucs: async () => { if (demucs) { await demucs.release(); demucs = null; } },
      };
      if (useGpu) env.fallbackToCpu = async (err) => { throw gpuFailed(err); };
      const a = msg.audio;
      msg.audio = null;
      const audio = { left: a.left, right: a.right, sr: a.sr, name: a.name, offset: a.offset || 0 };
      a.left = null; a.right = null;
      const st = await separateStage(audio, opts, env);
      const transfer = [st.gL.buffer];
      if (st.gR && st.gR !== st.gL) transfer.push(st.gR.buffer);
      if (st.rhythm) transfer.push(st.rhythm.buffer);
      post('stems', { stems: st, ms: performance.now() - t0, backend, lite }, transfer);
      return;
    }
    // analyze
    let basicPitch = null;
    const env = {
      ort,
      cancelled: () => cancelled,
      progress: (p) => post('progress', p),
      getBasicPitch: async () => {
        if (basicPitch) return basicPitch;
        const bytes = await fetchAsset('basicpitch');
        basicPitch = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
        return basicPitch;
      },
    };
    const stems = msg.stems;
    msg.stems = null;
    const result = await analyzeStage(stems, opts, env);
    // audio for playback goes back as 16-bit WAV files (half the size of the float samples)
    const [gL, gR] = result.stems.guitar;
    result.stems = null;
    const transfer = [];
    const guitarWav = encodeWav(gR && gR !== gL ? [gL, gR] : [gL], 44100);
    transfer.push(guitarWav.buffer);
    const partWavs = {};
    for (const p of result.parts) {
      if (p.audio && result.parts.length > 1) { partWavs[p.id] = encodeWav([p.audio], 44100); transfer.push(partWavs[p.id].buffer); }
      p.audio = null;
    }
    // Maps don't survive structured clone well in all browsers: convert tech to arrays
    for (const p of result.parts) p.tech = Array.from(p.tech.entries());
    post('result', { result, guitarWav, partWavs, ms: performance.now() - t0 }, transfer);
  } catch (err) {
    const message = err && err.message === 'cancelled' ? 'cancelled' : String((err && err.message) || err);
    post('error', { message, gpuFailed: !!(err && err.gpuFailed), reason: err && err.reason, usedGpu: useGpu });
  }
};
