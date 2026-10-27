// The best guitar isolation model, BS-RoFormer SW (6 stems), is not hosted by this app: it is
// downloaded from Hugging Face, where its ONNX export is published, and cut down on the device to
// the stems the app uses, as a chain of parts that needs under 0.9 GB of engine memory instead of
// 3 GB (engine/onnxprune.js). Only the cut is kept, in Cache Storage, and it is checked against
// the known SHA-256 of that cut.
import { pruneStream, bsRoformerPlan } from '../engine/onnxprune.js';

const SRC = {
  url: 'https://huggingface.co/elicwhite/bs-roformer-sw-6stem-onnx/resolve/a744f80957374e1735ad70fa122670b7961da8cc/bs_roformer_sw_6stem_fp16.onnx',
  size: 352778874,
};

export const BSR_CUTS = {
  guitar: { keep: [4], stems: ['guitar'], size: 103556907, sha256: '8b47c5ebae36e2f0de876fe6624f463ac1a30690c5c57fc8596e00038e4f8e71' },
  guitarOther: { keep: [4, 2], stems: ['guitar', 'other'], size: 153292675, sha256: '3e68796162314039b73b1d0dd5b1325b3049835b8684a54bdd4e7ec67c00f5cf' },
};
// 6 attention slices: a WebGPU shader may use at most 8 buffers (a split into 6 uses 8)
export const CUT_OPTS = { attentionGroups: 6, chain: true };
export const BSR_DOWNLOAD_MB = Math.round(SRC.size / 1e6);

const CACHE = 'guitar-tab-models'; // not tied to the app version: the cut never changes
const keyOf = (cut) => `https://models.invalid/bsr-${cut.sha256}`;

async function sha256Hex(bytes) {
  const d = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, '0')).join('');
}

async function openCache() {
  try { return self.caches ? await caches.open(CACHE) : null; } catch { return null; }
}

export async function hasBsrModel(name = 'guitar') {
  const c = await openCache();
  if (!c) return false;
  try { return !!(await c.match(keyOf(BSR_CUTS[name]))); } catch { return false; }
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * The file in 8 MB ranges, each fetched whole before it is handed on, so a dropped connection
 * repeats only that range. Falls back to one plain download if ranges are refused.
 */
async function* download(onBytes, shouldStop) {
  const RANGE = 8 << 20;
  let streamed = false;
  for (let at = 0; at < SRC.size; at += RANGE) {
    const end = Math.min(SRC.size, at + RANGE) - 1;
    for (let tries = 0; ; tries++) {
      if (shouldStop && shouldStop()) throw new Error('cancelled');
      try {
        const r = await fetch(SRC.url, { headers: { Range: `bytes=${at}-${end}` }, cache: 'no-store' });
        if (r.status === 200 && at === 0) {
          // no ranges here: stream the whole file once (a break can't be resumed)
          streamed = true;
          const reader = r.body.getReader();
          for (;;) {
            const { done, value } = await reader.read();
            if (done) return;
            onBytes(value.length);
            yield value;
          }
        }
        if (r.status !== 206) throw new Error(`HTTP ${r.status}`);
        const buf = new Uint8Array(await r.arrayBuffer());
        if (buf.length !== end - at + 1) throw new Error('download cut short');
        onBytes(buf.length);
        yield buf;
        break;
      } catch (err) {
        if (streamed || tries >= 6 || (err && err.message === 'cancelled')) {
          if (err && err.message === 'cancelled') throw err;
          throw new Error(`Could not download the isolation model from Hugging Face (${(err && err.message) || err}). Check the connection and try again, or pick the standard isolation model.`);
        }
        await wait(1000 * 2 ** tries);
      }
    }
  }
}

/**
 * The BS-RoFormer cut for `name` ('guitar' | 'guitarOther') as model bytes: from the cache,
 * or downloaded (353 MB, once) and cut as it arrives.
 * onProgress(fraction, label)
 */
export async function getBsrModel(name, onProgress, shouldStop) {
  const cut = BSR_CUTS[name];
  const cache = await openCache();
  if (cache) {
    try {
      const hit = await cache.match(keyOf(cut));
      if (hit) {
        const b = new Uint8Array(await hit.arrayBuffer());
        if (b.length === cut.size) return b;
      }
    } catch { /* fall through to a download */ }
  }
  let got = 0;
  const label = `Downloading the best isolation model from Hugging Face (once, ${BSR_DOWNLOAD_MB} MB)`;
  const bytes = await pruneStream(download((n) => {
    got += n;
    if (onProgress) onProgress(Math.min(1, got / SRC.size), `${label} · ${Math.round((100 * got) / SRC.size)}%`);
  }, shouldStop), bsRoformerPlan(cut.keep, CUT_OPTS));
  if (onProgress) onProgress(1, 'Checking the isolation model');
  const sha = await sha256Hex(bytes);
  if (sha !== cut.sha256) throw new Error('The downloaded isolation model is not the expected file. Try again, or pick the standard isolation model.');
  if (cache) {
    try { await cache.put(keyOf(cut), new Response(bytes)); } catch { /* storage full or blocked: works without */ }
  }
  return bytes;
}

/** Remove the cached cuts (settings: free up space). */
export async function deleteBsrModels() {
  const c = await openCache();
  if (!c) return;
  for (const cut of Object.values(BSR_CUTS)) { try { await c.delete(keyOf(cut)); } catch { /* ignore */ } }
}
