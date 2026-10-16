// Loads the big binary assets (models, engine) listed in assets.json.
// Parts can be raw binary or base64 text (used where a host only serves text types).
// Downloads are cached in Cache Storage when the browser allows it.

const BASE = new URL('../', import.meta.url).href;
let manifestP = null;

export function baseUrl() { return BASE; }

export function getManifest() {
  if (!manifestP) {
    manifestP = fetch(BASE + 'assets.json').then((r) => {
      if (!r.ok) throw new Error('Could not load the app files list (' + r.status + ')');
      return r.json();
    });
    manifestP.catch(() => { manifestP = null; });
  }
  return manifestP;
}

function decodeBase64(text) {
  const clean = text.replace(/[^A-Za-z0-9+/=]/g, '');
  if (typeof Uint8Array.fromBase64 === 'function') return Uint8Array.fromBase64(clean);
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4) - (clean.endsWith('==') ? 2 : clean.endsWith('=') ? 1 : 0));
  // decode in slices to keep the intermediate strings small
  const SLICE = 1 << 20; // multiple of 4
  let o = 0;
  for (let i = 0; i < clean.length; i += SLICE) {
    const bin = atob(clean.slice(i, i + SLICE));
    for (let k = 0; k < bin.length; k++) out[o++] = bin.charCodeAt(k);
  }
  return out;
}

async function readWithProgress(resp, onBytes) {
  const reader = resp.body && resp.body.getReader ? resp.body.getReader() : null;
  if (!reader) {
    const b = new Uint8Array(await resp.arrayBuffer());
    onBytes(b.length);
    return b;
  }
  const parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    n += value.length;
    onBytes(value.length);
  }
  const buf = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  return buf;
}

/** Fetch an asset by key. onProgress(fraction 0..1). Returns Uint8Array. */
export async function fetchAsset(key, onProgress) {
  const m = await getManifest();
  const a = m.assets[key];
  if (!a) throw new Error('missing asset ' + key);
  const enc = a.enc || 'raw';
  let cache = null;
  try { cache = self.caches ? await caches.open('guitar-tab-assets-' + m.version) : null; } catch { cache = null; }
  const total = a.dl || a.size;
  let loaded = 0;
  const bump = (n) => { loaded += n; if (onProgress) onProgress(Math.min(1, loaded / total)); };
  // several parts are written straight into one buffer of the final size (no second copy)
  const out = a.parts.length > 1 && a.size ? new Uint8Array(a.size) : null;
  let o = 0, single = null;
  for (const p of a.parts) {
    const url = BASE + p;
    const key2 = url + '#bytes';
    let buf = null;
    if (cache) {
      try {
        const hit = await cache.match(key2);
        if (hit) { buf = new Uint8Array(await hit.arrayBuffer()); }
      } catch { buf = null; }
    }
    if (buf) {
      bump(enc === 'b64' ? Math.ceil(buf.length / 3) * 4 : buf.length);
    } else {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`Could not download ${key} (${r.status})`);
      const raw = await readWithProgress(r, bump);
      buf = enc === 'b64' ? decodeBase64(new TextDecoder().decode(raw)) : raw;
      if (cache) { try { await cache.put(key2, new Response(buf)); } catch { /* storage full or blocked */ } }
    }
    if (!out) { single = buf; break; }
    if (o + buf.length > out.length) throw new Error(`Downloaded ${key} has the wrong size`);
    out.set(buf, o);
    o += buf.length;
  }
  return out || single;
}

/**
 * Download assets into the cache ahead of time (while the user is still choosing options), one
 * part at a time, skipping what is cached already. Stops between parts when shouldStop().
 */
export async function prefetchAssets(keys, { shouldStop } = {}) {
  const m = await getManifest();
  let cache = null;
  try { cache = self.caches ? await caches.open('guitar-tab-assets-' + m.version) : null; } catch { cache = null; }
  if (!cache) return;
  for (const key of keys) {
    const a = m.assets[key];
    if (!a) continue;
    for (const p of a.parts) {
      if (shouldStop && shouldStop()) return;
      const url = BASE + p;
      try {
        if (await cache.match(url + '#bytes')) continue;
        const r = await fetch(url);
        if (!r.ok) return;
        const raw = new Uint8Array(await r.arrayBuffer());
        const buf = a.enc === 'b64' ? decodeBase64(new TextDecoder().decode(raw)) : raw;
        await cache.put(url + '#bytes', new Response(buf));
      } catch { return; }
    }
  }
}

export async function hasAsset(key) {
  try { const m = await getManifest(); return !!m.assets[key]; } catch { return false; }
}
