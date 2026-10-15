// Service worker: offline app shell + cross-origin isolation (needed for multi-threaded processing).
// Big model files are cached by the processing worker itself (Cache Storage), on first use.
const VERSION = 'gt-shell-9c2a4b051c';
const SHELL = [
 "./",
 "./index.html",
 "manifest.webmanifest",
 "assets.json",
 "app/assets.js",
 "app/capture.js",
 "app/files.js",
 "app/main.js",
 "app/rec-worklet.js",
 "app/store.js",
 "app/tabview.js",
 "app/worker.js",
 "engine/basicpitch.js",
 "engine/beats.js",
 "engine/beatthis.js",
 "engine/demucs.js",
 "engine/dsp.js",
 "engine/export.js",
 "engine/fft.js",
 "engine/fretboard.js",
 "engine/notes.js",
 "engine/pipeline.js",
 "engine/pitchref.js",
 "engine/rhythm.js",
 "engine/spectral.js",
 "engine/stereo.js",
 "engine/techniques.js",
 "engine/tunings.js",
 "engine/wav.js",
 "vendor/ort/ort.wasm.bundle.min.mjs",
 "vendor/ort/ort.webgpu.bundle.min.mjs",
 "vendor/alphatab/alphaTab.mjs",
 "vendor/alphatab/alphaTab.core.mjs",
 "vendor/alphatab/alphaTab.worker.mjs",
 "vendor/alphatab/alphaTab.worklet.mjs",
 "vendor/alphatab/font/Bravura.woff2",
 "vendor/alphatab/font/Bravura.woff",
 "vendor/alphatab/soundfont/sonivox.sf3",
 "icons/icon-192.png",
 "icons/icon-512.png",
 "icons/apple-touch-icon.png"
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) {
      if (k.startsWith('gt-shell-') && k !== VERSION) await caches.delete(k);
    }
    await self.clients.claim();
  })());
});

function isolate(resp) {
  if (!resp || resp.status === 0 || resp.type === 'opaque') return resp;
  const h = new Headers(resp.headers);
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: h });
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  e.respondWith((async () => {
    const cache = await caches.open(VERSION);
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) {
      // refresh the shell in the background
      if (req.mode === 'navigate') e.waitUntil(fetch(req).then((r) => { if (r.ok) cache.put(req, r.clone()); }).catch(() => {}));
      return isolate(hit);
    }
    try {
      const r = await fetch(req);
      return isolate(r);
    } catch (err) {
      if (req.mode === 'navigate') {
        const shell = await cache.match('./index.html') || await cache.match('./');
        if (shell) return isolate(shell);
      }
      throw err;
    }
  })());
});
