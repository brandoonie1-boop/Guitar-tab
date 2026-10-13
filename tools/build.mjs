// Build: split big binaries into <=14 MB chunks, write assets.json, index.html, sw.js,
// and an artifact variant (content-only page) under dist-artifact/.
// usage: node tools/build.mjs [--ort <onnxruntime-web dist dir>]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const args = process.argv.slice(2);
const ortDir = args.includes('--ort') ? args[args.indexOf('--ort') + 1] : path.join(ROOT, 'node_modules/onnxruntime-web/dist');
const CHUNK = 14 * 1024 * 1024;

const r = (p) => path.join(ROOT, p);
const hash = (buf) => crypto.createHash('sha256').update(buf).digest('hex').slice(0, 10);

function chunkFile(src, name) {
  const buf = fs.readFileSync(src);
  const h = hash(buf);
  const parts = [];
  fs.mkdirSync(r('assets'), { recursive: true });
  for (let i = 0, k = 0; i < buf.length; i += CHUNK, k++) {
    const rel = `assets/${name}-${h}.${k}.bin`;
    const out = r(rel);
    if (!fs.existsSync(out)) fs.writeFileSync(out, buf.subarray(i, Math.min(buf.length, i + CHUNK)));
    parts.push(rel);
  }
  return { size: buf.length, sha: h, parts };
}

// remove stale chunks
if (fs.existsSync(r('assets'))) for (const f of fs.readdirSync(r('assets'))) fs.rmSync(r('assets/' + f));

const assets = {
  ortWasm: chunkFile(path.join(ortDir, 'ort-wasm-simd-threaded.wasm'), 'ort-wasm'),
  ortWasmGpu: chunkFile(path.join(ortDir, 'ort-wasm-simd-threaded.asyncify.wasm'), 'ort-wasm-gpu'),
  demucs: chunkFile(r('models/htdemucs_6s.f16.onnx'), 'demucs-6s'),
  basicpitch: chunkFile(r('models/basic_pitch.onnx'), 'basic-pitch'),
};
const version = hash(Buffer.from(JSON.stringify(assets) + fs.readFileSync(r('app/main.js')) + fs.readFileSync(r('engine/pipeline.js'))));
const manifest = {
  version,
  ort: { wasm: 'vendor/ort/ort.wasm.bundle.min.mjs', webgpu: 'vendor/ort/ort.webgpu.bundle.min.mjs' },
  assets,
};
fs.writeFileSync(r('assets.json'), JSON.stringify(manifest, null, 1));

const css = fs.readFileSync(r('app/styles.css'), 'utf8');
const body = fs.readFileSync(r('app/body.html'), 'utf8');
const pwaHead = `<meta name="theme-color" content="#2c5fb3">
<meta name="description" content="Turn any song into guitar tab. Isolates the guitar, works out the tuning and capo, and writes the tab, right in your browser.">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" href="icons/icon-192.png" sizes="192x192">
<link rel="apple-touch-icon" href="icons/apple-touch-icon.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Guitar Tab">
<meta name="apple-mobile-web-app-status-bar-style" content="default">`;

const pwa = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Guitar Tab</title>
${pwaHead}
<style>
${css}
:root { padding-top: env(safe-area-inset-top, 0px); }
</style>
</head>
<body>
${body}
<script type="module" src="app/main.js?v=${version}"></script>
</body>
</html>
`;
fs.writeFileSync(r('index.html'), pwa);

// service worker with the app shell list
const shell = ['./', './index.html', 'manifest.webmanifest', 'assets.json',
  ...['app', 'engine'].flatMap((d) => fs.readdirSync(r(d)).filter((f) => f.endsWith('.js')).map((f) => `${d}/${f}`)),
  'vendor/ort/ort.wasm.bundle.min.mjs', 'vendor/ort/ort.webgpu.bundle.min.mjs',
  'vendor/alphatab/alphaTab.mjs', 'vendor/alphatab/alphaTab.core.mjs', 'vendor/alphatab/alphaTab.worker.mjs', 'vendor/alphatab/alphaTab.worklet.mjs',
  'vendor/alphatab/font/Bravura.woff2', 'vendor/alphatab/font/Bravura.woff', 'vendor/alphatab/soundfont/sonivox.sf3',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png'];
const sw = fs.readFileSync(r('tools/sw.template.js'), 'utf8').replace('__VERSION__', version).replace('__SHELL__', JSON.stringify(shell, null, 1));
fs.writeFileSync(r('sw.js'), sw);

// artifact page: content only (the viewer adds the document skeleton)
fs.mkdirSync(r('dist-artifact'), { recursive: true });
const artifact = `<title>Guitar Tab</title>
<style>
${css}
</style>
${body}
<script type="module" src="app/main.js?v=${version}"></script>
`;
fs.writeFileSync(r('dist-artifact/index.html'), artifact);
// list of files the artifact needs (published path -> source path)
const files = {};
const add = (p) => { files[p] = p; };
for (const d of ['app', 'engine']) for (const f of fs.readdirSync(r(d))) if (f.endsWith('.js')) add(`${d}/${f}`);
for (const p of ['assets.json', 'vendor/ort/ort.wasm.bundle.min.mjs', 'vendor/ort/ort.webgpu.bundle.min.mjs', 'vendor/alphatab/alphaTab.mjs', 'vendor/alphatab/alphaTab.core.mjs',
  'vendor/alphatab/alphaTab.worker.mjs', 'vendor/alphatab/alphaTab.worklet.mjs', 'vendor/alphatab/font/Bravura.woff2', 'vendor/alphatab/font/Bravura.woff', 'vendor/alphatab/soundfont/sonivox.sf3']) add(p);
for (const a of Object.values(assets)) for (const p of a.parts) add(p);
fs.writeFileSync(r('dist-artifact/files.json'), JSON.stringify(files, null, 1));
console.log('built version', version);
for (const [k, a] of Object.entries(assets)) console.log(`  ${k}: ${(a.size / 1e6).toFixed(1)} MB in ${a.parts.length} part(s)`);
