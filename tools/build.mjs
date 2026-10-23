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
  // guitar isolation: shared weights + the model split into parts (tools/pack_demucs.py),
  // full 7.8 s segments and a lighter 3.9 s version for devices short on memory
  demucsWeights: chunkFile(r('models/demucs.weights.bin'), 'demucs-weights'),
  demucsFull: chunkFile(r('models/demucs-343980.chain'), 'demucs-full'),
  demucsLite: chunkFile(r('models/demucs-171990.chain'), 'demucs-lite'),
  basicpitch: chunkFile(r('models/basic_pitch.onnx'), 'basic-pitch'),
  // notes: high-resolution guitar transcription (Riley et al., MIT), tools/export_hrgt.py
  hrgtWeights: chunkFile(r('models/hrgt.weights.bin'), 'hrgt-weights'),
  hrgtChain: chunkFile(r('models/hrgt.chain'), 'hrgt-chain'),
  // beats and bar lines: Beat This! small (CPJKU, MIT), tools/export_beat_this.py
  beatModel: chunkFile(r('models/beat_this_small.f16.onnx'), 'beat-this'),
};
// the version changes whenever any app or engine file changes, so cached copies refresh
const srcFiles = ['app', 'engine'].flatMap((d) => fs.readdirSync(r(d)).filter((f) => f.endsWith('.js') || f.endsWith('.html') || f.endsWith('.css')).sort().map((f) => fs.readFileSync(r(`${d}/${f}`))));
const version = hash(Buffer.concat([Buffer.from(JSON.stringify(assets)), ...srcFiles]));
const manifest = {
  version,
  ort: { wasm: 'vendor/ort/ort.wasm.bundle.min.mjs', webgpu: 'vendor/ort/ort.webgpu.bundle.min.mjs' },
  assets,
};
fs.writeFileSync(r('assets.json'), JSON.stringify(manifest, null, 1));

const css = fs.readFileSync(r('app/styles.css'), 'utf8');
const body = fs.readFileSync(r('app/body.html'), 'utf8');
const pwaHead = `<meta name="theme-color" content="#f4f6f9">
<meta name="color-scheme" content="only light">
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

// ---------------------------------------------------------------------------
// Artifact build (dist-artifact/): the Claude artifact host serves only common web types,
// so the engine goes out as a real .wasm file and models/soundfont as base64 text parts.
const AD = r('dist-artifact');
fs.rmSync(AD, { recursive: true, force: true });
fs.mkdirSync(path.join(AD, 'assets'), { recursive: true });
const B64_RAW_CHUNK = 9 * 1024 * 1024; // -> 12 MB of text per part (host limit 16 MB per text file)
function b64Asset(src, name) {
  const buf = fs.readFileSync(src);
  const h = hash(buf);
  const parts = [];
  let dl = 0;
  for (let i = 0, k = 0; i < buf.length; i += B64_RAW_CHUNK, k++) {
    const rel = `assets/${name}-${h}.${k}.b64.txt`;
    const txt = buf.subarray(i, Math.min(buf.length, i + B64_RAW_CHUNK)).toString('base64');
    fs.writeFileSync(path.join(AD, rel), txt);
    dl += txt.length;
    parts.push(rel);
  }
  return { size: buf.length, dl, sha: h, enc: 'b64', parts };
}
function wasmAsset(src, name) {
  const buf = fs.readFileSync(src);
  const h = hash(buf);
  const rel = `assets/${name}-${h}.wasm`;
  fs.copyFileSync(src, path.join(AD, rel));
  return { size: buf.length, sha: h, enc: 'raw', parts: [rel] };
}
const artAssets = {
  ortWasm: wasmAsset(path.join(ortDir, 'ort-wasm-simd-threaded.wasm'), 'ort-wasm'),
  // the GPU engine is over the host's 15 MB binary limit, so it ships as base64 text parts
  ortWasmGpu: b64Asset(path.join(ortDir, 'ort-wasm-simd-threaded.asyncify.wasm'), 'ort-wasm-gpu'),
  demucsWeights: b64Asset(r('models/demucs.weights.bin'), 'demucs-weights'),
  demucsFull: b64Asset(r('models/demucs-343980.chain'), 'demucs-full'),
  demucsLite: b64Asset(r('models/demucs-171990.chain'), 'demucs-lite'),
  basicpitch: b64Asset(r('models/basic_pitch.onnx'), 'basic-pitch'),
  hrgtWeights: b64Asset(r('models/hrgt.weights.bin'), 'hrgt-weights'),
  hrgtChain: b64Asset(r('models/hrgt.chain'), 'hrgt-chain'),
  beatModel: b64Asset(r('models/beat_this_small.f16.onnx'), 'beat-this'),
  soundfont: b64Asset(r('vendor/alphatab/soundfont/sonivox.sf3'), 'soundfont'),
};
fs.writeFileSync(path.join(AD, 'assets.json'), JSON.stringify({ version, artifact: true, ort: manifest.ort, assets: artAssets }, null, 1));
const artifact = `<title>Guitar Tab</title>
<style>
${css}
</style>
${body}
<script type="module" src="app/main.js"></script>
`;
fs.writeFileSync(path.join(AD, 'index.html'), artifact);
// files the artifact needs: published path -> source path (relative to the repo root)
const files = {};
for (const d of ['app', 'engine']) for (const f of fs.readdirSync(r(d))) if (f.endsWith('.js')) files[`${d}/${f}`] = `${d}/${f}`;
for (const p of ['vendor/ort/ort.wasm.bundle.min.mjs', 'vendor/ort/ort.webgpu.bundle.min.mjs', 'vendor/alphatab/alphaTab.mjs', 'vendor/alphatab/alphaTab.core.mjs',
  'vendor/alphatab/alphaTab.worker.mjs', 'vendor/alphatab/alphaTab.worklet.mjs', 'vendor/alphatab/font/Bravura.woff2', 'vendor/alphatab/font/Bravura.woff']) files[p] = p;
files['assets.json'] = 'dist-artifact/assets.json';
for (const a of Object.values(artAssets)) for (const p of a.parts) files[p] = `dist-artifact/${p}`;
fs.writeFileSync(path.join(AD, 'files.json'), JSON.stringify(files, null, 1));
console.log('built version', version);
for (const [k, a] of Object.entries(assets)) console.log(`  ${k}: ${(a.size / 1e6).toFixed(1)} MB in ${a.parts.length} part(s)`);
console.log('artifact parts:', Object.values(artAssets).reduce((n, a) => n + a.parts.length, 0), 'files', Object.keys(files).length);
