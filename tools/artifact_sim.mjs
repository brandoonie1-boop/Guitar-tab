// Lay out dist-artifact the way the Claude artifact host serves it (only published files,
// page wrapped in a minimal skeleton) so it can be tested locally.
// usage: node tools/artifact_sim.mjs <out dir>
import fs from 'node:fs';
import path from 'node:path';
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const out = path.resolve(process.argv[2] || '/tmp/claude-0/artsim');
fs.rmSync(out, { recursive: true, force: true });
const files = JSON.parse(fs.readFileSync(path.join(ROOT, 'dist-artifact/files.json'), 'utf8'));
for (const [pub, src] of Object.entries(files)) {
  const dst = path.join(out, pub);
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(path.join(ROOT, src), dst);
}
const page = fs.readFileSync(path.join(ROOT, 'dist-artifact/index.html'), 'utf8');
fs.writeFileSync(path.join(out, 'index.html'), `<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}</style></head><body>${page}</body></html>`);
console.log('artifact sim at', out, Object.keys(files).length, 'files');
