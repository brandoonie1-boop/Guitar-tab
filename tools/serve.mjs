// Tiny static server for local testing. usage: node tools/serve.mjs [port] [--isolate]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const ri = process.argv.indexOf('--root');
const ROOT = ri >= 0 ? path.resolve(process.argv[ri + 1]) : path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const port = Number(process.argv[2] || 8080);
const isolate = process.argv.includes('--isolate');
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.png': 'image/png', '.wasm': 'application/wasm', '.bin': 'application/octet-stream', '.woff2': 'font/woff2', '.woff': 'font/woff', '.sf3': 'application/octet-stream', '.webmanifest': 'application/manifest+json', '.svg': 'image/svg+xml', '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.txt': 'text/plain; charset=utf-8' };
http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x');
  let p = decodeURIComponent(u.pathname);
  if (p.endsWith('/')) p += 'index.html';
  const f = path.join(ROOT, p);
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); res.end('not found'); return; }
  const h = { 'Content-Type': types[path.extname(f)] || 'application/octet-stream', 'Cache-Control': 'no-cache' };
  if (isolate) { h['Cross-Origin-Opener-Policy'] = 'same-origin'; h['Cross-Origin-Embedder-Policy'] = 'require-corp'; }
  res.writeHead(200, h);
  fs.createReadStream(f).pipe(res);
}).listen(port, () => console.log('serving', ROOT, 'on', port, isolate ? '(isolated)' : ''));
