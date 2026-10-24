// Render a saved tab with every tab-player sound to WAV files (headless Chromium).
// usage: node tests/synth_render.mjs <app url> <tab.json> <out prefix> [from] [to]
// tab.json: {alphaTex, syncPoints} (e.g. from a saved project). env: PLAYWRIGHT_CORE, as for browser_e2e.mjs
import fs from 'node:fs';
const pw = await import(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const { chromium } = pw.chromium ? pw : pw.default;
const [url, tabFile, prefix, from = '0', to = '20'] = process.argv.slice(2);
const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
const page = await (await browser.newContext()).newPage();
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('[' + m.type() + ']', m.text().slice(0, 300)); });
page.on('pageerror', (e) => console.log('PAGEERROR', e.message));
await page.goto(url + 'tests/synth_render.html');
await page.waitForFunction(() => document.getElementById('log').textContent.includes('ready'));
const res = await page.evaluate(([tab, a, b]) => window.renderAll(tab, a, b), [JSON.parse(fs.readFileSync(tabFile, 'utf8')), Number(from), Number(to)]);
console.log('events', res.events, JSON.stringify(res.first));
for (const [tone, r] of Object.entries(res.out)) {
  if (r.error) { console.log(tone, 'ERROR', r.error.slice(0, 400)); continue; }
  const pcm = Buffer.from(r.b64, 'base64');
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + pcm.length, 4); h.write('WAVE', 8); h.write('fmt ', 12); h.writeUInt32LE(16, 16);
  h.writeUInt16LE(1, 20); h.writeUInt16LE(2, 22); h.writeUInt32LE(r.sr, 24); h.writeUInt32LE(r.sr * 4, 28); h.writeUInt16LE(4, 32); h.writeUInt16LE(16, 34);
  h.write('data', 36); h.writeUInt32LE(pcm.length, 40);
  fs.writeFileSync(`${prefix}-${tone}.wav`, Buffer.concat([h, pcm]));
  console.log(tone, 'peak', r.peak.toFixed(3), 'render ms', r.ms);
}
await browser.close();
