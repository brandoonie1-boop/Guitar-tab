// The link flow with the link helper (helper/helper.ps1) running: paste a link, the helper
// downloads it, the app transcribes it. usage: node tests/helper_e2e.mjs <app url> <link>
import { chromium } from '/home/claude/work/jsenv/node_modules/playwright-core/index.mjs';

const [url, link] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
page.on('console', (m) => { if (m.type() === 'error' || /helper|link/i.test(m.text())) console.log(`[${m.type()}] ${m.text()}`.slice(0, 200)); });
page.on('pageerror', (e) => console.log('PAGEERROR', e.message));
const t0 = Date.now();
const el = (s) => `${((Date.now() - t0) / 1000).toFixed(1)}s ${s}`;
await page.goto(url);
await page.waitForLoadState('load');
await page.waitForTimeout(2500);
await page.fill('#linkInput', link);
await page.click('#linkForm button[type=submit]');
let last = '';
const deadline = Date.now() + 15 * 60 * 1000;
while (Date.now() < deadline) {
  const s = await page.evaluate(() => ({
    panel: document.getElementById('sourcePanel').hidden ? '' : document.getElementById('sourcePanel').innerText.slice(0, 160),
    work: document.getElementById('screenWork').hidden ? '' : document.getElementById('workStatus').textContent,
    done: !document.getElementById('screenResult').hidden,
    err: document.getElementById('startError').hidden ? '' : document.getElementById('startError').textContent,
  }));
  const line = s.done ? 'RESULT' : (s.err || s.work || s.panel);
  if (line && line !== last) { console.log(el(line.replace(/\s+/g, ' '))); last = line; }
  if (s.done || s.err) break;
  await page.waitForTimeout(500);
}
console.log(el('title: ' + await page.textContent('#songTitle')), '|', (await page.textContent('#facts')).replace(/\s+/g, ' '));
await browser.close();
