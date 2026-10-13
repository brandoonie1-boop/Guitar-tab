// End-to-end browser test with Playwright: load the app, transcribe a file, check the tab, edit a note.
// usage: node tests/browser_e2e.mjs <url> <audio file> [chromium|webkit] [--mobile] [--shots dir]
import { chromium, webkit } from '/home/claude/work/jsenv/node_modules/playwright-core/index.mjs';

const [url, file, browserName = 'chromium'] = process.argv.slice(2);
const mobile = process.argv.includes('--mobile');
const si = process.argv.indexOf('--shots');
const shots = si >= 0 ? process.argv[si + 1] : '/tmp/claude-0/shots';
const bt = browserName === 'webkit' ? webkit : chromium;
const browser = await bt.launch({ args: browserName === 'chromium' ? ['--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'] : [] });
const ctx = await browser.newContext(mobile ? { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: browserName !== 'webkit' ? true : undefined, hasTouch: true } : { viewport: { width: 1280, height: 900 } });
const page = await ctx.newPage();
const logs = [];
page.on('console', (m) => { const t = `[${m.type()}] ${m.text()}`; logs.push(t); if (m.type() === 'error' || m.type() === 'warning' || /processed with|GPU/.test(m.text())) console.log(t.slice(0, 300)); });
page.on('pageerror', (e) => console.log('PAGEERROR', e.message));
const t0 = Date.now();
const el = (s) => ((Date.now() - t0) / 1000).toFixed(1) + 's ' + s;
await page.goto(url);
await page.waitForTimeout(1500);
// service worker reload for isolation
await page.waitForLoadState('load');
await page.waitForTimeout(1500);
console.log(el('isolated: ' + await page.evaluate(() => self.crossOriginIsolated)), 'gpu row visible:', await page.evaluate(() => !document.getElementById('gpuRow').hidden), 'navigator.gpu:', await page.evaluate(() => !!navigator.gpu));
await page.screenshot({ path: `${shots}/${browserName}${mobile ? '-m' : ''}-1-start.png`, fullPage: true });
await page.setInputFiles('#fileInput', file);
await page.waitForFunction(() => !document.getElementById('startBtn').disabled, null, { timeout: 60000 });
console.log(el('decoded: ' + await page.textContent('#pickedInfo')));
await page.click('#startBtn');
let last = '';
const deadline = Date.now() + 20 * 60 * 1000;
while (Date.now() < deadline) {
  const done = await page.evaluate(() => !document.getElementById('screenResult').hidden);
  const err = await page.evaluate(() => { const e = document.getElementById('startError'); return e && !e.hidden ? e.textContent : null; });
  if (err) { console.log(el('ERROR: ' + err)); break; }
  if (done) break;
  const st = await page.textContent('#workStatus');
  if (st !== last) { console.log(el(st)); last = st; }
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(3000);
console.log(el('result: ' + (await page.textContent('#facts'))));
console.log('setup:', (await page.textContent('#setupBox')).slice(0, 300));
const svgCount = await page.evaluate(() => document.querySelectorAll('#tab svg').length);
console.log('alphaTab svg elements:', svgCount);
await page.screenshot({ path: `${shots}/${browserName}${mobile ? '-m' : ''}-2-result.png`, fullPage: false });
await page.screenshot({ path: `${shots}/${browserName}${mobile ? '-m' : ''}-3-result-full.png`, fullPage: true });
// text view
await page.click('#viewText');
const ascii = await page.textContent('#ascii');
console.log(ascii.split('\n').slice(0, 16).join('\n'));
await page.click('#viewTab');
// try clicking a note: find a note glyph via alphaTab bounds lookup
const clicked = await page.evaluate(() => {
  const g = document.querySelector('#tab .at-surface') || document.querySelector('#tab');
  return !!g;
});
console.log('surface', clicked);
if (process.argv.includes('--interact')) {
  // click on a fret number in the rendered tab (partials render lazily when scrolled into view)
  await page.evaluate(() => document.getElementById('tab').scrollIntoView({ block: 'start' }));
  await page.waitForTimeout(1500);
  const box = await page.evaluate(() => {
    const texts = Array.from(document.querySelectorAll('#tab svg text')).filter((t) => /^\d+$/.test(t.textContent.trim()));
    const t = texts[Math.min(5, texts.length - 1)];
    if (!t) return null;
    t.scrollIntoView({ block: 'center' });
    const r = t.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2, n: texts.length, txt: t.textContent };
  });
  console.log('fret texts', box);
  if (!box) {
    const info = await page.evaluate(() => {
      const tags = {};
      for (const e of document.querySelectorAll('#tab *')) tags[e.tagName] = (tags[e.tagName] || 0) + 1;
      const svg = document.querySelectorAll('#tab svg')[1];
      return { tags, sample: svg ? svg.outerHTML.slice(0, 1200) : document.querySelector('#tab').innerHTML.slice(0, 1200) };
    });
    console.log(JSON.stringify(info.tags)); console.log(info.sample);
  }
  if (box) {
    await page.waitForTimeout(300);
    const b2 = await page.evaluate(() => {
      const texts = Array.from(document.querySelectorAll('#tab svg text')).filter((t) => /^\d+$/.test(t.textContent.trim()));
      const r = texts[Math.min(5, texts.length - 1)].getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    });
    await page.mouse.click(b2.x, b2.y);
    await page.waitForTimeout(800);
    const sheet = await page.evaluate(() => { const s = document.querySelector('.edit-sheet'); return s ? s.innerText : null; });
    console.log('edit sheet:', sheet && sheet.replace(/\n/g, ' | '));
    await page.screenshot({ path: `${shots}/${browserName}${mobile ? '-m' : ''}-4-edit.png` });
    if (sheet) {
      const chips = await page.$$('.edit-sheet .chip');
      const other = [];
      for (const c of chips) if ((await c.getAttribute('aria-pressed')) === 'false') other.push(c);
      if (other.length) { await other[0].click(); await page.waitForTimeout(1500); console.log('moved note; undo enabled:', !(await page.isDisabled('#undoBtn'))); }
    }
  }
  // switch to an alternative setup
  const alts = await page.$$('#setupBox .alt');
  if (alts.length > 1) { await alts[1].click(); await page.waitForTimeout(2500); console.log('after alt:', (await page.textContent('#facts'))); }
  await page.click('#undoBtn'); await page.waitForTimeout(2000);
  console.log('after undo:', (await page.textContent('#facts')));
  // export GP
  await page.click('#exportBtn');
  await page.waitForTimeout(300);
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 15000 }).catch(() => null), page.click('.dialog button:has-text("Guitar Pro")')]);
  console.log('gp download:', dl ? dl.suggestedFilename() : 'none');
  if (dl) { const p = await dl.path(); const fs = await import('node:fs'); console.log('gp bytes', fs.statSync(p).size); }
  const [dl2] = await Promise.all([page.waitForEvent('download', { timeout: 30000 }).catch(() => null), page.click('.dialog button:has-text("Everything")')]);
  console.log('zip download:', dl2 ? dl2.suggestedFilename() : 'none');
  if (dl2) { const p = await dl2.path(); const fs = await import('node:fs'); console.log('zip bytes', fs.statSync(p).size); fs.copyFileSync(p, `${shots}/export.zip`); }
  await page.click('.dialog button:has-text("Done")');
  // play recording for 2 seconds
  await page.click('#playBtn'); await page.waitForTimeout(2500);
  console.log('time label while playing:', await page.textContent('#timeLbl'));
  await page.click('#playBtn');
  await page.selectOption('#srcSel', 'synth'); await page.waitForTimeout(3000);
  await page.click('#playBtn'); await page.waitForTimeout(2500);
  console.log('time label synth:', await page.textContent('#timeLbl'));
  await page.click('#playBtn');
  // library
  await page.click('#navLibrary'); await page.waitForTimeout(1000);
  console.log('library:', (await page.textContent('#library')).slice(0, 120));
  await page.click('#library .btn:has-text("Open")'); await page.waitForTimeout(4000);
  console.log('reopened facts:', await page.textContent('#facts'));
  await page.screenshot({ path: `${shots}/${browserName}${mobile ? '-m' : ''}-5-reopened.png` });
}
await browser.close();
