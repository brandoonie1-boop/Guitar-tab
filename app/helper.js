// The Guitar Tab link helper (helper/helper.ps1) runs on the user's own computer and
// downloads songs from links with yt-dlp. Browsers can't do that themselves (sites like
// YouTube don't hand their audio to other pages), and doing it from a web server breaks those
// sites' rules and gets blocked. The helper listens on 127.0.0.1 only and answers only this app.

export const HELPER_URL = 'http://127.0.0.1:47823';
export const HELPER_INSTALL_CMD = "[Net.ServicePointManager]::SecurityProtocol=3072; iex ((New-Object Net.WebClient).DownloadString('https://guitar-tab.onrender.com/helper/install.ps1'))";
export const HELPER_SETUP_FILE = 'helper/GuitarTabHelperSetup.cmd';
const H = { 'X-GuitarTab': '1' };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const IS_WINDOWS = /Windows/.test(navigator.userAgent);

async function call(path, timeoutMs = 15000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(HELPER_URL + path, { headers: H, cache: 'no-store', signal: ctl.signal });
    return r;
  } finally { clearTimeout(t); }
}

/** {helper, ytdlp, deno} when the helper is running on this computer, otherwise null. */
export async function helperPing(timeoutMs = 2500) {
  try {
    const r = await call('/ping', timeoutMs);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

function friendly(msg) {
  const m = String(msg || '');
  if (/confirm you.?re not a bot|Sign in to confirm/i.test(m)) return 'YouTube wants to check that this isn\'t a bot. Play any video on youtube.com in your browser, then try again in a minute.';
  if (/private video|members-only|Join this channel/i.test(m)) return 'That video is private or for channel members only.';
  if (/Video unavailable|not available|removed|terminated/i.test(m)) return 'That video is unavailable (removed, or blocked in your country).';
  if (/age|inappropriate/i.test(m) && /confirm|sign in/i.test(m)) return 'That video is age-restricted, so YouTube only plays it for signed-in accounts.';
  if (/Unsupported URL/i.test(m)) return "The helper doesn't know how to get audio from that site. Try a YouTube, SoundCloud or Bandcamp link to the same song.";
  if (/DRM/i.test(m)) return 'That site protects its audio with DRM.';
  if (/larger than max-filesize|too big/i.test(m)) return 'That file is too big.';
  if (/Couldn't read the song's name/i.test(m)) return "Couldn't read the song's name from that page. Paste a YouTube link to the song instead.";
  if (/HTTP Error 4\d\d|Unable to download webpage/i.test(m)) return `The site refused the download (${m.replace(/^.*?(HTTP Error \d+[^;)]*).*$/, '$1')}).`;
  return m || 'The download did not finish.';
}

const EXT = { 'audio/mp4': 'm4a', 'video/mp4': 'mp4', 'audio/webm': 'webm', 'audio/mpeg': 'mp3', 'audio/ogg': 'ogg', 'audio/wav': 'wav', 'audio/flac': 'flac', 'audio/aac': 'aac' };

/**
 * Download a song through the helper. Returns a File.
 * @param onStatus ({progress 0..1, title}) while it downloads
 * @param isCancelled () => bool
 */
export async function helperDownload(url, { onStatus, isCancelled } = {}) {
  const r = await call('/start?url=' + encodeURIComponent(url), 30000);
  const j = await r.json().catch(() => ({}));
  if (!r.ok || !j.id) throw new Error(friendly(j.error) || 'The helper could not start the download.');
  const id = j.id;
  let s = null;
  for (let n = 0; ; n++) {
    if (isCancelled && isCancelled()) { call('/cancel?id=' + id).catch(() => {}); const e = new Error('cancelled'); e.code = 'cancelled'; throw e; }
    await sleep(n < 10 ? 400 : 800);
    const sr = await call('/status?id=' + id);
    s = await sr.json();
    if (onStatus) onStatus(s);
    if (s.state === 'done') break;
    if (s.state === 'error') throw new Error(friendly(s.error));
    if (n > 1500) throw new Error('The download is taking too long.');
  }
  const a = await call('/audio?id=' + id, 120000);
  if (!a.ok) throw new Error('The helper could not hand over the audio.');
  const blob = await a.blob();
  let title = '';
  try { title = decodeURIComponent(a.headers.get('X-Title') || ''); } catch { /* ignore */ }
  title = title || (s && s.title) || 'Song';
  const safe = title.replace(/[\\/:*?"<>|]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 100) || 'Song';
  const type = (blob.type || '').split(';')[0];
  return new File([blob], `${safe}.${EXT[type] || 'm4a'}`, { type: type || 'audio/mp4' });
}
