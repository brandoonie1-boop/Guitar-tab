// Getting a song in from a link or a recording.
//
// Browsers don't let a page download audio from other sites unless the site allows it, and
// streaming sites (YouTube, Spotify, ...) don't. So a link is handled one of three ways:
// - a direct link to an audio/video file is downloaded when the host allows it;
// - on a computer, the song is recorded while it plays: the YouTube video right inside the
//   app, or any other tab / the computer's sound, through the browser's screen-share picker;
// - anywhere (including iPhone), the app can listen through the microphone while the song
//   plays on another device, or the user picks an iPhone screen recording of it.
// Everything stays on the device.

export const IS_IOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
export const canCaptureTab = () => !IS_IOS && !!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);
export const canListen = () => !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
// a YouTube player can sit inside this (cross-origin isolated) page only as a credentialless iframe
export const canEmbedPlayer = () => typeof HTMLIFrameElement !== 'undefined' && 'credentialless' in HTMLIFrameElement.prototype;

const MEDIA_EXT = /\.(mp3|m4a|aac|wav|wave|flac|ogg|oga|opus|webm|weba|mp4|m4v|mov|aif|aiff|caf|3gp)$/i;

function parseStart(t) {
  if (!t) return 0;
  const m = String(t).match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$/);
  if (!m) return 0;
  return (Number(m[1] || 0) * 3600) + (Number(m[2] || 0) * 60) + Number(m[3] || 0);
}

/** What kind of link is this? Returns {kind, ...}. */
export function classifyLink(text) {
  let u;
  try { u = new URL(String(text || '').trim()); } catch { return { kind: 'invalid' }; }
  if (!/^https?:$/.test(u.protocol)) return { kind: 'invalid' };
  const host = u.hostname.toLowerCase().replace(/^(www|m)\./, '');
  let yt = null;
  if (host === 'youtu.be') yt = u.pathname.slice(1).split('/')[0];
  else if (host === 'youtube.com' || host === 'music.youtube.com' || host === 'youtube-nocookie.com') {
    if (u.pathname === '/watch') yt = u.searchParams.get('v');
    else { const m = u.pathname.match(/^\/(?:embed|shorts|live|v)\/([\w-]{11})/); if (m) yt = m[1]; }
  }
  if (yt && /^[\w-]{11}$/.test(yt)) return { kind: 'youtube', id: yt, start: parseStart(u.searchParams.get('t') || u.searchParams.get('start')), url: `https://www.youtube.com/watch?v=${yt}` };
  if (/(^|\.)spotify\.com$/.test(host)) return { kind: 'stream', service: 'Spotify', url: u.href, protected: true };
  if (host === 'music.apple.com') return { kind: 'stream', service: 'Apple Music', url: u.href, protected: true };
  if (/(^|\.)tidal\.com$/.test(host)) return { kind: 'stream', service: 'Tidal', url: u.href, protected: true };
  if (/(^|\.)deezer\.com$/.test(host)) return { kind: 'stream', service: 'Deezer', url: u.href, protected: true };
  if (/(^|\.)soundcloud\.com$/.test(host)) return { kind: 'stream', service: 'SoundCloud', url: u.href };
  if (/(^|\.)bandcamp\.com$/.test(host)) return { kind: 'stream', service: 'Bandcamp', url: u.href };
  if (host.endsWith('dropbox.com')) {
    u.searchParams.delete('dl');
    u.searchParams.set('raw', '1');
    return { kind: 'direct', url: u.href, share: 'Dropbox' };
  }
  if (MEDIA_EXT.test(u.pathname)) return { kind: 'direct', url: u.href };
  return { kind: 'page', url: u.href, service: host };
}

/** Download a direct link to a media file (only works when the host allows it). */
export async function fetchMedia(url, { onProgress, maxBytes = 2e9 } = {}) {
  let r;
  try {
    r = await fetch(url, { mode: 'cors', credentials: 'omit', redirect: 'follow' });
  } catch {
    const e = new Error('blocked');
    e.code = 'blocked';
    throw e;
  }
  if (!r.ok) { const e = new Error(`The link answered with error ${r.status}.`); e.code = 'http'; throw e; }
  const type = (r.headers.get('content-type') || '').toLowerCase();
  if (/text\/html/.test(type)) { const e = new Error('That link is a web page, not an audio or video file.'); e.code = 'page'; throw e; }
  const total = Number(r.headers.get('content-length') || 0);
  if (total > maxBytes) { const e = new Error(`That file is too big (${Math.round(total / 1e6)} MB).`); e.code = 'size'; throw e; }
  const reader = r.body && r.body.getReader ? r.body.getReader() : null;
  let blob;
  if (reader) {
    const parts = [];
    let n = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      n += value.length;
      if (n > maxBytes) { reader.cancel(); const e = new Error('That file is too big.'); e.code = 'size'; throw e; }
      if (onProgress) onProgress(total ? n / total : 0, n);
    }
    blob = new Blob(parts, { type });
  } else {
    blob = await r.blob();
  }
  let name = decodeURIComponent(new URL(r.url || url).pathname.split('/').pop() || 'linked audio');
  const cd = r.headers.get('content-disposition') || '';
  const m = cd.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i);
  if (m) { try { name = decodeURIComponent(m[1]); } catch { name = m[1]; } }
  return new File([blob], name, { type: blob.type || type || 'application/octet-stream' });
}

/** Ask the browser for a tab / screen to record, with its sound. */
export async function captureDisplay({ thisTab = false } = {}) {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: { frameRate: 1, width: { max: 640 }, height: { max: 480 } },
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, suppressLocalAudioPlayback: false },
    preferCurrentTab: thisTab,
    selfBrowserSurface: thisTab ? 'include' : 'exclude',
    surfaceSwitching: 'include',
    systemAudio: 'include',
  });
  if (!stream.getAudioTracks().length) {
    stream.getTracks().forEach((t) => t.stop());
    const e = new Error('No sound was shared. Try again and turn on "Share tab audio" (or "Share system audio").');
    e.code = 'no-audio';
    throw e;
  }
  return stream;
}

/** The microphone, with the phone's voice processing turned off (it would eat the music). */
export async function captureMic() {
  return navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: { ideal: 2 } },
  });
}

/** An AudioContext at 44.1 kHz if the browser allows it. Create it inside the tap (iPhone). */
export function makeAudioContext() {
  const AC = self.AudioContext || self.webkitAudioContext;
  let ctx;
  try { ctx = new AC({ sampleRate: 44100 }); } catch { ctx = new AC(); }
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});
  return ctx;
}

/**
 * Record a MediaStream's audio (44.1 kHz when the browser allows it).
 * Returns {stop() -> Promise<{left, right, sr, duration}>, cancel(), seconds()}.
 */
export async function recordStream(stream, { ctx = null, onLevel, onEnded, maxSeconds = 15 * 60 } = {}) {
  if (!ctx) ctx = makeAudioContext();
  if (ctx.state === 'suspended') { try { await ctx.resume(); } catch { /* ignore */ } }
  await ctx.audioWorklet.addModule(new URL('./rec-worklet.js', import.meta.url));
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'gt-recorder', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 2, channelCountMode: 'explicit' });
  const mute = ctx.createGain();
  mute.gain.value = 0;
  src.connect(node);
  node.connect(mute).connect(ctx.destination);
  const L = [], R = [];
  let frames = 0, stopped = false, flushWait = null;
  node.port.onmessage = (e) => {
    const { l, r, peak } = e.data;
    L.push(l); R.push(r);
    frames += l.length;
    if (onLevel) onLevel(peak, frames / ctx.sampleRate);
    if (flushWait) { const f = flushWait; flushWait = null; f(); }
    if (frames / ctx.sampleRate >= maxSeconds && onEnded) onEnded('limit');
  };
  const ended = () => { if (!stopped && onEnded) onEnded('ended'); };
  for (const t of stream.getTracks()) t.addEventListener('ended', ended);
  const close = () => {
    stopped = true;
    for (const t of stream.getTracks()) { t.removeEventListener('ended', ended); t.stop(); }
    try { src.disconnect(); node.disconnect(); } catch { /* ignore */ }
    ctx.close().catch(() => {});
  };
  return {
    seconds: () => frames / ctx.sampleRate,
    cancel: close,
    async stop() {
      if (stopped) throw new Error('already stopped');
      // collect what is still inside the worklet
      await new Promise((resolve) => { flushWait = resolve; node.port.postMessage('stop'); setTimeout(resolve, 600); });
      const sr = ctx.sampleRate;
      close();
      const join = (arr) => {
        const out = new Float32Array(frames);
        let o = 0;
        for (const c of arr) { out.set(c.subarray(0, Math.min(c.length, frames - o)), o); o += c.length; if (o >= frames) break; }
        return out;
      };
      const left = join(L), right = join(R);
      L.length = 0; R.length = 0;
      return { left, right, sr, duration: frames / sr };
    },
  };
}

/** Cut silence (below -50 dBFS) from the start and end of a recording, keeping 0.3 s. */
export function trimSilence(rec, thresh = 0.003) {
  const { left, right, sr } = rec;
  const n = left.length;
  let a = 0, b = n - 1;
  while (a < n && Math.abs(left[a]) < thresh && Math.abs(right[a]) < thresh) a++;
  while (b > a && Math.abs(left[b]) < thresh && Math.abs(right[b]) < thresh) b--;
  if (a >= b) return null; // nothing but silence
  const pad = Math.round(0.3 * sr);
  a = Math.max(0, a - pad); b = Math.min(n, b + pad);
  if (a === 0 && b === n) return rec;
  return { left: left.slice(a, b), right: right.slice(a, b), sr, duration: (b - a) / sr };
}

/**
 * A YouTube player inside the page, driven through YouTube's documented iframe messages.
 * onInfo({state, time, duration, title}); onError(code) (101/150: the owner doesn't allow embedding).
 */
export function youtubePlayer(container, id, { start = 0, onInfo = null, onError = null } = {}) {
  const iframe = document.createElement('iframe');
  iframe.setAttribute('credentialless', '');
  iframe.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
  iframe.title = 'YouTube video';
  iframe.referrerPolicy = 'strict-origin-when-cross-origin';
  const q = new URLSearchParams({ enablejsapi: '1', playsinline: '1', rel: '0', origin: location.origin });
  if (start) q.set('start', String(Math.floor(start)));
  iframe.src = `https://www.youtube-nocookie.com/embed/${id}?${q}`;
  container.append(iframe);
  const info = { state: -1, time: 0, duration: 0, title: '', ready: false };
  const post = (obj) => { try { iframe.contentWindow.postMessage(JSON.stringify(obj), '*'); } catch { /* ignore */ } };
  const send = (func, args = []) => post({ event: 'command', func, args, id: 1, channel: 'widget' });
  const onMsg = (e) => {
    if (e.source !== iframe.contentWindow) return;
    let d;
    try { d = typeof e.data === 'string' ? JSON.parse(e.data) : e.data; } catch { return; }
    if (!d || typeof d !== 'object' || !d.event) return;
    if (d.event === 'onReady') info.ready = true;
    if ((d.event === 'infoDelivery' || d.event === 'initialDelivery') && d.info) {
      info.ready = true;
      const i = d.info;
      if (typeof i.playerState === 'number') info.state = i.playerState;
      if (typeof i.currentTime === 'number') info.time = i.currentTime;
      if (typeof i.duration === 'number' && i.duration > 0) info.duration = i.duration;
      if (i.videoData && i.videoData.title) info.title = i.videoData.title;
    }
    if (d.event === 'onStateChange' && typeof d.info === 'number') info.state = d.info;
    if (d.event === 'onError') { if (api.onError) api.onError(d.info); return; }
    if (api.onInfo) api.onInfo(info);
  };
  window.addEventListener('message', onMsg);
  // the player answers once it hears from the page; keep saying hello until it does
  let tries = 0;
  const hello = setInterval(() => {
    if (info.ready || tries++ > 60) { clearInterval(hello); return; }
    post({ event: 'listening', id: 1, channel: 'widget' });
  }, 250);
  const api = {
    iframe,
    info,
    onInfo,
    onError,
    play: () => send('playVideo'),
    pause: () => send('pauseVideo'),
    seek: (t) => send('seekTo', [t, true]),
    unmute: () => { send('unMute'); send('setVolume', [100]); },
    destroy() { clearInterval(hello); window.removeEventListener('message', onMsg); iframe.remove(); },
  };
  return api;
}
