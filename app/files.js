// Reading audio files, writing WAV / ZIP, and handing files to the user.
import { encodeWav } from '../engine/wav.js';

export async function decodeFile(file) {
  const buf = await file.arrayBuffer();
  const Ctx = self.OfflineAudioContext || self.webkitOfflineAudioContext;
  if (!Ctx) throw new Error('This browser cannot decode audio.');
  const ctx = new Ctx(2, 44100, 44100);
  const audioBuffer = await new Promise((resolve, reject) => {
    let settled = false;
    const ok = (b) => { if (!settled) { settled = true; resolve(b); } };
    const bad = (e) => { if (!settled) { settled = true; reject(e || new Error('decode failed')); } };
    try {
      const p = ctx.decodeAudioData(buf, ok, bad);
      if (p && p.then) p.then(ok, bad);
    } catch (e) { bad(e); }
  });
  // the decoded channels themselves, not copies: they are handed (transferred) to the worker
  // later, so a long song is held in memory once rather than twice
  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : null;
  return { left, right, sr: audioBuffer.sampleRate, duration: audioBuffer.duration, channels: audioBuffer.numberOfChannels };
}

export function wavBlob(channels, sr = 44100) {
  return new Blob([encodeWav(channels, sr)], { type: 'audio/wav' });
}

// --- tiny ZIP writer (stored, no compression) ---
let CRC_TABLE = null;
function crc32(data) {
  if (!CRC_TABLE) {
    CRC_TABLE = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      CRC_TABLE[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export function makeZip(files) {
  const enc = new TextEncoder();
  const chunks = [];
  const central = [];
  let offset = 0;
  const now = new Date();
  const dosTime = ((now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1)) & 0xffff;
  const dosDate = (((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate()) & 0xffff;
  for (const f of files) {
    const name = enc.encode(f.name);
    const data = f.data instanceof Uint8Array ? f.data : typeof f.data === 'string' ? enc.encode(f.data) : new Uint8Array(f.data);
    const crc = crc32(data);
    const lh = new DataView(new ArrayBuffer(30));
    lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true); lh.setUint16(8, 0, true);
    lh.setUint16(10, dosTime, true); lh.setUint16(12, dosDate, true); lh.setUint32(14, crc, true);
    lh.setUint32(18, data.length, true); lh.setUint32(22, data.length, true); lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
    chunks.push(new Uint8Array(lh.buffer), name, data);
    const ch = new DataView(new ArrayBuffer(46));
    ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true); ch.setUint16(10, 0, true);
    ch.setUint16(12, dosTime, true); ch.setUint16(14, dosDate, true); ch.setUint32(16, crc, true);
    ch.setUint32(20, data.length, true); ch.setUint32(24, data.length, true); ch.setUint16(28, name.length, true);
    ch.setUint32(42, offset, true);
    central.push(new Uint8Array(ch.buffer), name);
    offset += 30 + name.length + data.length;
  }
  const cdSize = central.reduce((s, c) => s + c.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
  end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
  return new Blob([...chunks, ...central, new Uint8Array(end.buffer)], { type: 'application/zip' });
}

let downloadsCap;
async function getDownloads() {
  if (downloadsCap !== undefined) return downloadsCap;
  try {
    downloadsCap = self.claude && typeof self.claude.use === 'function' ? await self.claude.use('downloads') : null;
  } catch { downloadsCap = null; }
  return downloadsCap;
}

export function inArtifact() {
  return !!(self.claude && typeof self.claude.use === 'function');
}

// formats the Claude artifact viewer lets a page save directly
const ARTIFACT_OK = new Set(['txt', 'json', 'md', 'zip', 'csv', 'html', 'pdf']);

/**
 * Give a file to the user. Returns 'saved', 'shared', 'declined' or throws.
 * In the Claude artifact viewer only some extensions are allowed; callers zip the rest.
 */
export async function saveFile(filename, blob) {
  if (inArtifact()) {
    const dl = await getDownloads();
    if (!dl) throw new Error('Saving files is not available here.');
    const ext = filename.split('.').pop().toLowerCase();
    let name = filename, data = blob;
    if (!ARTIFACT_OK.has(ext)) {
      const bytes = new Uint8Array(await blob.arrayBuffer());
      data = makeZip([{ name: filename, data: bytes }]);
      name = filename.replace(/\.[^.]+$/, '') + '.zip';
    }
    try {
      await dl.save({ filename: name, data });
      return 'saved';
    } catch (e) {
      if (e && e.code === 'declined') return 'declined';
      throw new Error(e && e.message ? e.message : 'Could not save the file.');
    }
  }
  // iPhone / iPad: the share sheet ("Save to Files") is the friendliest path
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  if (isIOS && navigator.canShare) {
    try {
      const f = new File([blob], filename, { type: blob.type || 'application/octet-stream' });
      if (navigator.canShare({ files: [f] })) {
        await navigator.share({ files: [f] });
        return 'shared';
      }
    } catch (e) {
      if (e && e.name === 'AbortError') return 'declined';
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 4000);
  return 'saved';
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch { return false; }
}
