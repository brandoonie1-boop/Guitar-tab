// Minimal WAV read/write.

export function decodeWav(buf) {
  const dv = new DataView(buf.buffer || buf, buf.byteOffset || 0, buf.byteLength);
  const str = (o, n) => String.fromCharCode(...new Uint8Array(dv.buffer, dv.byteOffset + o, n));
  if (str(0, 4) !== 'RIFF' || str(8, 4) !== 'WAVE') throw new Error('not a WAV file');
  let o = 12, fmt = null, data = null;
  while (o + 8 <= dv.byteLength) {
    const id = str(o, 4), size = dv.getUint32(o + 4, true);
    if (id === 'fmt ') fmt = { format: dv.getUint16(o + 8, true), channels: dv.getUint16(o + 10, true), sr: dv.getUint32(o + 12, true), bits: dv.getUint16(o + 22, true) };
    else if (id === 'data') data = { off: o + 8, size: Math.min(size, dv.byteLength - o - 8) };
    o += 8 + size + (size & 1);
  }
  if (!fmt || !data) throw new Error('bad WAV');
  if (fmt.format === 0xfffe) fmt.format = fmt.bits === 32 ? 3 : 1;
  const bps = fmt.bits / 8, nch = fmt.channels;
  const n = Math.floor(data.size / (bps * nch));
  const chans = Array.from({ length: nch }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nch; c++) {
      const p = data.off + (i * nch + c) * bps;
      let v;
      if (fmt.format === 3) v = bps === 8 ? dv.getFloat64(p, true) : dv.getFloat32(p, true);
      else if (bps === 2) v = dv.getInt16(p, true) / 32768;
      else if (bps === 3) { const x = dv.getUint8(p) | (dv.getUint8(p + 1) << 8) | (dv.getInt8(p + 2) << 16); v = x / 8388608; }
      else if (bps === 4) v = dv.getInt32(p, true) / 2147483648;
      else v = (dv.getUint8(p) - 128) / 128;
      chans[c][i] = v;
    }
  }
  return { sr: fmt.sr, channels: chans };
}

export function encodeWav(channels, sr) {
  const nch = channels.length, n = channels[0].length;
  const buf = new ArrayBuffer(44 + n * nch * 2);
  const dv = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); dv.setUint32(4, 36 + n * nch * 2, true); w(8, 'WAVE');
  w(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, nch, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * nch * 2, true); dv.setUint16(32, nch * 2, true); dv.setUint16(34, 16, true);
  w(36, 'data'); dv.setUint32(40, n * nch * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nch; c++) {
      const v = Math.max(-1, Math.min(1, channels[c][i]));
      dv.setInt16(o, v < 0 ? v * 32768 : v * 32767, true);
      o += 2;
    }
  }
  return new Uint8Array(buf);
}
