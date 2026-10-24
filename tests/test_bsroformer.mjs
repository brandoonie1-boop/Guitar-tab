// BS-RoFormer wrapper (engine/bsroformer.js) without the model: a stand-in "model" that hands the
// mix back as every stem must give the input back exactly (STFT / iSTFT, chunking, padding and the
// cross-fades all cancel out), for short and long songs and any overlap.
// usage: node tests/test_bsroformer.mjs
import { separateBsr, BSR, _internal } from '../engine/bsroformer.js';
import { soloProbe } from '../engine/pipeline.js';
import { spreadOrder } from '../engine/dsp.js';

const ort = { Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } } };
function identity(stems) {
  return {
    stems,
    async run(feeds) {
      const re = feeds.spec_real.data, im = feeds.spec_imag.data;
      const k = stems.length, n = re.length;
      const oRe = new Float32Array(k * n), oIm = new Float32Array(k * n);
      for (let s = 0; s < k; s++) { oRe.set(re, s * n); oIm.set(im, s * n); }
      return { out_spec_real: { data: oRe }, out_spec_imag: { data: oIm } };
    },
  };
}

let failed = 0;
const check = (name, ok, detail = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name} ${detail}`); if (!ok) failed++; };

// STFT -> iSTFT of one chunk is the identity
{
  const x = new Float32Array(BSR.chunk).map((_, i) => Math.sin(i * 0.01) * 0.5 + Math.sin(i * 0.37) * 0.2);
  const re = new Float32Array(2 * BSR.bins * BSR.frames), im = new Float32Array(2 * BSR.bins * BSR.frames);
  _internal.stftChannel(x, re, im, 0);
  const y = new Float32Array(BSR.chunk);
  _internal.istftChannel(re, im, 0, y);
  let e = 0, r = 0;
  for (let i = 0; i < x.length; i++) { e += (x[i] - y[i]) ** 2; r += x[i] ** 2; }
  check('stft/istft round trip', 10 * Math.log10(r / e) > 100, `${(10 * Math.log10(r / e)).toFixed(1)} dB`);
}

for (const [secs, overlap] of [[2.5, 0.25], [9.3, 0.25], [9.3, 0.1], [13, 0.5]]) {
  const n = Math.round(secs * BSR.sr);
  let seed = 3;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
  const L = new Float32Array(n).map((_, i) => 0.3 * Math.sin(i * 0.013) + 0.1 * rnd());
  const R = new Float32Array(n).map((_, i) => 0.3 * Math.sin(i * 0.017) + 0.1 * rnd());
  const out = await separateBsr(ort, identity(['guitar', 'other']), L, R, { keep: { guitar: 'stereo', other: 'mono' }, overlap });
  let e = 0, r = 0, em = 0;
  for (let i = 0; i < n; i++) {
    e += (out.guitar[0][i] - L[i]) ** 2 + (out.guitar[1][i] - R[i]) ** 2;
    r += L[i] ** 2 + R[i] ** 2;
    em += (out.other[0][i] - 0.5 * (L[i] + R[i])) ** 2;
  }
  const snr = 10 * Math.log10(r / e), snrM = 10 * Math.log10(r / 2 / em);
  check(`identity model, ${secs} s, overlap ${overlap}`, snr > 90 && snrM > 90, `stereo ${snr.toFixed(1)} dB, mono ${snrM.toFixed(1)} dB`);
}

// the early look at a few chunks (engine/pipeline.js soloProbe): a model that keeps everything
// (a guitar on its own) stops after the probe chunks; one that takes half away runs to the end
// and gives the same result as without the look (chunks add up in any order)
{
  const n = Math.round(30 * BSR.sr);
  const L = new Float32Array(n).map((_, i) => 0.3 * Math.sin(i * 0.013));
  const R = new Float32Array(n).map((_, i) => 0.3 * Math.sin(i * 0.017));
  let runs = 0;
  const counted = (m) => ({ ...m, run: (f) => { runs++; return m.run(f); } });
  const p1 = soloProbe();
  const r1 = await separateBsr(ort, counted(identity(['guitar'])), L, R, { keep: { guitar: 'stereo' }, probe: p1 });
  check('guitar alone: stops after the probe chunks', r1.alone === true && runs === p1.count, `${runs} chunks, ${p1.db && p1.db.toFixed(0)} dB`);
  const half = (m) => ({ ...m, async run(f) { const r = await m.run(f); r.out_spec_real.data.forEach((v, i, a) => { a[i] = v * 0.5; }); r.out_spec_imag.data.forEach((v, i, a) => { a[i] = v * 0.5; }); return r; } });
  runs = 0;
  const r2 = await separateBsr(ort, counted(half(identity(['guitar']))), L, R, { keep: { guitar: 'stereo' }, probe: soloProbe() });
  const all = runs;
  const r3 = await separateBsr(ort, half(identity(['guitar'])), L, R, { keep: { guitar: 'stereo' } });
  let d = 0;
  for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(r2.guitar[0][i] - r3.guitar[0][i]), Math.abs(r2.guitar[1][i] - r3.guitar[1][i]));
  check('band: every chunk, same result in any order', !r2.alone && all > p1.count && d < 1e-6, `${all} chunks, max difference ${d.toExponential(1)}`);
  const ord = spreadOrder(40, 5);
  check('spread order: every chunk once', ord.length === 40 && new Set(ord).size === 40 && ord.slice(0, 5).join() === '4,12,20,28,36', ord.slice(0, 6).join());
}

// a stem the model doesn't have is an error, not silence
try {
  await separateBsr(ort, identity(['guitar']), new Float32Array(1000), new Float32Array(1000), { keep: { piano: 'mono' } });
  check('unknown stem rejected', false);
} catch (err) { check('unknown stem rejected', /no piano stem/.test(err.message)); }

console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
