// Streaming ONNX pruner (engine/onnxprune.js) on a small hand-made model: nodes and weights that
// the kept outputs don't need go, the rest is unchanged; any chunking of the stream gives the same
// bytes. (On the real separator the result is checked against the cut's SHA-256 in app/models.js.)
// usage: node tests/test_onnxprune.mjs
import { pruneStream, parseFields, encodeFields } from '../engine/onnxprune.js';

const LEN = 2, VARINT = 0;
const enc = new TextEncoder(), dec = new TextDecoder();
const s = (x) => enc.encode(x);
const node = (op, ins, outs, name) => [
  ...ins.map((i) => ({ f: 1, wt: LEN, v: s(i) })), ...outs.map((o) => ({ f: 2, wt: LEN, v: s(o) })),
  { f: 3, wt: LEN, v: s(name) }, { f: 4, wt: LEN, v: s(op) },
];
const tensor = (name, n) => [{ f: 1, wt: VARINT, v: n }, { f: 2, wt: VARINT, v: 1 }, { f: 8, wt: LEN, v: s(name) }, { f: 9, wt: LEN, v: new Uint8Array(4 * n).fill(n) }];
const vi = (name) => [{ f: 1, wt: LEN, v: s(name) }];
const graph = [
  { f: 1, wt: LEN, v: node('Add', ['x', 'w1'], ['y1'], 'a') },
  { f: 1, wt: LEN, v: node('Mul', ['y1', 'w2'], ['out1'], 'b') },
  { f: 1, wt: LEN, v: node('Neg', ['x', 'w3'], ['out2'], 'c') },
  { f: 2, wt: LEN, v: s('g') },
  { f: 5, wt: LEN, v: tensor('w1', 3) },
  { f: 5, wt: LEN, v: tensor('w2', 5) },
  { f: 5, wt: LEN, v: tensor('w3', 7) },
  { f: 11, wt: LEN, v: vi('x') },
  { f: 12, wt: LEN, v: vi('out1') },
  { f: 12, wt: LEN, v: vi('out2') },
  { f: 13, wt: LEN, v: vi('y1') },
];
const model = [{ f: 1, wt: VARINT, v: 8 }, { f: 2, wt: LEN, v: s('test') }, { f: 7, wt: LEN, v: graph }, { f: 8, wt: LEN, v: [{ f: 2, wt: VARINT, v: 17 }] }];
const bytes = (() => { const p = encodeFields(model); const o = new Uint8Array(p.reduce((a, b) => a + b.length, 0)); let k = 0; for (const x of p) { o.set(x, k); k += x.length; } return o; })();
const plan = { edit() {}, outputs: { out1: [1, 3] } };
async function* chunks(b, size) { for (let i = 0; i < b.length; i += size) yield b.subarray(i, i + size); }

let failed = 0;
const check = (name, ok, detail = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name} ${detail}`); if (!ok) failed++; };

const whole = await pruneStream(chunks(bytes, 1 << 20), plan);
const tiny = await pruneStream(chunks(bytes, 1), plan);
check('same result for any chunking', whole.length === tiny.length && whole.every((v, i) => v === tiny[i]));
const top = parseFields(whole);
const g = parseFields(top.find((x) => x.f === 7).v);
const names = (f, sub) => g.filter((x) => x.f === f).map((x) => dec.decode(parseFields(x.v).find((y) => y.f === sub).v));
check('kept the nodes out1 needs', names(1, 3).join() === 'a,b', names(1, 3).join());
check('kept the weights out1 needs', names(5, 8).join() === 'w1,w2', names(5, 8).join());
check('outputs rewritten', names(12, 1).join() === 'out1');
check('other model fields kept', top.filter((x) => x.f !== 7).length === 3 && dec.decode(top.find((x) => x.f === 2).v) === 'test');
const w2 = g.filter((x) => x.f === 5).map((x) => parseFields(x.v)).find((t) => dec.decode(t.find((y) => y.f === 8).v) === 'w2');
check('weights unchanged', w2.find((y) => y.f === 9).v.every((v) => v === 5));

// a tensor whose data comes before its name can't be streamed: a clear error
const bad = encodeFields([{ f: 7, wt: LEN, v: [{ f: 1, wt: LEN, v: node('Add', ['x', 'w1'], ['out1'], 'a') }, { f: 5, wt: LEN, v: [{ f: 9, wt: LEN, v: new Uint8Array(8) }, { f: 8, wt: LEN, v: s('w1') }] }] }]);
const badBytes = new Uint8Array(bad.reduce((a, b) => a + b.length, 0)); { let k = 0; for (const x of bad) { badBytes.set(x, k); k += x.length; } }
try { await pruneStream(chunks(badBytes, 7), plan); check('unsupported layout reported', false); } catch (err) { check('unsupported layout reported', /not supported/.test(err.message)); }

console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
