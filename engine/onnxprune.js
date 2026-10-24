// Streaming ONNX pruner: reads a model as it downloads and keeps only what some of its outputs
// need, without ever holding the whole file. Used to cut BS-RoFormer SW (six stems, 353 MB) down
// to the stems the app uses (guitar: 103 MB) on the device, straight from its Hugging Face file.
//
// Works on the protobuf wire format directly (no ONNX library). It relies on the usual field
// order of serialized models (graph nodes before initializers, a tensor's name before its data),
// and checks it.

const VARINT = 0, I64 = 1, LEN = 2, I32 = 5;
const enc = new TextEncoder(), dec = new TextDecoder();

/** Pulls bytes from an async iterator of Uint8Array chunks. */
class Reader {
  constructor(chunks) {
    this.it = chunks[Symbol.asyncIterator] ? chunks[Symbol.asyncIterator]() : chunks[Symbol.iterator]();
    this.buf = new Uint8Array(0);
    this.p = 0;
    this.pos = 0; // absolute position
  }

  async fill(n) {
    while (this.buf.length - this.p < n) {
      const { value, done } = await this.it.next();
      if (done) return false;
      const rest = this.buf.subarray(this.p);
      if (!rest.length) { this.buf = value; this.p = 0; continue; }
      const b = new Uint8Array(rest.length + value.length);
      b.set(rest); b.set(value, rest.length);
      this.buf = b; this.p = 0;
    }
    return true;
  }

  async varint() {
    let v = 0, mul = 1;
    for (;;) {
      if (!(await this.fill(1))) throw new Error('model file ended early');
      const c = this.buf[this.p++]; this.pos++;
      v += (c & 0x7f) * mul;
      if (!(c & 0x80)) return v;
      mul *= 128;
    }
  }

  async bytes(n) {
    const out = new Uint8Array(n);
    let o = 0;
    while (o < n) {
      if (this.p >= this.buf.length && !(await this.fill(1))) throw new Error('model file ended early');
      const k = Math.min(n - o, this.buf.length - this.p);
      out.set(this.buf.subarray(this.p, this.p + k), o);
      this.p += k; o += k; this.pos += k;
    }
    return out;
  }

  async skip(n) {
    while (n > 0) {
      if (this.p >= this.buf.length && !(await this.fill(1))) throw new Error('model file ended early');
      const k = Math.min(n, this.buf.length - this.p);
      this.p += k; n -= k; this.pos += k;
    }
  }

  async atEnd() { return !(await this.fill(1)); }
}

// ---- wire format helpers on whole messages ----

function readVarint(b, i) {
  let v = 0, mul = 1;
  for (;;) {
    const c = b[i++];
    v += (c & 0x7f) * mul;
    if (!(c & 0x80)) return [v, i];
    mul *= 128;
  }
}

/** Message bytes -> [{f, wt, v}] (LEN values are subarrays, varints numbers). */
export function parseFields(b) {
  const out = [];
  let i = 0;
  while (i < b.length) {
    let key;
    [key, i] = readVarint(b, i);
    const f = Math.floor(key / 8), wt = key & 7;
    let v;
    if (wt === VARINT) [v, i] = readVarint(b, i);
    else if (wt === I64) { v = b.subarray(i, i + 8); i += 8; }
    else if (wt === I32) { v = b.subarray(i, i + 4); i += 4; }
    else if (wt === LEN) { let l; [l, i] = readVarint(b, i); v = b.subarray(i, i + l); i += l; }
    else throw new Error('bad wire type ' + wt);
    out.push({ f, wt, v });
  }
  return out;
}

function varintBytes(v) {
  const out = [];
  do { let c = v % 128; v = Math.floor(v / 128); if (v > 0) c |= 0x80; out.push(c); } while (v > 0);
  return Uint8Array.from(out);
}

/** [{f, wt, v}] -> list of Uint8Array pieces (v: number | Uint8Array | nested field list). */
export function encodeFields(fields, out = []) {
  for (const { f, wt, v } of fields) {
    out.push(varintBytes(f * 8 + wt));
    if (wt === VARINT) out.push(varintBytes(v));
    else if (wt === I64 || wt === I32) out.push(v);
    else {
      const body = Array.isArray(v) ? encodeFields(v) : [v];
      out.push(varintBytes(body.reduce((s, x) => s + x.length, 0)));
      for (const x of body) out.push(x);
    }
  }
  return out;
}

const str = (v) => dec.decode(v);

// ONNX field numbers
const M_GRAPH = 7;
const G_NODE = 1, G_INIT = 5, G_INPUT = 11, G_OUTPUT = 12, G_VALUEINFO = 13;
const N_INPUT = 1, N_OUTPUT = 2, N_NAME = 3, N_OPTYPE = 4, N_ATTR = 5;
const A_NAME = 1, A_T = 5;
const T_DIMS = 1, T_DTYPE = 2, T_NAME = 8, T_RAW = 9;

function nodeInfo(bytes) {
  const fields = parseFields(bytes);
  const n = { fields, inputs: [], outputs: [], name: '', op: '' };
  for (const x of fields) {
    if (x.f === N_INPUT) n.inputs.push(str(x.v));
    else if (x.f === N_OUTPUT) n.outputs.push(str(x.v));
    else if (x.f === N_NAME) n.name = str(x.v);
    else if (x.f === N_OPTYPE) n.op = str(x.v);
  }
  return n;
}

/** A Constant node's int64 tensor value. */
function constInts(node) {
  for (const a of node.fields) {
    if (a.f !== N_ATTR) continue;
    const af = parseFields(a.v);
    const t = af.find((x) => x.f === A_T);
    if (!t) continue;
    const raw = parseFields(t.v).find((x) => x.f === T_RAW);
    if (!raw) continue;
    const r = raw.v;
    return Array.from(new BigInt64Array(r.buffer.slice(r.byteOffset, r.byteOffset + r.byteLength)), Number);
  }
  return null;
}

function setConstInts(node, values) {
  node.fields = node.fields.map((a) => {
    if (a.f !== N_ATTR) return a;
    const af = parseFields(a.v).map((x) => {
      if (x.f !== A_T) return x;
      const tf = parseFields(x.v).map((y) => (y.f === T_RAW ? { ...y, v: new Uint8Array(BigInt64Array.from(values.map(BigInt)).buffer) } : y));
      return { ...x, v: tf };
    });
    return { ...a, v: af };
  });
}

function valueInfo(name, dims, elemType = 1) {
  const tensorType = [{ f: 1, wt: VARINT, v: elemType }];
  if (dims) tensorType.push({ f: 2, wt: LEN, v: dims.map((d) => ({ f: 1, wt: LEN, v: [{ f: 1, wt: VARINT, v: d }] })) });
  return [{ f: 1, wt: LEN, v: enc.encode(name) }, { f: 2, wt: LEN, v: [{ f: 1, wt: LEN, v: tensorType }] }];
}

const attr = (name, type, f, v) => ({ f: N_ATTR, wt: LEN, v: [{ f: A_NAME, wt: LEN, v: enc.encode(name) }, { f, wt: f === A_I_ ? VARINT : LEN, v }, { f: A_TYPE, wt: VARINT, v: type }] });
const A_I_ = 3, A_TYPE = 20, T_INT = 2;
function makeNode(op, name, inputs, outputs, attrs = []) {
  const n = {
    fields: [
      ...inputs.map((s) => ({ f: N_INPUT, wt: LEN, v: enc.encode(s) })),
      ...outputs.map((s) => ({ f: N_OUTPUT, wt: LEN, v: enc.encode(s) })),
      { f: N_NAME, wt: LEN, v: enc.encode(name) },
      { f: N_OPTYPE, wt: LEN, v: enc.encode(op) },
      ...attrs,
    ],
    inputs, outputs, name, op,
  };
  return n;
}
const intsTensor = (vals) => [
  { f: T_DIMS, wt: VARINT, v: vals.length }, { f: T_DTYPE, wt: VARINT, v: 7 },
  { f: T_RAW, wt: LEN, v: new Uint8Array(BigInt64Array.from(vals.map(BigInt)).buffer) },
];

/**
 * Attention over a big batch, in groups: Einsum(q, k) -> Mul(scale) -> Softmax -> Einsum(., v)
 * becomes the same per slice of the batch (axis 0) and a Concat. Same numbers; the score tensor
 * (batch x heads x T x T) only ever exists for one slice at a time.
 */
function groupAttention(nodes, prefix, sizes) {
  const at = nodes.findIndex((n) => n.name === prefix + 'Einsum');
  const e1 = nodes[at], mul = nodes.find((n) => n.name === prefix + 'Mul');
  const sm = nodes.find((n) => n.name === prefix + 'Softmax'), e2 = nodes.find((n) => n.name === prefix + 'Einsum_1');
  if (!e1 || !mul || !sm || !e2 || mul.inputs[0] !== e1.outputs[0] || sm.inputs[0] !== mul.outputs[0] || e2.inputs[0] !== sm.outputs[0]) {
    throw new Error('unexpected separator model layout (' + prefix + ')');
  }
  const [q, k] = e1.inputs, v = e2.inputs[1], scale = mul.inputs[1], out = e2.outputs[0];
  const keepAttrs = (n) => n.fields.filter((x) => x.f === N_ATTR);
  const G = sizes.length;
  const p = prefix + 'g';
  const add = [makeNode('Constant', p + 'sizes', [], [p + 'sizes'], [attr('value', 4, A_T, intsTensor(sizes))])];
  for (const [nm, src] of [['q', q], ['k', k], ['v', v]]) {
    add.push(makeNode('Split', p + 'split_' + nm, [src, p + 'sizes'], sizes.map((_, g) => `${p}${nm}${g}`), [attr('axis', T_INT, A_I_, 0)]));
  }
  for (let g = 0; g < G; g++) {
    add.push(makeNode('Einsum', `${p}${g}_qk`, [`${p}q${g}`, `${p}k${g}`], [`${p}${g}_s`], keepAttrs(e1)));
    add.push(makeNode('Mul', `${p}${g}_scale`, [`${p}${g}_s`, scale], [`${p}${g}_m`]));
    add.push(makeNode('Softmax', `${p}${g}_softmax`, [`${p}${g}_m`], [`${p}${g}_p`], keepAttrs(sm)));
    add.push(makeNode('Einsum', `${p}${g}_pv`, [`${p}${g}_p`, `${p}v${g}`], [`${p}${g}_o`], keepAttrs(e2)));
  }
  add.push(makeNode('Concat', p + 'concat', sizes.map((_, g) => `${p}${g}_o`), [out], [attr('axis', T_INT, A_I_, 0)]));
  // the four nodes come one after another from `at`: the groups take their place
  const drop = new Set([e1, mul, sm, e2]);
  const rest = nodes.filter((n) => !drop.has(n));
  rest.splice(at, 0, ...add);
  nodes.length = 0;
  nodes.push(...rest);
}

/**
 * BS-RoFormer SW (6 stems: bass, drums, other, vocals, guitar, piano): keep some stems.
 * The stems' masks are joined by one Concat (axis 1); the shapes after it carry the stem count.
 * opts.attentionGroups: run each time attention (62 bands x 8 heads x 345 x 345 scores, 236 MB)
 * in that many slices of the bands; opts.chain: one part per transformer layer (buildChain).
 * Together they take the engine's peak memory from about 3.1 GB to under 0.9 GB.
 */
export function bsRoformerPlan(keepIdx, opts = {}) {
  const k = keepIdx.length;
  const groups = opts.attentionGroups || 1;
  return {
    parts: opts.chain ? (n) => {
      const m = /^\/layers\.(\d+)\./.exec(n.name);
      if (m) return 1 + Number(m[1]);
      if (n.name.startsWith('/mask_estimators') || n.name.startsWith('/final_norm')) return 13;
      return null;
    } : null,
    edit(nodes) {
      if (groups > 1) {
        const bands = 62;
        const sizes = Array.from({ length: groups }, (_, g) => Math.floor((bands * (g + 1)) / groups) - Math.floor((bands * g) / groups));
        for (let L = 0; L < 12; L++) groupAttention(nodes, `/layers.${L}.0/layers.0.0/attend/`, sizes);
      }
      const byName = new Map(nodes.map((n) => [n.name, n]));
      const cat = byName.get('/Concat_25');
      if (!cat || cat.inputs.length !== 6) throw new Error('unexpected separator model layout');
      const ins = keepIdx.map((i) => cat.inputs[i]);
      cat.fields = [...ins.map((s) => ({ f: N_INPUT, wt: LEN, v: enc.encode(s) })), ...cat.fields.filter((x) => x.f !== N_INPUT)];
      cat.inputs = ins;
      for (const nm of ['/Constant_156', '/Constant_158', '/Constant_159']) {
        const n = byName.get(nm);
        const v = n && constInts(n);
        if (!v || v[1] !== 6) throw new Error('unexpected separator model layout (' + nm + ')');
        v[1] = k;
        setConstInts(n, v);
      }
    },
    outputs: { out_spec_real: [1, k, 2, 1025, 345], out_spec_imag: [1, k, 2, 1025, 345] },
  };
}

/**
 * Prune a model as it streams in.
 * @param chunks async iterable of Uint8Array
 * @param plan {edit(nodes), outputs: {name: dims}}
 * @param onProgress(bytesRead)
 * @returns {Promise<Uint8Array>} the pruned model
 */
export async function pruneStream(chunks, plan, onProgress) {
  const r = new Reader(chunks);
  const head = [], tail = [];
  let graphPieces = null;
  let tick = 0;
  const report = () => { if (onProgress && r.pos - tick > (1 << 20)) { tick = r.pos; onProgress(r.pos); } };
  while (!(await r.atEnd())) {
    const key = await r.varint();
    const f = Math.floor(key / 8), wt = key & 7;
    if (f === M_GRAPH && wt === LEN) {
      const glen = await r.varint();
      graphPieces = await pruneGraph(r, r.pos + glen, plan, report);
      continue;
    }
    // other top-level fields are copied as they are
    const pieces = graphPieces ? tail : head;
    pieces.push(varintBytes(key));
    if (wt === VARINT) pieces.push(varintBytes(await r.varint()));
    else if (wt === I64) pieces.push(await r.bytes(8));
    else if (wt === I32) pieces.push(await r.bytes(4));
    else if (wt === LEN) { const l = await r.varint(); pieces.push(varintBytes(l), await r.bytes(l)); }
    else throw new Error('bad wire type ' + wt);
  }
  if (!graphPieces) throw new Error('no graph in the model file');
  if (onProgress) onProgress(r.pos);
  if (plan.parts) return buildChain(head, tail, graphPieces, plan);
  return join(modelPieces(head, tail, graphPieces, plan));
}

function join(pieces) {
  const out = new Uint8Array(pieces.reduce((s, x) => s + x.length, 0));
  let o = 0;
  for (const x of pieces) { out.set(x, o); o += x.length; }
  return out;
}

function modelPieces(head, tail, g, plan, part = null) {
  const pieces = [];
  const nodes = part ? part.nodes : g.nodes;
  const inits = part ? part.inits : g.inits;
  encodeFields(nodes.map((n) => ({ f: G_NODE, wt: LEN, v: n.fields })), pieces);
  for (const x of g.other) pieces.push(x);
  encodeFields(inits.map((t) => ({ f: G_INIT, wt: LEN, v: t })), pieces);
  if (part) encodeFields(part.ins.map((nm) => ({ f: G_INPUT, wt: LEN, v: valueInfo(nm, null) })), pieces);
  else for (const x of g.inputs) pieces.push(x);
  const outs = part ? part.outs : Object.keys(plan.outputs);
  encodeFields(outs.map((nm) => ({ f: G_OUTPUT, wt: LEN, v: valueInfo(nm, plan.outputs[nm] || null) })), pieces);
  const glen = pieces.reduce((t, x) => t + x.length, 0);
  return [...head, varintBytes(M_GRAPH * 8 + LEN), varintBytes(glen), ...pieces, ...tail];
}

/**
 * The model as a chain of parts that run one after another (plan.parts(node) -> part number, or
 * null: the part its inputs come from). Only the tensors passed between parts live across them,
 * so the engine's peak memory is one part's, not the whole model's. Constant subgraphs are copied
 * into every part that uses them.
 * Container: 'BSRC', u32 header length, JSON {parts: [{offset, length, in, out}]}, then the parts.
 */
function buildChain(head, tail, g, plan) {
  const { nodes, inits } = g;
  const tname = (t) => str(t.find((x) => x.f === T_NAME).v);
  const initByName = new Map(inits.map((t) => [tname(t), t]));
  const prod = new Map();
  nodes.forEach((n, i) => { for (const o of n.outputs) prod.set(o, i); });
  const isStatic = new Set(initByName.keys());
  for (const n of nodes) {
    if ((n.inputs.length || n.op === 'Constant') && n.inputs.every((x) => !x || isStatic.has(x))) for (const o of n.outputs) isStatic.add(o);
  }
  const assign = new Map();
  nodes.forEach((n, i) => {
    if (n.outputs.length && n.outputs.every((o) => isStatic.has(o))) return;
    let p = plan.parts(n);
    if (p == null) p = Math.max(0, ...n.inputs.filter((x) => prod.has(x) && assign.has(prod.get(x))).map((x) => assign.get(prod.get(x))));
    assign.set(i, p);
  });
  for (const [i, p] of assign) {
    for (const x of nodes[i].inputs) if (prod.has(x) && assign.has(prod.get(x)) && assign.get(prod.get(x)) > p) throw new Error('model cannot be split there (' + nodes[i].name + ')');
  }
  const K = Math.max(...assign.values()) + 1;
  const consumers = new Map();
  for (const [i, p] of assign) for (const x of nodes[i].inputs) if (x && !isStatic.has(x)) { if (!consumers.has(x)) consumers.set(x, new Set()); consumers.get(x).add(p); }
  const finals = new Set(Object.keys(plan.outputs));
  const blobs = [], header = [];
  let offset = 0;
  for (let k = 0; k < K; k++) {
    const dyn = [...assign].filter(([, p]) => p === k).map(([i]) => i).sort((a, b) => a - b);
    if (!dyn.length) continue;
    // the constants this part needs
    const st = new Set(), ini = new Set();
    const stack = dyn.flatMap((i) => nodes[i].inputs.filter((x) => x && isStatic.has(x)));
    while (stack.length) {
      const t = stack.pop();
      if (initByName.has(t)) { ini.add(t); continue; }
      const j = prod.get(t);
      if (j === undefined || st.has(j)) continue;
      st.add(j);
      for (const x of nodes[j].inputs) if (x && isStatic.has(x)) stack.push(x);
    }
    const produced = new Set(dyn.flatMap((i) => nodes[i].outputs));
    const ins = [...new Set(dyn.flatMap((i) => nodes[i].inputs).filter((x) => x && !isStatic.has(x) && !produced.has(x)))].sort();
    const outs = [...produced].filter((o) => finals.has(o) || [...(consumers.get(o) || [])].some((p) => p !== k)).sort();
    const part = {
      nodes: [...[...st].sort((a, b) => a - b), ...dyn].map((i) => nodes[i]),
      inits: [...ini].sort().map((nm) => initByName.get(nm)),
      ins, outs,
    };
    const bytes = join(modelPieces(head, tail, g, plan, part));
    blobs.push(bytes);
    header.push({ offset, length: bytes.length, in: ins, out: outs });
    offset += bytes.length;
  }
  const h = enc.encode(JSON.stringify({ parts: header }));
  const pre = new Uint8Array(8);
  pre.set(enc.encode('BSRC'));
  new DataView(pre.buffer).setUint32(4, h.length, true);
  return join([pre, h, ...blobs]);
}

async function pruneGraph(r, end, plan, report) {
  const nodes = [];
  const other = []; // graph fields kept as they are (name, ...)
  const inputs = []; // the graph's inputs (encoded fields)
  let need = null;
  const inits = [];
  const outs = Object.keys(plan.outputs);
  const decide = () => {
    plan.edit(nodes);
    need = new Set(outs);
    const keep = [];
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      if (n.outputs.some((o) => need.has(o))) { keep.push(n); for (const x of n.inputs) if (x) need.add(x); }
    }
    keep.reverse();
    nodes.length = 0;
    nodes.push(...keep);
  };
  while (r.pos < end) {
    const key = await r.varint();
    const f = Math.floor(key / 8), wt = key & 7;
    if (wt !== LEN) { // not expected in a graph, keep it
      other.push(varintBytes(key), wt === VARINT ? varintBytes(await r.varint()) : await r.bytes(wt === I64 ? 8 : 4));
      continue;
    }
    const len = await r.varint();
    if (f === G_NODE) {
      if (need) throw new Error('model layout not supported (nodes after initializers)');
      nodes.push(nodeInfo(await r.bytes(len)));
    } else if (f === G_INIT) {
      if (!need) decide();
      const tEnd = r.pos + len;
      const tf = [];
      let keepIt = null;
      while (r.pos < tEnd) {
        const k2 = await r.varint();
        const f2 = Math.floor(k2 / 8), w2 = k2 & 7;
        if (w2 === VARINT) { tf.push({ f: f2, wt: w2, v: await r.varint() }); continue; }
        if (w2 === I64 || w2 === I32) { tf.push({ f: f2, wt: w2, v: await r.bytes(w2 === I64 ? 8 : 4) }); continue; }
        const l2 = await r.varint();
        if (f2 === T_RAW) {
          if (keepIt === null) throw new Error('model layout not supported (tensor data before its name)');
          if (keepIt) tf.push({ f: f2, wt: w2, v: await r.bytes(l2) });
          else await r.skip(l2);
        } else {
          const v = await r.bytes(l2);
          if (f2 === T_NAME) keepIt = need.has(str(v));
          tf.push({ f: f2, wt: w2, v });
        }
        report();
      }
      if (keepIt) inits.push(tf);
    } else if (f === G_OUTPUT || f === G_VALUEINFO) {
      await r.skip(len); // outputs are rewritten; intermediate shapes are optional
    } else if (f === G_INPUT) {
      inputs.push(varintBytes(key), varintBytes(len), await r.bytes(len));
    } else {
      other.push(varintBytes(key), varintBytes(len), await r.bytes(len));
    }
    report();
  }
  if (!need) decide();
  return { nodes, other, inits, inputs };
}
