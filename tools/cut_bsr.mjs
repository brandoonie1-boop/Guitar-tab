// Cut the BS-RoFormer download down to the app's guitar model the way the browser does
// (app/models.js: engine/onnxprune.js, 6 attention slices, a chain of parts) and check it against
// the SHA-256 the app pins, so Node tests (tests/eval_band.mjs) run the very same separator.
// usage: node tools/cut_bsr.mjs <bs_roformer_sw_6stem_fp16.onnx> <out .bin> [guitar|guitarOther]
import fs from 'node:fs';
import crypto from 'node:crypto';
import { pruneStream, bsRoformerPlan } from '../engine/onnxprune.js';
import { BSR_CUTS, CUT_OPTS } from '../app/models.js';

const [src, out, name = 'guitar'] = process.argv.slice(2);
const cut = BSR_CUTS[name];
async function* chunks() {
  const fd = fs.openSync(src, 'r');
  const buf = Buffer.alloc(8 << 20);
  for (;;) {
    const n = fs.readSync(fd, buf, 0, buf.length, null);
    if (!n) break;
    yield new Uint8Array(buf.subarray(0, n)); // a copy is made by the cutter where it keeps bytes
  }
  fs.closeSync(fd);
}
const bytes = await pruneStream(chunks(), bsRoformerPlan(cut.keep, CUT_OPTS));
const sha = crypto.createHash('sha256').update(bytes).digest('hex');
if (sha !== cut.sha256) throw new Error(`cut ${sha} is not the app's ${cut.sha256}`);
fs.writeFileSync(out, bytes);
console.log(`${out}: ${bytes.length} bytes, SHA-256 matches the app's ${name} cut`);
