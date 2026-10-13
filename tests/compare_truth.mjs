// Compare a transcription (notes.json from run_file --out) with a truth JSON.
import fs from 'node:fs';

const [outJson, truthJson] = process.argv.slice(2);
const parts = JSON.parse(fs.readFileSync(outJson, 'utf8'));
const truth = JSON.parse(fs.readFileSync(truthJson, 'utf8'));
const est = parts.flatMap((p) => p.notes.filter((n) => !n.dropped));
const ref = truth.notes;
const tol = 0.05;
const used = new Set();
let tp = 0, sameString = 0;
const misses = [];
for (const r of ref) {
  let best = -1, bd = 1e9;
  est.forEach((e, i) => {
    if (used.has(i) || e.pitch !== r.pitch) return;
    const d = Math.abs(e.t0 - r.t0);
    if (d <= tol && d < bd) { bd = d; best = i; }
  });
  if (best >= 0) {
    used.add(best); tp++;
    const e = est[best];
    if (e.string === r.string && e.fret === r.fret) sameString++;
  } else misses.push(r);
}
const fps = est.filter((_, i) => !used.has(i));
const P = tp / Math.max(1, est.length), R = tp / Math.max(1, ref.length);
console.log(`notes: ref ${ref.length}, est ${est.length}, matched ${tp}; precision ${(100 * P).toFixed(1)}% recall ${(100 * R).toFixed(1)}% F1 ${(200 * P * R / Math.max(1e-9, P + R)).toFixed(1)}%`);
console.log(`string+fret correct for matched notes: ${(100 * sameString / Math.max(1, tp)).toFixed(1)}%`);
console.log('first misses:', misses.slice(0, 8).map((m) => `${m.t0.toFixed(2)}:${m.pitch}`).join(' '));
console.log('first false positives:', fps.slice(0, 12).map((m) => `${m.t0.toFixed(2)}:${m.pitch}(${m.amp.toFixed(2)})`).join(' '));
