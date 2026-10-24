// Tab player events (app/guitarsynth.js scoreEvents) from an alphaTab score: timing from the bars'
// sync points, strings and pitches, hammer-ons, slides, bends, chords and ties.
// usage: node tests/test_guitarsynth.mjs
import * as alphaTab from '../vendor/alphatab/alphaTab.mjs';
import { scoreEvents } from '../app/guitarsynth.js';

const tex = `\\title "t"
\\tempo 120
\\track "Guitar"
\\staff {tabs}
\\tuning (E4 B3 G3 D3 A2 E2)
\\capo 2
\\ts (4 4) 5.3{h}.4 7.3.4 (0.6 2.5 2.4).4 5.2{sl}.8 7.2.8 |
3.1{b (0 4)}.2 -.1.4 r.4 |`;
const score = alphaTab.importer.ScoreLoader.loadAlphaTex(tex, new alphaTab.Settings());
// bar 1 at 1 s, bar 2 at 3 s on the recording
const sync = [{ barIndex: 0, barOccurence: 0, barPosition: 0, millisecondOffset: 1000 }, { barIndex: 1, barOccurence: 0, barPosition: 0, millisecondOffset: 3000 }];
const ev = scoreEvents(score, 0, sync, () => 0.5);

let failed = 0;
const check = (name, ok, detail = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name} ${detail}`); if (!ok) failed++; };
const near = (a, b) => Math.abs(a - b) < 1e-6;
check('one event per played note (the tie extends its note)', ev.length === 8, String(ev.length));
const [h0, h1, c0, c1, c2, s0, s1, b] = ev;
check('times from the sync points', near(h0.t, 1) && near(h1.t, 1.5) && near(c0.t, 2) && near(s0.t, 2.5) && near(s1.t, 2.75) && near(b.t, 3));
check('string and pitch (capo 2 included)', h0.s === 3 && h0.pitch === 55 + 2 + 5 && c0.s === 0 && c0.pitch === 40 + 2, `${h0.s} ${h0.pitch} ${c0.pitch}`);
check('hammer-on: the second note is not picked', !h0.legato && h1.legato);
check('chord: one index per string', [c0, c1, c2].map((e) => e.chordIndex).join() === '0,1,2' && h0.chordIndex === -1);
check('slide: glides into the next note, which is not picked', s0.slideTo === s1 && s1.slidDest === true);
check('bend: a full step', JSON.stringify(b.bend) === '[[0,0],[1,200]]', JSON.stringify(b.bend));
check('tie: the bent note rings through the tied beat', near(b.end, 4.5), String(b.end));
check('loudness from the transcription', ev.every((e) => e.amp === 0.5));

// without sync points the score's tempo sets the time
const ev2 = scoreEvents(score, 0, [], null);
check('tempo when there are no sync points', near(ev2[1].t, 0.5) && near(ev2[7].t, 2), `${ev2[1].t} ${ev2[7].t}`);

console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
