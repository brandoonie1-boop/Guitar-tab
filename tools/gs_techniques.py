"""Playing techniques in GuitarSet, read from its per-string pitch contours (measured on the
hexaphonic pickup, one string at a time): bends, vibrato, and how each note follows the previous
one on its string (a slide: the pitch glides across; legato or picked: it steps). Silver labels
for testing the app's technique marks on real playing.

Writes {take: [[t0, dur, midi, string, marks]]} where marks is a dict, e.g. {"bend": 2} (semitones,
rounded to halves), {"vibrato": 1}, {"slideFrom": 1} (glided in from the previous note on the
string), {"stepFrom": 1} (followed the previous note on the string with no gap and no glide:
a hammer-on, pull-off or a quick re-pick; the contour alone can't tell which).
usage: python tools/gs_techniques.py <GuitarSet annotation dir (.jams)> <out.json>
"""
import json
import os
import sys

import numpy as np


def contour(ann):
    d = ann['data']
    t = np.array(d['time'], float)
    f = np.array([v['frequency'] if v.get('voiced', True) else 0 for v in d['value']], float)
    return t, f


def marks_for(note, t, f, prev, prev_end):
    t0, dur, midi = note
    out = {}
    sel = (t >= t0 + 0.02) & (t <= t0 + dur) & (f > 0)
    if sel.sum() >= 6:
        c = 1200 * np.log2(f[sel] / (440 * 2 ** ((round(midi) - 69) / 12)))
        tt = t[sel]
        base = np.median(c[:max(2, int(0.06 / 0.0058))])
        hi = np.percentile(c, 90)
        held = (c >= base + 0.7 * (hi - base)).sum() * 0.0058
        if hi - base >= 80 and held >= 0.05:
            out['bend'] = round(2 * (hi - base) / 100) / 2  # semitones, halves
        else:
            k = max(1, int(0.125 / 0.0058))
            sm = np.convolve(c, np.ones(2 * k + 1) / (2 * k + 1), mode='same')
            det = (c - sm)[k:-k] if len(c) > 2 * k + 2 else c - c.mean()
            if len(det) > 10:
                zc = np.sum(np.diff(np.sign(det)) != 0)
                rate = zc / (2 * len(det) * 0.0058)
                depth = 2 * np.sqrt(2) * det.std()
                if 3.5 <= rate <= 9 and depth >= 25 and dur >= 0.3:
                    out['vibrato'] = 1
    if prev is not None and t0 - prev_end < 0.03 and round(midi) != round(prev[2]):
        # the transition from the previous note on this string: a glide or a step
        sel = (t >= prev_end - 0.08) & (t <= t0 + 0.05) & (f > 0)
        if sel.sum() >= 4:
            m = 69 + 12 * np.log2(f[sel] / 440)
            lo, hi = sorted((round(prev[2]), round(midi)))
            between = ((m > lo + 0.5) & (m < hi - 0.5)).sum()
            if hi - lo >= 2 and between >= 3:
                out['slideFrom'] = 1
            else:
                out['stepFrom'] = 1
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    res = {}
    counts = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith('.jams'):
            continue
        j = json.load(open(os.path.join(src, fn)))
        conts, notes = {}, {}
        for a in j['annotations']:
            s = a['annotation_metadata'].get('data_source')
            if a['namespace'] == 'pitch_contour':
                conts[int(s)] = contour(a)
            elif a['namespace'] == 'note_midi':
                d = a['data']
                rows = d if isinstance(d, list) else [dict(zip(d, v)) for v in zip(*d.values())]
                notes[int(s)] = sorted((r['time'], r['duration'], r['value']) for r in rows)
        take = []
        for s, ns in notes.items():
            t, f = conts.get(s, (np.zeros(0), np.zeros(0)))
            prev, prev_end = None, -1
            for n in ns:
                mk = marks_for(n, t, f, prev, prev_end)
                for k in mk:
                    counts[k] = counts.get(k, 0) + 1
                take.append([round(n[0], 4), round(n[1], 4), round(n[2], 3), s, mk])
                prev, prev_end = n, n[0] + n[1]
        res[fn[:-5]] = sorted(take)
    json.dump(res, open(out, 'w'))
    print(len(res), 'takes', sum(len(v) for v in res.values()), 'notes', counts)


if __name__ == '__main__':
    main()
