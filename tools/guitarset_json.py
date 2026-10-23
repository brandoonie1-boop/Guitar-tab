"""GuitarSet (https://zenodo.org/records/3371780, annotation.zip) or EGSet12
(https://zenodo.org/records/11406378) JAMS files -> the one JSON file the tests read:
{take: {notes: [[t, dur, midi, string (0 = low E)]], beats: [[t, position in bar]], tempo, key, chords, duration}}

usage: python tools/guitarset_json.py <folder with .jams files> <out.json>
"""
import json, glob, os, sys
src, out = sys.argv[1], sys.argv[2]
res = {}
for p in sorted(glob.glob(os.path.join(src, '*.jams'))):
    name = os.path.basename(p)[:-5]
    d = json.load(open(p))
    notes, beats, tempo, key, chords = [], [], None, None, []
    for a in d['annotations']:
        ns = a['namespace']
        if ns == 'note_midi':
            s = int(a['annotation_metadata']['data_source'])
            for n in a['data']:
                notes.append([n['time'], n['duration'], n['value'], s])
        elif ns == 'beat_position':
            beats = [[b['time'], b['value']['position']] for b in a['data']]
        elif ns == 'tempo' and a['data']:
            tempo = a['data'][0]['value']
        elif ns == 'key_mode' and a['data']:
            key = a['data'][0]['value']
        elif ns == 'chord' and not chords:
            chords = [[c['time'], c['duration'], c['value']] for c in a['data']]
    notes.sort()
    res[name] = {'notes': notes, 'beats': beats, 'tempo': tempo, 'key': key, 'chords': chords, 'duration': d['file_metadata']['duration']}
json.dump(res, open(out, 'w'))
n = sum(len(v['notes']) for v in res.values())
print('tracks', len(res), 'notes', n)
