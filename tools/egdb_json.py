"""EGDB (Chen et al., ICASSP 2022: electric guitar through a hexaphonic pickup, 240 clips, direct
input and five amps) labels -> the annotation JSON the tests read (tools/guitarset_json.py format):
{clip: {notes: [[t, dur, midi, string (0 = low E)]], duration}}. Each label file has one MIDI
track per string, track/channel 0 = string 1 (high E).
usage: python tools/egdb_json.py <EGDB dir with audio_label/ and audio_DI/> <out.json>"""
import json
import os
import sys

import mido
import soundfile as sf

OPEN = [40, 45, 50, 55, 59, 64]  # standard tuning, low E first


def notes_of(path):
    on, notes, t = {}, [], 0.0
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            on[(msg.channel, msg.note)] = t
        elif msg.type in ('note_off', 'note_on') and (msg.channel, msg.note) in on:
            t0 = on.pop((msg.channel, msg.note))
            if t > t0:
                notes.append([round(t0, 5), round(t - t0, 5), msg.note, 5 - msg.channel])
    return sorted(notes)


def main():
    root, out = sys.argv[1], sys.argv[2]
    res, bad = {}, 0
    for f in sorted(os.listdir(os.path.join(root, 'audio_label')), key=lambda s: int(s.split('.')[0])):
        name = f.split('.')[0]
        notes = notes_of(os.path.join(root, 'audio_label', f))
        for n in notes:
            fret = n[2] - OPEN[n[3]]
            if not 0 <= fret <= 24:
                bad += 1
        res[name] = {'notes': notes, 'duration': sf.info(os.path.join(root, 'audio_DI', name + '.wav')).duration}
    json.dump(res, open(out, 'w'))
    print(len(res), 'clips', sum(len(v['notes']) for v in res.values()), 'notes', bad, 'outside frets 0-24 in standard tuning')


if __name__ == '__main__':
    main()
