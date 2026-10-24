"""Cut Guitar-TECHS 'all single notes' recordings into one sample per string and fret.

usage: python tools/gtechs_notes.py <Guitar-TECHS dir> <out dir>
Writes <out>/<player>_<signal>_s<string 0=low E>_f<fret>.f32 (mono float32, 48 kHz, starting
10 ms before the pick attack) and <out>/index.json with the measured pitch of each note.
"""
import json, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from midi import read_midi

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SR = 48000
OPEN = [40, 45, 50, 55, 59, 64]
NAMES = {'E': 0, 'A': 1, 'D': 2, 'G': 3, 'B': 4, 'e': 5}


def load(p):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def f0_yin(x, fmin=70, fmax=1500):
    """YIN pitch of a steady stretch (Hz)."""
    n = len(x)
    W = n // 2
    taus = np.arange(int(SR / fmax), int(SR / fmin))
    d = np.array([np.sum((x[:W] - x[t: t + W]) ** 2) for t in taus])
    cum = np.cumsum(d) / np.arange(1, len(d) + 1)
    dn = d / np.maximum(cum, 1e-12)
    cand = np.where(dn < 0.15)[0]
    i = cand[0] if len(cand) else int(np.argmin(dn))
    while i + 1 < len(dn) and dn[i + 1] < dn[i]:
        i += 1
    # parabolic refinement
    if 0 < i < len(dn) - 1:
        a, b, c = dn[i - 1], dn[i], dn[i + 1]
        off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0
    else:
        off = 0
    return SR / (taus[0] + i + off)


index = []
for player in ['P1', 'P2']:
    base = os.path.join(SRC, f'{player}_singlenotes')
    tracks = read_midi(os.path.join(base, 'midi', 'midi_allsinglenotes.mid'))
    notes = []
    for t in tracks:
        s = NAMES.get(t['name'].strip())
        if s is None:
            continue
        for (a, b, k, v) in t['notes']:
            fret = k - OPEN[s]
            if b - a < 1.0 or not 0 <= fret <= 24:
                continue  # stray converter notes
            notes.append((a, b, k, s, fret))
    notes.sort()
    for sig in ['directinput', 'micamp']:
        x = load(os.path.join(base, 'audio', sig, f'{sig}_allsinglenotes.wav'))
        for i, (a, b, k, s, fret) in enumerate(notes):
            nxt = notes[i + 1][0] if i + 1 < len(notes) else a + 4.0
            # search the attack between 150 ms before and 300 ms after the MIDI onset
            lo, hi = max(0, int((a - 0.15) * SR)), int((a + 0.3) * SR)
            seg = x[lo:hi]
            env = np.convolve(np.abs(seg), np.ones(48) / 48, mode='same')
            pk = env.max()
            att = lo + int(np.argmax(env > 0.2 * pk))
            # walk back to where the attack leaves the floor
            floor = np.median(np.abs(x[max(0, att - 4800): att])) + 1e-6
            j = att
            while j > lo and env[j - lo] > 3 * floor:
                j -= 1
            start = max(0, j - int(0.01 * SR))
            end = min(len(x), int(min(nxt, a + 4.2) * SR) - int(0.02 * SR))
            y = x[start:end].astype(np.float32)
            f0 = f0_yin(y[int(0.25 * SR): int(0.25 * SR) + 4096].astype(np.float64)) if len(y) > SR // 2 else 0
            want = 440 * 2 ** ((k - 69) / 12)
            cents = 1200 * np.log2(f0 / want) if f0 > 0 else 9999
            name = f'{player}_{sig}_s{s}_f{fret}'
            y.tofile(os.path.join(OUT, name + '.f32'))
            index.append({'name': name, 'player': player, 'signal': sig, 'string': s, 'fret': fret, 'midi': k,
                          'len': round(len(y) / SR, 3), 'peak': float(np.abs(y).max()),
                          'rms': float(np.sqrt((y[: int(0.3 * SR)] ** 2).mean())), 'cents': round(float(cents), 1)})
json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=0)
bad = [r for r in index if abs(r['cents']) > 60]
print('notes', len(index), 'pitch off by >60 cents:', len(bad))
for r in bad[:40]:
    print('  ', r['name'], r['midi'], r['cents'])
