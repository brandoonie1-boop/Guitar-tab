"""Synthetic guitar parts with exact note labels, played with recorded single notes (Guitar-TECHS,
cut by tools/gtechs_notes.py), for fine-tuning the guitar note model (tools/train_hrgt.py) on the
kind of playing distortion is used for: power-chord riffs (open or palm muted, gallops), single-note
lines, strummed and arpeggiated chords, octaves and double stops, in standard, drop and lowered
tunings. A string plays one note at a time, as on a guitar; chords are strummed a few ms apart;
palm mutes are short and dark; timing and loudness vary a little from note to note.

The audio is the clean signal (direct input, sometimes the amp microphone); distortion and the
rest of the amp are added while training. Writes one 16 kHz int16 .npy per part and a
manifest.json in the format of tools/hrgt_data.py.

usage: python tools/synth_riffs.py <notes dir> <out dir> [--minutes 150] [--seed 1]
"""
import argparse
import json
import os

import numpy as np
from scipy.signal import resample_poly, lfilter

SR = 48000
STD = [40, 45, 50, 55, 59, 64]
# where the note model puts the onset: 9 ms after the start of the recordings' pick transient
# (measured with the original model on clean renders)
LABEL_DELAY = 0.009
TUNINGS = [  # (weight, open strings low to high)
    (40, STD),
    (14, [38, 45, 50, 55, 59, 64]),  # drop D
    (10, [39, 44, 49, 54, 58, 63]),  # Eb standard
    (6, [38, 43, 48, 53, 57, 62]),  # D standard
    (6, [36, 43, 48, 53, 57, 62]),  # drop C
    (4, [37, 44, 49, 54, 58, 63]),  # drop C#
    (4, [35, 42, 47, 52, 56, 61]),  # B standard
    (3, [38, 45, 50, 54, 57, 62]),  # open D
    (3, [38, 43, 50, 55, 59, 62]),  # open G
    (3, [38, 45, 50, 55, 57, 62]),  # DADGAD
    (3, [38, 45, 52, 57, 59, 64]),  # DAEABE (open strings ring a lot)
]
SCALES = {'minor_pent': [0, 3, 5, 7, 10], 'minor': [0, 2, 3, 5, 7, 8, 10], 'blues': [0, 3, 5, 6, 7, 10],
          'major': [0, 2, 4, 5, 7, 9, 11], 'major_pent': [0, 2, 4, 7, 9], 'dorian': [0, 2, 3, 5, 7, 9, 10],
          'phrygian': [0, 1, 3, 5, 7, 8, 10], 'harm_minor': [0, 2, 3, 5, 7, 8, 11]}
# chord shapes as frets per string (None = not played), relative to a barre fret where marked
OPEN_CHORDS = {
    'E': [0, 2, 2, 1, 0, 0], 'Em': [0, 2, 2, 0, 0, 0], 'A': [None, 0, 2, 2, 2, 0], 'Am': [None, 0, 2, 2, 1, 0],
    'D': [None, None, 0, 2, 3, 2], 'Dm': [None, None, 0, 2, 3, 1], 'G': [3, 2, 0, 0, 0, 3], 'C': [None, 3, 2, 0, 1, 0],
    'E7': [0, 2, 0, 1, 0, 0], 'A7': [None, 0, 2, 0, 2, 0], 'D7': [None, None, 0, 2, 1, 2], 'Cadd9': [None, 3, 2, 0, 3, 0],
    'Asus2': [None, 0, 2, 2, 0, 0], 'Dsus4': [None, None, 0, 2, 3, 3], 'Fmaj7': [None, None, 3, 2, 1, 0],
}
BARRE_SHAPES = {  # offsets from the barre fret
    'E': [0, 2, 2, 1, 0, 0], 'Em': [0, 2, 2, 0, 0, 0], 'E7': [0, 2, 0, 1, 0, 0], 'Em7': [0, 2, 0, 0, 0, 0],
    'A': [None, 0, 2, 2, 2, 0], 'Am': [None, 0, 2, 2, 1, 0], 'A7': [None, 0, 2, 0, 2, 0], 'Am7': [None, 0, 2, 0, 1, 0],
    'Amaj7': [None, 0, 2, 1, 2, 0],
}


class Bank:
    """Recorded notes by (player, signal, string, fret), with the pitch error of each."""

    def __init__(self, d):
        self.d = d
        self.idx = {}
        for r in json.load(open(os.path.join(d, 'index.json'))):
            if abs(r['cents']) <= 40 and r['len'] >= 0.6:
                self.idx[(r['player'], r['signal'], r['string'], r['fret'])] = r
        self.cache = {}

    def get(self, player, signal, s, pitch):
        """Audio of `pitch` on string s (48 kHz): the recorded fret when there is one, otherwise the
        nearest recorded fret on that string, re-pitched."""
        f = pitch - STD[s]
        cands = [k for k in range(25) if (player, signal, s, k) in self.idx]
        if not cands:
            return None
        k = min(cands, key=lambda c: abs(c - f))
        r = self.idx[(player, signal, s, k)]
        key = (r['name'], pitch)
        if key not in self.cache:
            y = np.fromfile(os.path.join(self.d, r['name'] + '.f32'), np.float32)
            shift = (pitch - (STD[s] + k)) - r['cents'] / 100  # semitones to move it by
            if abs(shift) > 0.03:
                # resample: up a semitone = 2^(1/12) times faster
                ratio = 2 ** (shift / 12)
                n = int(len(y) / ratio)
                y = np.interp(np.arange(n) * ratio, np.arange(len(y)), y).astype(np.float32)
            if len(self.cache) > 3000:
                self.cache.clear()
            self.cache[key] = y / (np.sqrt(np.mean(y[: int(0.3 * SR)] ** 2)) + 1e-6)
        return self.cache[key]


class Part:
    """Notes on strings: (t0, dur, string, fret, velocity 0-1, palm mute, let ring)."""

    def __init__(self, tuning):
        self.tuning = tuning
        self.notes = []

    def add(self, t0, dur, s, fret, vel, pm=False, ring=False):
        if fret is not None and 0 <= fret <= 22 and 0 <= s <= 5:
            self.notes.append([t0, dur, s, fret, vel, pm, ring])


def chord_frets(shape, base=0):
    return [None if f is None else f + base for f in shape]


def power_chord(tun, rs, fret, octave):
    """Root on string rs at `fret`, the fifth and (optionally) the octave on the next strings, by
    pitch, so drop tunings get their one-finger shape."""
    root = tun[rs] + fret
    frets = [None] * 6
    frets[rs] = fret
    frets[rs + 1] = root + 7 - tun[rs + 1]
    if octave:
        frets[rs + 2] = root + 12 - tun[rs + 2]
    return frets


def strum(part, r, t, frets, vel, down=True, spread=None, pm=False, dur=1.0, strings=None):
    order = [s for s in range(6) if frets[s] is not None and (strings is None or s in strings)]
    if not down:
        order = order[::-1]
    sp = spread if spread is not None else r.uniform(0.006, 0.03)
    for i, s in enumerate(order):
        part.add(t + i * sp * r.uniform(0.7, 1.3), dur - i * sp, s, frets[s], vel * r.uniform(0.85, 1.05) * (1 if down else 0.8), pm=pm)


def section_chug(part, r, t, bars, beat, tun):
    """Power-chord riff on the low strings: palm-muted 8ths/16ths with open accents."""
    key = r.integers(0, 8)
    degrees = [0, 3, 5, 7, 8, 10, 12, 1, 6]
    root_string = 0 if r.random() < 0.7 else 1
    grid = 4 if r.random() < 0.55 else 2  # 16ths or 8ths
    step = beat / grid
    n = bars * 4 * grid
    pattern = r.random(n) < r.uniform(0.55, 0.95)  # which steps are played
    chords = [int(key + r.choice(degrees)) for _ in range(max(1, bars * r.integers(1, 3)))]
    gallop = r.random() < 0.25
    octave = r.random() < 0.4
    for i in range(n):
        if not pattern[i]:
            continue
        if gallop and i % 4 == 3:
            continue
        c = chords[min(len(chords) - 1, i * len(chords) // n)]
        accent = (i % (grid * 2) == 0 and r.random() < 0.5) or r.random() < 0.1
        open_pedal = (not accent) and r.random() < 0.35
        t0 = t + i * step + r.normal(0, 0.004)
        if open_pedal:
            part.add(t0, step * 0.9, root_string, 0, r.uniform(0.6, 0.9), pm=True)
            continue
        dur = step * (r.integers(2, 5) if accent else 0.9)
        pm = not accent and r.random() < 0.85
        v = r.uniform(0.75, 1.0) if accent else r.uniform(0.55, 0.85)
        frets = power_chord(tun, root_string, c, octave or accent)
        strum(part, r, t0, frets, v, down=True, spread=r.uniform(0.003, 0.012), pm=pm, dur=dur)


def section_power_sustain(part, r, t, bars, beat, tun):
    key = r.integers(0, 10)
    t0 = t
    end = t + bars * 4 * beat
    while t0 < end - 0.05:
        dur = beat * r.choice([1, 2, 2, 3, 4])
        rs = 0 if r.random() < 0.5 else 1
        c = int(key + r.choice([0, 2, 3, 5, 7, 8, 10]))
        frets = power_chord(tun, rs, c, r.random() < 0.7)
        strum(part, r, t0, frets, r.uniform(0.7, 1.0), dur=min(dur, end - t0))
        t0 += dur


def scale_positions(tun, key, scale, lo, hi):
    """(string, fret) for scale notes between frets lo..hi."""
    out = []
    for s in range(6):
        for f in range(max(0, lo), hi + 1):
            if (tun[s] + f - key) % 12 in scale:
                out.append((s, f))
    return out


def section_lead(part, r, t, bars, beat, tun):
    scale = SCALES[r.choice(list(SCALES))]
    key = 40 + r.integers(0, 12)
    pos = int(r.integers(0, 15))
    cand = scale_positions(tun, key, scale, pos, pos + 4)
    cand.sort(key=lambda sf: tun[sf[0]] + sf[1])
    if len(cand) < 4:
        return
    i = int(r.integers(0, len(cand)))
    t0 = t
    end = t + bars * 4 * beat
    rhythm = r.choice(['8', '16', 't', 'mix'])
    while t0 < end - 0.05:
        if rhythm == 'mix':
            d = beat * r.choice([0.25, 0.25, 0.5, 0.5, 1, 1.5, 2])
        else:
            d = beat * {'8': 0.5, '16': 0.25, 't': 1 / 3}[rhythm] * (r.choice([1, 1, 1, 2]))
        if r.random() < 0.08:  # rest
            t0 += d
            continue
        i = int(np.clip(i + r.choice([-2, -1, -1, 1, 1, 2, 0, 3, -3]), 0, len(cand) - 1))
        s, f = cand[i]
        part.add(t0 + r.normal(0, 0.005), d * r.uniform(0.7, 1.0), s, f, r.uniform(0.5, 1.0))
        t0 += d


def section_strum(part, r, t, bars, beat, tun):
    if r.random() < 0.5:
        names = list(OPEN_CHORDS)
        prog = [OPEN_CHORDS[names[r.integers(0, len(names))]] for _ in range(r.integers(2, 5))]
    else:
        prog = [chord_frets(BARRE_SHAPES[list(BARRE_SHAPES)[r.integers(0, len(BARRE_SHAPES))]], int(r.integers(1, 10))) for _ in range(r.integers(2, 5))]
    pattern = r.choice(['D', 'DU', 'D_DU_UDU', 'DDU', 'DUDU'])
    sub = 2 if pattern in ('DU', 'D_DU_UDU', 'DDU') else (4 if pattern == 'DUDU' else 1)
    step = beat / sub
    n = bars * 4 * sub
    for i in range(n):
        ch = prog[min(len(prog) - 1, i * len(prog) // n)]
        sym = pattern[i % len(pattern)]
        if sym == '_' or r.random() < 0.06:
            continue
        down = sym == 'D' if pattern != 'D' else True
        strings = None if down else ({2, 3, 4, 5} if r.random() < 0.6 else None)
        strum(part, r, t + i * step + r.normal(0, 0.006), ch, r.uniform(0.5, 1.0), down=down, dur=step * r.uniform(0.9, 2.5),
              strings=strings, pm=r.random() < 0.05)


def section_arpeggio(part, r, t, bars, beat, tun):
    names = list(OPEN_CHORDS)
    prog = [OPEN_CHORDS[names[r.integers(0, len(names))]] if r.random() < 0.6 else chord_frets(BARRE_SHAPES[list(BARRE_SHAPES)[r.integers(0, len(BARRE_SHAPES))]], int(r.integers(1, 9))) for _ in range(r.integers(2, 5))]
    sub = int(r.choice([2, 2, 3, 4]))
    step = beat / sub
    n = bars * 4 * sub
    for i in range(n):
        ch = prog[min(len(prog) - 1, i * len(prog) // n)]
        strs = [s for s in range(6) if ch[s] is not None]
        order = strs + strs[-2:0:-1]
        s = order[i % len(order)]
        # let ring: until the chord changes
        nxt = (min(len(prog) - 1, i * len(prog) // n) + 1) * n / len(prog)
        part.add(t + i * step + r.normal(0, 0.006), max(step, (nxt - i) * step), s, ch[s], r.uniform(0.45, 0.9), ring=True)


def section_double(part, r, t, bars, beat, tun):
    kind = r.choice(['octave', 'third', 'fourth', 'sixth'])
    key = r.integers(0, 12)
    sub = int(r.choice([1, 2, 2, 4]))
    step = beat / sub
    n = bars * 4 * sub
    base = int(r.integers(0, 10))
    for i in range(n):
        if r.random() < 0.15:
            continue
        f = int(np.clip(base + r.choice([-3, -2, 0, 2, 3, 5]), 0, 17))
        if kind == 'octave':
            s = int(r.choice([0, 1, 2]))
            pair = [(s, f), (s + 2, f + 2)]
        else:
            s = int(r.integers(1, 5))
            off = {'third': -1, 'fourth': 0, 'sixth': -2}[kind]
            pair = [(s, f), (s + 1, max(0, f + off))] if kind != 'sixth' else [(s - 1, f), (s + 1, max(0, f + off))]
        t0 = t + i * step + r.normal(0, 0.005)
        for j, (ss, ff) in enumerate(pair):
            part.add(t0 + j * r.uniform(0, 0.01), step * r.uniform(0.6, 1.0), ss, ff, r.uniform(0.6, 0.95), pm=r.random() < 0.1)
        if r.random() < 0.2:
            base = int(np.clip(base + r.choice([-2, 2, 5, -5]), 0, 15))


STYLES = [(section_chug, 26), (section_power_sustain, 12), (section_lead, 22), (section_strum, 14), (section_arpeggio, 12), (section_double, 14)]


def render(part, bank, r, player, signal):
    """Mix the part's notes (48 kHz) and return (audio, labelled notes)."""
    notes = sorted(part.notes, key=lambda n: n[0])
    if not notes:
        return None, []
    total = max(n[0] + n[1] for n in notes) + 2.0
    y = np.zeros(int(total * SR) + SR, np.float32)
    detune = r.normal(0, 8)  # cents, the whole part
    # a string plays one note at a time: each note ends where the next on its string starts
    by_string = {s: [n for n in notes if n[2] == s] for s in range(6)}
    labels = []
    for s, ns in by_string.items():
        for k, (t0, dur, _, fret, vel, pm, ring) in enumerate(ns):
            t0 = max(0.02, t0)
            nxt = ns[k + 1][0] if k + 1 < len(ns) else 1e9
            end = min(t0 + dur, nxt) if not ring else min(nxt, t0 + max(dur, 0.2))
            if end - t0 < 0.04:
                continue
            pitch = part.tuning[s] + fret
            x = bank.get(player, signal, s, pitch)
            if x is None:
                continue
            end = min(end, t0 + len(x) / SR - 0.02)  # the recording's own length
            if abs(detune) > 1:
                ratio = 2 ** (detune / 1200)
                x = np.interp(np.arange(int(len(x) / ratio)) * ratio, np.arange(len(x)), x).astype(np.float32)
            start = int((t0 - 0.010) * SR)  # the recordings start 10 ms before the pick attack
            n = min(len(x), int((end - t0 + 0.01) * SR) + int(0.03 * SR), len(y) - start)
            if n <= int(0.03 * SR):
                continue
            seg = x[:n].copy()
            g = (0.25 + 0.75 * vel) ** 1.6
            if pm:
                tau = r.uniform(0.05, 0.14)
                seg *= np.exp(-np.arange(n) / (tau * SR)).astype(np.float32)
                a = np.exp(-2 * np.pi * r.uniform(700, 1800) / SR)
                seg = lfilter([1 - a], [1, -a], seg).astype(np.float32) * 1.6
                end = min(end, t0 + 3 * tau)
            # release: the string is damped over 10-40 ms at its end
            rel = int(r.uniform(0.01, 0.04) * SR)
            stop = int((end - t0 + 0.01) * SR)
            if stop < n:
                fade_end = min(n, stop + rel)
                seg[stop:fade_end] *= np.linspace(1, 0, fade_end - stop, dtype=np.float32)
                seg[fade_end:] = 0
            y[start:start + n] += g * seg
            labels.append([round(t0 + LABEL_DELAY, 4), round(end + LABEL_DELAY, 4), int(pitch), int(round(127 * vel))])
    labels.sort()
    pk = np.abs(y).max() + 1e-9
    y16 = resample_poly(y / pk, 1, 3).astype(np.float32)
    return y16, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('notes')
    ap.add_argument('out')
    ap.add_argument('--minutes', type=float, default=150)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bank = Bank(a.notes)
    r = np.random.default_rng(a.seed)
    tw = np.array([w for w, _ in TUNINGS], float)
    sw = np.array([w for _, w in STYLES], float)
    man, total, i = [], 0.0, 0
    while total < a.minutes * 60:
        tun = TUNINGS[r.choice(len(TUNINGS), p=tw / tw.sum())][1]
        part = Part(tun)
        player = 'P1' if r.random() < 0.5 else 'P2'
        signal = 'directinput' if r.random() < 0.8 else 'micamp'
        t = 0.3
        for _ in range(r.integers(2, 5)):
            style = STYLES[r.choice(len(STYLES), p=sw / sw.sum())][0]
            bpm = r.uniform(80, 200) if style in (section_chug, section_double) else r.uniform(65, 170)
            beat = 60 / bpm
            bars = int(r.integers(2, 7))
            style(part, r, t, bars, beat, tun)
            t += bars * 4 * beat + r.choice([0, 0, beat, 2 * beat])
        y, labels = render(part, bank, r, player, signal)
        if y is None or len(labels) < 10:
            continue
        name = f'synth_{a.seed}_{i:04d}'
        np.save(os.path.join(a.out, name + '.npy'), np.round(y * 32000).astype(np.int16))
        man.append({'name': name, 'source': 'synth', 'group': player, 'set': 'synth', 'take': name, 'kind': 'di' if signal == 'directinput' else 'mic',
                    'audio': name + '.npy', 'dur': len(y) / 16000, 'tuning': tun, 'notes': labels})
        total += len(y) / 16000
        i += 1
        if i % 20 == 0:
            print(f'{i} parts, {total / 60:.1f} min', flush=True)
    json.dump(man, open(os.path.join(a.out, 'manifest.json'), 'w'))
    print('parts', len(man), 'minutes', round(total / 60, 1), 'notes', sum(len(m['notes']) for m in man))


if __name__ == '__main__':
    main()
