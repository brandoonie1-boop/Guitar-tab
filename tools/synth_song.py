"""Render a synthetic 'song' from a tab: Karplus-Strong guitar(s) + drums + bass. Writes WAV + truth JSON.
usage: python synth_song.py out_prefix case_name
"""
import sys, json, numpy as np, soundfile as sf

sr = 44100
rng = np.random.default_rng(7)

def ks(freq, dur, bright=0.5, decay=0.9965):
    n = int(dur * sr)
    p = max(2, int(round(sr / freq)))
    buf = rng.uniform(-1, 1, p)
    # pick position filter (brightness)
    buf = np.convolve(buf, [bright, 1 - bright], mode='same')
    out = np.zeros(n)
    idx = 0
    for i in range(n):
        out[i] = buf[idx]
        nxt = (idx + 1) % p
        buf[idx] = decay * 0.5 * (buf[idx] + buf[nxt])
        idx = nxt
    # body-ish resonance: simple one-pole lowpass + slight highpass
    env = np.ones(n)
    rel = int(0.02 * sr)
    if n > rel:
        env[-rel:] = np.linspace(1, 0, rel)
    return out * env

CASES = {
    # (tuning low->high, capo, bpm, events [(beat, dur_beats, [(string 6..1, fret)])])
    'daeabe7': ([38, 45, 52, 57, 59, 64], 7, 120, None),
    'std_chords': ([40, 45, 50, 55, 59, 64], 0, 100, None),
}

def daeabe_events():
    ev = []
    for bar, low in enumerate([7, 5, 3, 5, 7, 5, 3, 5]):
        b0 = bar * 4
        ev += [(b0, 4, [(6, low)]), (b0 + 0.5, 3.4, [(4, 0)]), (b0 + 1, 3.0, [(3, 0)]), (b0 + 1.5, 2.4, [(2, 0)]),
               (b0 + 2, 2.0, [(1, 0)]), (b0 + 2.5, 1.4, [(2, 0)]), (b0 + 3, 1.0, [(3, 0)]), (b0 + 3.5, 0.5, [(5, 0)])]
    return ev

def chords_events():
    C = [(5, 3), (4, 2), (3, 0), (2, 1), (1, 0)]
    G = [(6, 3), (5, 2), (4, 0), (3, 0), (2, 0), (1, 3)]
    Am = [(5, 0), (4, 2), (3, 2), (2, 1), (1, 0)]
    F = [(6, 1), (5, 3), (4, 3), (3, 2), (2, 1), (1, 1)]
    ev = []
    for bar, ch in enumerate([C, G, Am, F, C, G, Am, F]):
        for k in [0, 1, 1.5, 2.5, 3]:
            ev.append((bar * 4 + k, 0.9 if k in (0, 1.5, 3) else 0.45, ch))
    return ev

def render(case, out_prefix, pan=-0.3, drums=True):
    tuning, capo, bpm, _ = CASES[case]
    events = daeabe_events() if case == 'daeabe7' else chords_events()
    spb = 60 / bpm
    total_beats = max(b + d for b, d, _ in events) + 2
    n = int(total_beats * spb * sr) + sr
    gL = np.zeros(n); gR = np.zeros(n)
    truth = []
    # per string: last note end for cutting
    notes = []
    for b, d, frets in events:
        for k, (sn, f) in enumerate(sorted(frets, key=lambda x: -x[0])):
            s = 6 - sn
            t0 = b * spb + k * 0.008
            notes.append([t0, d * spb, tuning[s] + capo + f, s, f])
    notes.sort()
    for i, nt in enumerate(notes):
        for j in range(i + 1, len(notes)):
            if notes[j][3] == nt[3] and notes[j][0] > nt[0]:
                nt[1] = min(nt[1], notes[j][0] - nt[0])
                break
    for t0, dur, pitch, s, f in notes:
        freq = 440 * 2 ** ((pitch - 69) / 12)
        x = ks(freq, dur + 0.01) * 0.25
        a = int(t0 * sr)
        x = x[: n - a]
        gL[a:a + len(x)] += x * (1 - pan) / 2 * 2
        gR[a:a + len(x)] += x * (1 + pan) / 2 * 2
        truth.append({'t0': t0, 't1': t0 + dur, 'pitch': int(pitch), 'string': int(s), 'fret': int(f)})
    mixL, mixR = gL.copy(), gR.copy()
    if drums:
        for b in np.arange(0, total_beats, 1.0):
            a = int(b * spb * sr); L = min(int(0.2 * sr), n - a)
            env = np.exp(-np.arange(L) / (0.04 * sr))
            if int(b) % 2 == 0:
                k = 0.5 * np.sin(2 * np.pi * 50 * np.arange(L) / sr * (1 + 1.5 * env)) * env
            else:
                k = 0.25 * rng.standard_normal(L) * np.exp(-np.arange(L) / (0.06 * sr))
            mixL[a:a + L] += k; mixR[a:a + L] += k
        for b in np.arange(0.5, total_beats, 1.0):
            a = int(b * spb * sr); L = min(int(0.04 * sr), n - a)
            h = 0.06 * rng.standard_normal(L) * np.exp(-np.arange(L) / (0.008 * sr))
            mixL[a:a + L] += h; mixR[a:a + L] += h
    peak = max(np.abs(mixL).max(), np.abs(mixR).max())
    mix = np.stack([mixL, mixR], 1) / (peak * 1.05)
    sf.write(out_prefix + '.wav', mix.astype(np.float32), sr, subtype='PCM_16')
    json.dump({'tuning': tuning, 'capo': capo, 'bpm': bpm, 'notes': truth}, open(out_prefix + '.truth.json', 'w'))
    print('wrote', out_prefix, len(truth), 'notes', mix.shape[0] / sr, 's')

if __name__ == '__main__':
    render(sys.argv[2], sys.argv[1])
