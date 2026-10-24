"""Sample banks for the tab player (app/guitarsynth.js): one recording per string and fret.

usage:
  python tools/build_guitar_sounds.py electric <Guitar-TECHS notes dir (gtechs_notes.py)> <out dir>
  python tools/build_guitar_sounds.py nylon <Iowa MIS guitar samples dir> <out dir>
  python tools/build_guitar_sounds.py acoustic <Guitar-TECHS notes dir>,<GuitarSet dir> <out dir>
  python tools/build_guitar_sounds.py ir <npy impulse response> <out.wav>
  python tools/build_guitar_sounds.py pack sounds/electric.bin mp3=<dir>/electric.mp3 json=<dir>/electric.json
  python tools/build_guitar_sounds.py pack sounds/cabs.bin bright=<dir>/cab-bright.wav dark=<dir>/cab-dark.wav

A bank is <name>.mp3 (every note one after another, with a short gap) and <name>.json:
{sr, notes: [{s: string (0 = low E), f: fret, at: seconds into the file, len, cents}]}.
The player finds each note's attack near `at` itself (MP3 decoders differ in their start delay).
"""
import glob, json, os, subprocess, sys
import numpy as np

SR = 48000
NOTE_LEN = 2.5  # seconds kept per note
FADE = 0.25
GAP = 0.25
OPEN = [40, 45, 50, 55, 59, 64]
FRETS = list(range(0, 13)) + list(range(14, 23, 2))


def load(p, sr=SR):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def attack(x, frac=0.1):
    env = np.convolve(np.abs(x), np.ones(96) / 96, mode='same')
    i = int(np.argmax(env > frac * env.max()))
    # back to where the attack starts to rise
    floor = np.median(env[: max(1, i)]) if i > 10 else 0
    while i > 0 and env[i - 1] > max(2 * floor, 0.02 * env.max()):
        i -= 1
    return max(0, i - int(0.004 * SR))


def shape(x, cents=0.0):
    a = attack(x)
    y = x[a: a + int(NOTE_LEN * SR)].copy()
    if len(y) < int(NOTE_LEN * SR):
        y = np.pad(y, (0, int(NOTE_LEN * SR) - len(y)))
    n = int(FADE * SR)
    y[-n:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n))
    y[:48] *= np.linspace(0, 1, 48)  # no click at the cut
    return y


def write_bank(out_dir, name, notes, target_db=-20.0):
    """notes: list of (string, fret, samples, cents). Levels: the first 300 ms at target_db."""
    os.makedirs(out_dir, exist_ok=True)
    parts, index, at = [], [], GAP
    for s, f, y, cents in notes:
        r = np.sqrt(np.mean(y[: int(0.3 * SR)] ** 2)) + 1e-9
        y = y * (10 ** (target_db / 20) / r)
        pk = np.abs(y).max()
        if pk > 0.95:
            y *= 0.95 / pk
        parts += [y, np.zeros(int(GAP * SR))]
        index.append({'s': s, 'f': f, 'at': round(at, 4), 'len': round(len(y) / SR, 4), 'cents': round(float(cents), 1)})
        at += (len(y) + int(GAP * SR)) / SR
    audio = np.concatenate([np.zeros(int(GAP * SR))] + parts).astype(np.float32)
    raw = os.path.join(out_dir, name + '.f32')
    audio.tofile(raw)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', raw,
                    '-c:a', 'libmp3lame', '-b:a', '128k', os.path.join(out_dir, name + '.mp3')], check=True)
    os.remove(raw)
    json.dump({'sr': SR, 'notes': index}, open(os.path.join(out_dir, name + '.json'), 'w'), separators=(',', ':'))
    print(name, len(index), 'notes', round(at / 60, 1), 'min', round(os.path.getsize(os.path.join(out_dir, name + '.mp3')) / 1e6, 2), 'MB')


def electric(src, out):
    idx = json.load(open(os.path.join(src, 'index.json')))
    by = {(r['player'], r['signal'], r['string'], r['fret']): r for r in idx}
    notes = []
    for s in range(6):
        for f in FRETS:
            # player 1's direct-input recording; player 2's if it is missing or out of tune
            for player in ['P1', 'P2']:
                r = by.get((player, 'directinput', s, f))
                if r and abs(r['cents']) < 40:
                    x = np.fromfile(os.path.join(src, r['name'] + '.f32'), np.float32).astype(np.float64)
                    notes.append((s, f, shape(x), r['cents']))
                    break
            else:
                print('  no take for string', s, 'fret', f)
    write_bank(out, 'electric', notes)


def nylon(src, out):
    """Iowa MIS: Guitar.ff.<string>.<note>.aiff; string from the file name, fret from the pitch."""
    names = {'sulE': 0, 'sulA': 1, 'sulD': 2, 'sulG': 3, 'sulB': 4, 'sul_E': 5}
    pcs = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
    have = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith('.aiff'):
            continue
        parts = fn.split('.')
        s = names[parts[2]]
        note = parts[3]
        midi = 12 * (int(note[-1]) + 1) + pcs[note[:-1]]
        f = midi - OPEN[s]
        if 0 <= f <= 22:
            have[(s, f)] = os.path.join(src, fn)
    notes = []
    for s in range(6):
        for f in FRETS:
            p = have.get((s, f))
            if p:
                x = load(p)
                # a gentle high-pass: the recordings carry some low rumble
                x = np.convolve(x, [1, -1], mode='same') * 0.5 + x * 0.5
                notes.append((s, f, shape(x), 0.0))
    write_bank(out, 'nylon', notes)


def avg_spectrum(xs, n=8192):
    """Mean power spectrum of the first 0.6 s after each attack."""
    acc = np.zeros(n // 2 + 1)
    w = np.hanning(n)
    for x in xs:
        a = attack(x)
        for k in range(3):
            seg = x[a + k * n // 2: a + k * n // 2 + n]
            if len(seg) < n:
                break
            acc += np.abs(np.fft.rfft(seg * w)) ** 2
    return acc / max(1, len(xs))


def smooth_oct(p, frac=1 / 3):
    """Fractional-octave smoothing of a power spectrum (bins of an 8192-point FFT at 48 kHz)."""
    f = np.arange(len(p)) * SR / ((len(p) - 1) * 2)
    out = np.empty_like(p)
    c = np.concatenate([[0], np.cumsum(p)])
    for i in range(len(p)):
        lo = f[i] * 2 ** (-frac / 2); hi = f[i] * 2 ** (frac / 2)
        a = int(np.searchsorted(f, lo)); b = max(a + 1, int(np.searchsorted(f, hi)))
        out[i] = (c[b] - c[a]) / (b - a)
    return out


def acoustic(src, out):
    """Steel-string acoustic: the electric guitar's direct-input notes (real steel strings ringing)
    with each string's tone matched to GuitarSet's miked steel-string acoustic. The match compares
    moments where GuitarSet's annotations have one note sounding alone (the microphone hears just
    that string) with the direct-input note on the same string and fret.
    src: '<Guitar-TECHS notes dir (gtechs_notes.py)>,<GuitarSet dir (annotation/, audio_mono-mic/)>'"""
    di_dir, gs_dir = src.split(',')
    idx = json.load(open(os.path.join(di_dir, 'index.json')))
    by = {(r['player'], r['signal'], r['string'], r['fret']): r for r in idx}
    di_cache = {}

    def di_note(s, fr):
        r = by.get(('P1', 'directinput', s, fr)) or by.get(('P2', 'directinput', s, fr))
        if not r or abs(r['cents']) >= 40:
            return None
        if r['name'] not in di_cache:
            di_cache[r['name']] = np.fromfile(os.path.join(di_dir, r['name'] + '.f32'), np.float32).astype(np.float64)
        return di_cache[r['name']]

    n = 4096
    w = np.hanning(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    Pm = np.zeros((6, n // 2 + 1)); Pd = np.zeros((6, n // 2 + 1)); cnt = [0] * 6
    for jp in sorted(glob.glob(os.path.join(gs_dir, 'annotation', '*.jams'))):
        take = os.path.basename(jp)[:-5]
        d = json.load(open(jp))
        notes = []
        for a in d['annotations']:
            if a['namespace'] == 'note_midi':
                st = int(a['annotation_metadata']['data_source'])
                notes += [(x['time'], x['time'] + x['duration'], x['value'], st) for x in a['data']]
        notes.sort()
        mic = None
        for i, (t0, t1, v, st) in enumerate(notes):
            fr = int(round(v)) - OPEN[st]
            if t1 - t0 < 0.25 or not 0 <= fr <= 15:
                continue
            # nothing else sounding from 50 ms before to the end of the analysis window
            win_end = t0 + 0.05 + n / 44100
            if any(o is not notes[i] and o[0] < win_end and o[1] > t0 - 0.05 for o in notes[max(0, i - 8): i + 8]):
                continue
            x = di_note(st, fr)
            if x is None:
                continue
            if mic is None:
                mic = load(os.path.join(gs_dir, 'audio_mono-mic', take + '_mic.wav'))  # resampled to 48 kHz
            a = int((t0 + 0.02) * SR)
            seg = mic[a: a + n]
            if len(seg) < n:
                continue
            ad = attack(x) + int(0.02 * SR)
            Pm[st] += np.abs(np.fft.rfft(seg * w)) ** 2 / max(1e-12, (seg ** 2).mean())
            sd = x[ad: ad + n]
            Pd[st] += np.abs(np.fft.rfft(sd * w)) ** 2 / max(1e-12, (sd ** 2).mean())
            cnt[st] += 1
    print('  single-note moments per string:', cnt)
    notes_out = []
    for s in range(6):
        g = np.sqrt(smooth_oct(Pm[s]) / np.maximum(smooth_oct(Pd[s]), 1e-20))
        g = np.clip(g, 10 ** (-20 / 20), 10 ** (24 / 20))
        g[f < 70] *= (f[f < 70] / 70) ** 2
        g /= np.sqrt(np.mean(g[(f > 80) & (f < 6000)] ** 2))
        h = np.fft.irfft(g, n)
        h = np.roll(h, n // 2)[n // 2 - 1024: n // 2 + 1024] * np.hanning(2048)
        print(f'  string {s}: match EQ at 100/250/500/1k/2k/4k/8k Hz:', [round(float(20 * np.log10(g[np.argmin(np.abs(f - q))])), 1) for q in [100, 250, 500, 1000, 2000, 4000, 8000]])
        for fr in FRETS:
            x = di_note(s, fr)
            if x is None:
                continue
            r = by.get(('P1', 'directinput', s, fr)) or by.get(('P2', 'directinput', s, fr))
            y = np.convolve(x, h)[1024: 1024 + len(x)]
            notes_out.append((s, fr, shape(y), r['cents']))
    write_bank(out, 'acoustic', notes_out)


def ir(src, out):
    h = np.load(src).astype(np.float64)
    h /= np.sqrt((h ** 2).sum())
    pcm = (np.clip(h / np.abs(h).max() * 0.9, -1, 1) * 32767).astype('<i2')
    import wave
    with wave.open(out, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    print('ir', out, len(h), 'samples')


def pack(out, *items):
    """One asset file: b'GTSB', u32 header length, JSON header {name: [offset, length]}, then the files.
    items: name=path ..."""
    blobs, head, at = [], {}, 0
    for it in items:
        name, path = it.split('=', 1)
        b = open(path, 'rb').read()
        head[name] = [at, len(b)]
        blobs.append(b)
        at += len(b)
    h = json.dumps(head, separators=(',', ':')).encode()
    with open(out, 'wb') as f:
        f.write(b'GTSB' + len(h).to_bytes(4, 'little') + h)
        for b in blobs:
            f.write(b)
    print('packed', out, round(os.path.getsize(out) / 1e6, 2), 'MB', list(head))


if __name__ == '__main__':
    if sys.argv[1] == 'pack':
        pack(sys.argv[2], *sys.argv[3:])
    else:
        {'electric': electric, 'nylon': nylon, 'acoustic': acoustic, 'ir': ir}[sys.argv[1]](sys.argv[2], sys.argv[3])
