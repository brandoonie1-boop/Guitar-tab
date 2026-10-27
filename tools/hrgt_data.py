"""Training data for fine-tuning the guitar note model (tools/train_hrgt.py).

Writes one 16 kHz int16 .npy per recording into <out dir> and a manifest.json with the notes
of each recording, already shifted to line up with its audio.

  Guitar-TECHS (https://zenodo.org/records/14963133, CC BY 4.0): every take has a direct-input
  and an amp-microphone recording, and per-string MIDI from a pickup that runs a little late.
  The lag is found per recording from the audio's onsets (the MIDI onsets against a spectral-flux
  onset envelope, whose peaks come 22.5 ms before the onsets the note model reports: measured on
  GuitarSet, where the model's onsets and the annotations agree to 0.3 ms on average);
  recordings where under 60 % of the onsets line up are left out,
  and so are the bend and harmonics takes (the written pitch does not describe the sound).

  GuitarSet anchor (https://zenodo.org/records/3371780, CC BY 4.0): microphone recordings of the
  takes the tests do not use (not sorted(takes)[::3], not the training script's validation takes),
  without their notes: training only asks the new model to keep giving the original model's
  output on this clean acoustic guitar (distillation), so it doesn't drift away from acoustic
  guitar while it learns distortion. No annotations are used.

  Slakh (BabySlakh, https://zenodo.org/records/4603870, CC BY 4.0: songs rendered from MIDI, each
  instrument a separate stem with its MIDI): the guitar stems with their notes (exact; recorded
  distorted programs are marked to get no rig on top and no distillation), and the rest of each
  song as band backing for training (drums, bass, everything else; no guitars).

usage: python tools/hrgt_data.py gtechs <Guitar-TECHS dir> <out dir>
       python tools/hrgt_data.py gs-anchor <GuitarSet annotations.json> <audio dir> <out dir>
       python tools/hrgt_data.py slakh <babyslakh_16k dir> <guitar out dir> <backing out dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from midi import read_midi  # noqa: E402

SR = 16000
STD = [40, 45, 50, 55, 59, 64]
GT_STRING = {'E': 0, 'A': 1, 'D': 2, 'G': 3, 'B': 4, 'e': 5}
SKIP = ('Bendings', 'Harmonics')  # also PinchHarmonics
ENV_LEAD = 0.0225  # onset-envelope peaks precede the note model's onsets by this much


def load16k(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def onset_envelope(x, hop=80, nfft=1024):
    """Spectral flux at 200 frames/s (log magnitude, 30 Hz - 8 kHz), normalised, and the time of
    each frame (its window's centre)."""
    n = 1 + (len(x) - nfft) // hop
    win = np.hanning(nfft).astype(np.float32)
    lo, hi = int(30 * nfft / SR), int(8000 * nfft / SR)
    mags = []
    for s in range(0, n, 4096):  # in blocks to bound memory
        idx = np.arange(nfft)[None, :] + hop * np.arange(s, min(n, s + 4096))[:, None]
        mags.append(np.log1p(10 * np.abs(np.fft.rfft(x[idx] * win, axis=1)[:, lo:hi])).astype(np.float32))
    mag = np.concatenate(mags)
    env = np.zeros(n, np.float32)
    env[1:] = np.maximum(0, np.diff(mag, axis=0)).sum(1)
    env -= np.convolve(env, np.ones(41) / 41, mode='same')  # remove the slow trend
    env = np.maximum(env, 0)
    return env / (env.max() + 1e-9), (np.arange(n) * hop + nfft / 2) / SR


def measure_lag(x, onsets):
    """Lag (s) to subtract from the MIDI times, and the share of onsets that line up after it."""
    env, t = onset_envelope(x)
    fps = 1 / (t[1] - t[0])
    peaks = np.where((env[1:-1] > 0.08) & (env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:]))[0] + 1
    pt = t[peaks]
    ons = np.unique(np.round(np.asarray(onsets) * 200) / 200)
    # coarse: correlation of the envelope with the onset train, smoothed (10 ms)
    best = (0.0, -1.0)
    for lag in np.arange(-0.15, 0.1501, 0.0025):
        tt = ons - lag
        fi = np.clip(np.round((tt - t[0]) * fps).astype(int), 0, len(env) - 1)
        score = float(np.sum(np.maximum(env[fi], np.maximum(env[np.maximum(fi - 1, 0)], env[np.minimum(fi + 1, len(env) - 1)]))))
        if score > best[1]:
            best = (lag, score)
    lag = best[0]
    # share of onsets with an envelope peak within 25 ms
    tt = ons - lag
    j = np.searchsorted(pt, tt)
    d = np.minimum(np.abs(pt[np.clip(j, 0, len(pt) - 1)] - tt), np.abs(pt[np.clip(j - 1, 0, len(pt) - 1)] - tt))
    return float(lag - ENV_LEAD), float(np.mean(d <= 0.025)) if len(ons) else 0.0


def gtechs(src, out):
    os.makedirs(out, exist_ok=True)
    man = []
    for set_dir in sorted(glob.glob(os.path.join(src, 'P[0-9]_*'))):
        set_name = os.path.basename(set_dir)
        for mf in sorted(glob.glob(os.path.join(set_dir, 'midi', '*.mid'))):
            take = os.path.basename(mf)[5:-4]  # midi_<take>.mid
            if any(s in take for s in SKIP):
                continue
            notes = []
            for tr in read_midi(mf):
                s = GT_STRING.get(tr['name'].strip())
                if s is None:
                    continue
                for t0, t1, p, v in tr['notes']:
                    if 0 <= p - STD[s] <= 24 and t1 > t0:
                        notes.append([t0, t1, p, v])
            notes.sort()
            if not notes:
                continue
            for kind, short in (('directinput', 'di'), ('micamp', 'mic')):
                ap = os.path.join(set_dir, 'audio', kind, f'{kind}_{take}.wav')
                if not os.path.exists(ap):
                    continue
                x = load16k(ap)
                lag, match = measure_lag(x, [n[0] for n in notes])
                name = f'{set_name}_{take}_{short}'
                ok = match >= 0.6
                print(f'{name:40s} {len(x) / SR:6.1f} s  notes {len(notes):5d}  lag {1000 * lag:6.1f} ms  matched {100 * match:4.0f}%{"" if ok else "  (left out)"}', flush=True)
                if not ok:
                    continue
                pk = np.abs(x).max() + 1e-9
                np.save(os.path.join(out, name + '.npy'), np.round(x / pk * 32000).astype(np.int16))
                shifted = [[round(max(0.0, a - lag), 4), round(max(0.02, b - lag), 4), p, v] for a, b, p, v in notes]
                man.append({'name': name, 'source': 'gtechs', 'group': set_name[:2], 'set': set_name, 'take': take, 'kind': short,
                            'audio': name + '.npy', 'dur': len(x) / SR, 'lag': lag, 'match': match, 'notes': shifted})
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'))
    print('recordings', len(man), 'hours', round(sum(m['dur'] for m in man) / 3600, 2), 'notes', sum(len(m['notes']) for m in man))


def gs_anchor(ann_path, audio_dir, out):
    os.makedirs(out, exist_ok=True)
    names = sorted(json.load(open(ann_path)))
    test = set(names[::3])
    val = set(names[1::3][::5][:8])  # tools/train_hrgt.py validation takes
    man = []
    for nm in names:
        if nm in test or nm in val:
            continue
        x = load16k(os.path.join(audio_dir, nm + '_mic.wav'))
        pk = np.abs(x).max() + 1e-9
        np.save(os.path.join(out, nm + '.npy'), np.round(x / pk * 32000).astype(np.int16))
        man.append({'name': nm, 'source': 'guitarset', 'group': nm[:2], 'set': 'gs_anchor', 'take': nm, 'kind': 'mic',
                    'audio': nm + '.npy', 'dur': len(x) / SR, 'notes': []})
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'))
    print('recordings', len(man), 'hours', round(sum(m['dur'] for m in man) / 3600, 2), '(no notes: distillation only)')


DISTORTED_PROGRAMS = {29, 30}  # General MIDI: overdriven guitar, distortion guitar
# The rendered notes sound later than their MIDI note-on (the instrument patches' attacks): where
# the original note model hears the onsets on clean BabySlakh guitar parts, median per program
# (nylon 12.5 ms, steel 9-25, jazz 28-34, clean 33-36, muted 47); the distorted patches get the
# electric figure.
PROGRAM_DELAY = {24: 0.0125, 25: 0.017, 26: 0.031, 27: 0.034, 28: 0.04}
DEFAULT_DELAY = 0.033


def slakh(src, gout, bout, label_delay=None):
    import yaml
    os.makedirs(gout, exist_ok=True)
    os.makedirs(bout, exist_ok=True)
    man = []
    for track in sorted(glob.glob(os.path.join(src, 'Track*'))):
        meta = yaml.safe_load(open(os.path.join(track, 'metadata.yaml')))
        tname = os.path.basename(track)
        sums = {}
        for sid, st in sorted(meta['stems'].items()):
            wav = os.path.join(track, 'stems', sid + '.wav')
            if not os.path.exists(wav):  # (BabySlakh's audio_rendered flags are all false; the files say what was rendered)
                continue
            x = load16k(wav)
            if st.get('inst_class') == 'Guitar':
                mid = os.path.join(track, 'MIDI', sid + '.mid')
                dl = label_delay if label_delay is not None else PROGRAM_DELAY.get(st.get('program_num'), DEFAULT_DELAY)
                notes = [[round(a + dl, 4), round(b + dl, 4), p, v] for tr in read_midi(mid) for a, b, p, v in tr['notes'] if 21 <= p <= 108 and b > a] if os.path.exists(mid) else []
                if len(notes) < 20 or np.abs(x).max() < 1e-4:
                    continue
                notes.sort()
                dist = st.get('program_num') in DISTORTED_PROGRAMS
                name = f'{tname}_{sid}'
                np.save(os.path.join(gout, name + '.npy'), np.round(x / (np.abs(x).max() + 1e-9) * 32000).astype(np.int16))
                man.append({'name': name, 'source': 'slakh', 'group': tname, 'set': 'slakh', 'take': name, 'kind': 'dist' if dist else 'clean',
                            'program': st.get('program_num'), 'audio': name + '.npy', 'dur': len(x) / SR, 'kd': 0 if dist else 1, 'norig': dist, 'notes': notes})
                print(f'{name:22s} {st.get("midi_program_name", "?"):24s} {len(x) / SR:6.1f} s notes {len(notes):5d}{"  (recorded distorted)" if dist else ""}', flush=True)
            else:
                key = 'drums' if st.get('is_drum') else ('bass' if st.get('inst_class') == 'Bass' else 'other')
                if key in sums:
                    n = min(len(sums[key]), len(x))
                    sums[key] = sums[key][:n] + x[:n]
                else:
                    sums[key] = x
        if sums:
            os.makedirs(os.path.join(bout, tname), exist_ok=True)
            for key, x in sums.items():
                np.save(os.path.join(bout, tname, key + '.npy'), np.round(np.clip(x / (np.abs(x).max() + 1e-9), -1, 1) * 32000).astype(np.int16))
    json.dump(man, open(os.path.join(gout, 'manifest.json'), 'w'))
    print('guitar parts', len(man), 'hours', round(sum(m['dur'] for m in man) / 3600, 2), 'notes', sum(len(m['notes']) for m in man),
          '| backing songs', len(os.listdir(bout)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['gtechs', 'gs-anchor', 'slakh'])
    ap.add_argument('src')
    ap.add_argument('rest', nargs='+')
    ap.add_argument('--label-delay', type=float, default=None, help='slakh: seconds to add to the MIDI times (default: per program, PROGRAM_DELAY)')
    a = ap.parse_args()
    if a.kind == 'gtechs':
        gtechs(a.src, a.rest[0])
    elif a.kind == 'gs-anchor':
        gs_anchor(a.src, a.rest[0], a.rest[1])
    else:
        slakh(a.src, a.rest[0], a.rest[1], a.label_delay)


if __name__ == '__main__':
    main()
