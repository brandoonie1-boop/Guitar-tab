"""Fine-tune the guitar note model (guitar-fl: Kong et al.'s CRNN as trained for guitar by Riley,
Edwards & Dixon) so it hears notes through distortion and amps about as well as on a clean guitar.

Each training clip is a clean recording (direct input or a clean amp microphone from
tools/hrgt_data.py, or a synthetic part from tools/synth_riffs.py) sent through a random guitar rig
at 48 kHz: pickup voicing, sometimes a drive pedal, one or two clipping stages (tanh, arctan, hard,
asymmetric, ...), tone stack, a speaker cabinet (synthetic, or one of the app's two measured ones),
sometimes a room, a compressor and noise; or left clean. Targets:
  - distillation: what the original model gives for the clean clip (all four outputs), so the new
    model hears the processed clip the way the original hears the clean one, without picking up
    the timing habits or glitches of one dataset's labels;
  - the annotations: Kong et al.'s onset/offset regression and frame targets. Guitar-TECHS onsets
    move to the original model's onset peak within 60 ms on the clean clip (its MIDI pickup lags
    more on the low strings) and short notes the model cannot hear there are left out (pickup
    glitches); synthetic labels are exact.
A source marked always clean (the GuitarSet anchor of tools/hrgt_data.py, label weight 0) only
asks the new model to keep the original's output on clean acoustic guitar, which the electric
training data would otherwise let it drift away from.
Frozen: the input normalisation, every branch's first convolution block and the velocity branch's
convolutions; BatchNorm statistics stay as trained.

usage: python tools/train_hrgt.py <guitar-fl.pth> <out dir> --data <dir>:<weight>:<label weight>:<snap 0|1>[:<always clean>] ...
         --cabs assets/sound-cabs-<sha>.0.bin --gs <GuitarSet annotations.json>:<audio dir>
         [--steps 3000] [--batch 4] [--lr 1e-4] [--val-every 300] [--threads 4] [--bench]
Writes <out dir>/last.pt (resumable), step_<n>.pt at each validation, log.jsonl.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import fftconvolve, resample_poly, sosfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrgt_torch import load_model, score  # noqa: E402

SR = 16000
CLIP = 160000  # 10 s = 1001 frames
T = CLIP // 160 + 1
FPS = 100
BEGIN = 21


# ---- a random guitar rig ------------------------------------------------------------------------

def biquad(kind, f0, q, gain_db, sr):
    """RBJ biquad as one SOS row."""
    A = 10 ** (gain_db / 40)
    w = 2 * math.pi * f0 / sr
    c, s = math.cos(w), math.sin(w)
    al = s / (2 * q)
    if kind == 'lp':
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]; a = [1 + al, -2 * c, 1 - al]
    elif kind == 'hp':
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; a = [1 + al, -2 * c, 1 - al]
    elif kind == 'peak':
        b = [1 + al * A, -2 * c, 1 - al * A]; a = [1 + al / A, -2 * c, 1 - al / A]
    elif kind in ('hshelf', 'lshelf'):
        r = 2 * math.sqrt(A) * al
        if kind == 'hshelf':
            b = [A * ((A + 1) + (A - 1) * c + r), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - r)]
            a = [(A + 1) - (A - 1) * c + r, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - r]
        else:
            b = [A * ((A + 1) - (A - 1) * c + r), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - r)]
            a = [(A + 1) + (A - 1) * c + r, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - r]
    else:
        raise ValueError(kind)
    return np.array([b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]])


def shaper(kind, v):
    if kind == 'tanh':
        return np.tanh(v)
    if kind == 'atan':
        return (2 / np.pi) * np.arctan(v * np.pi / 2)
    if kind == 'hard':
        return np.clip(v, -1, 1)
    if kind == 'cubic':
        u = np.clip(v, -1, 1)
        return 1.5 * (u - u ** 3 / 3)
    if kind == 'asym':
        return np.where(v >= 0, np.tanh(v), np.tanh(0.6 * v) / 0.6 * 0.8)
    if kind == 'diode':
        return v / (1 + np.abs(v))
    raise ValueError(kind)


STYLES = ['clean', 'edge', 'crunch', 'high', 'fuzz']
STYLE_P = [0.16, 0.12, 0.22, 0.38, 0.12]


def rig(x, r, cabs, style=None):
    """x: clean 16 kHz clip -> the same clip through a random rig (16 kHz)."""
    sr = 48000
    y = resample_poly(x.astype(np.float64), 3, 1)
    y /= np.abs(y).max() + 1e-9
    style = style or r.choice(STYLES, p=STYLE_P)
    sos = [biquad('hshelf', r.uniform(1500, 4000), 0.7, r.uniform(-6, 6), sr), biquad('lshelf', r.uniform(150, 400), 0.7, r.uniform(-4, 4), sr)]
    y = sosfilt(np.vstack(sos), y)
    # a drive pedal in front (mid hump, soft clip, top cut)
    if (style in ('crunch', 'high') and r.random() < 0.4) or r.random() < 0.1:
        y = sosfilt(np.vstack([biquad('hp', r.uniform(300, 700), 0.7, 0, sr), biquad('peak', r.uniform(650, 1100), 0.8, r.uniform(4, 10), sr)]), y) * 0.6 + 0.4 * y
        g = 10 ** (r.uniform(6, 22) / 20)
        y = np.tanh(g * y / (np.abs(y).max() + 1e-9))
        y = sosfilt(biquad('lp', r.uniform(4000, 7000), 0.7, 0, sr)[None], y)
    drive = {'clean': r.uniform(-6, 4), 'edge': r.uniform(4, 14), 'crunch': r.uniform(14, 26), 'high': r.uniform(24, 40), 'fuzz': r.uniform(34, 50)}[style]
    stages = 2 if style == 'high' and r.random() < 0.6 else 1
    kind = 'hard' if style == 'fuzz' and r.random() < 0.5 else r.choice(['tanh', 'tanh', 'atan', 'cubic', 'asym', 'diode'])
    tight = r.uniform(60, 250) if style in ('high', 'fuzz') else r.uniform(30, 120)
    y = sosfilt(np.vstack([biquad('hp', tight, 0.7, 0, sr), biquad('hshelf', r.uniform(800, 2500), 0.7, r.uniform(0, 8), sr)]), y)
    y /= np.sqrt(np.mean(y ** 2)) + 1e-9  # drive relative to the clip's level
    for k in range(stages):
        g = 10 ** ((drive if k == 0 else drive * r.uniform(0.4, 0.8)) / 20) * 0.1
        bias = r.uniform(-0.2, 0.2) if kind != 'asym' else 0.0
        y = shaper(kind, g * y + bias) - shaper(kind, np.array(bias))
        if stages > 1 and k == 0:
            y = sosfilt(np.vstack([biquad('lp', r.uniform(5000, 9000), 0.7, 0, sr), biquad('hp', r.uniform(60, 200), 0.7, 0, sr)]), y)
            y /= np.sqrt(np.mean(y ** 2)) + 1e-9
    # tone stack
    y = sosfilt(np.vstack([biquad('lshelf', r.uniform(100, 300), 0.7, r.uniform(-6, 6), sr),
                           biquad('peak', r.uniform(400, 1000), r.uniform(0.5, 1.5), r.uniform(-10, 6), sr),
                           biquad('hshelf', r.uniform(2000, 5000), 0.7, r.uniform(-6, 6), sr)]), y)
    if r.random() < 0.3:  # power amp squeezing a little
        p = np.abs(y).max() + 1e-9
        g2 = r.uniform(1, 3)
        y = np.tanh(g2 * y / p) / np.tanh(g2) * p
    # speaker cabinet
    if style == 'clean' and r.random() < 0.3:
        pass  # straight into the desk
    elif cabs and r.random() < 0.3:
        ir = cabs[r.integers(len(cabs))]
        y = fftconvolve(y, ir)[:len(y)]
    else:
        sos = [biquad('hp', r.uniform(60, 130), 0.7, 0, sr), biquad('lp', r.uniform(3500, 7000), r.uniform(0.6, 1.0), 0, sr),
               biquad('lp', r.uniform(4500, 9000), 0.7, 0, sr), biquad('peak', r.uniform(1500, 4000), r.uniform(1, 4), r.uniform(2, 7), sr),
               biquad('peak', r.uniform(400, 900), r.uniform(0.7, 2), r.uniform(-6, -1), sr)]
        if r.random() < 0.5:
            sos.append(biquad('peak', r.uniform(90, 200), r.uniform(1, 3), r.uniform(1, 6), sr))
        y = sosfilt(np.vstack(sos), y)
    y = y.astype(np.float32)
    import pedalboard as pb  # noqa: E402 (heavy import, only when training)
    fx = []
    if r.random() < 0.3:
        fx.append(pb.Reverb(room_size=r.uniform(0.05, 0.4), wet_level=r.uniform(0.03, 0.2), dry_level=1.0, damping=0.5))
    if r.random() < 0.25:
        fx.append(pb.Compressor(threshold_db=r.uniform(-30, -10), ratio=r.uniform(2, 6), attack_ms=r.uniform(1, 20), release_ms=r.uniform(50, 200)))
    if fx:
        y /= np.abs(y).max() + 1e-9
        y = pb.Pedalboard(fx)(y, sr)
    y = resample_poly(y.astype(np.float64), 1, 3)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    if r.random() < 0.35:
        n = len(y)
        noise = r.standard_normal(n) * rms * 10 ** (-r.uniform(40, 70) / 20)
        t = np.arange(n) / SR
        f = 50 if r.random() < 0.5 else 60
        hum = sum(np.sin(2 * np.pi * f * h * t + r.uniform(0, 6.3)) / h for h in range(1, 6)) * rms * 10 ** (-r.uniform(45, 70) / 20)
        y = y + noise + hum
    y *= 10 ** (r.uniform(-32, -10) / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9)
    return np.clip(y, -1, 1).astype(np.float32)


def separate_sim(g, b, r, gamma=(0.6, 1.8), floor=0.15, jitter=0.35, smooth=(6, 5)):
    """A guitar isolated from guitar g + backing b (16 kHz) by a simulated imperfect separator:
    the ideal ratio mask on the mixture's spectrogram raised to a power (harder or softer
    decisions), smeared over frequency and time, jittered, with a floor that lets some of the
    band through. Ranges: gamma (lo, hi), floor and jitter up to the given value, smooth up to
    (freq bins, frames)."""
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft
    n = len(g)
    _, _, G = stft(g, SR, nperseg=1024, noverlap=768)
    _, _, B = stft(b, SR, nperseg=1024, noverlap=768)
    pg, pb = np.abs(G) ** 2, np.abs(B) ** 2
    m = pg / (pg + pb + 1e-10)
    m = m ** r.uniform(*gamma)
    m = uniform_filter(m, size=(int(r.integers(1, smooth[0] + 1)), int(r.integers(1, smooth[1] + 1))), mode='nearest')
    m = m * np.exp(r.normal(0, r.uniform(0.02, max(0.03, jitter)), m.shape))
    fl = r.uniform(0.0, floor)
    m = np.clip(fl + (1 - fl) * m, 0, 1)
    _, y = istft(m * (G + B), SR, nperseg=1024, noverlap=768)
    return y[:n].astype(np.float32) if len(y) >= n else np.pad(y, (0, n - len(y))).astype(np.float32)


def tanh_amp(x, g, sr=SR):
    """The JS test's simulated amp (tests/eval_notes.mjs --fx tanh:G)."""
    pk = np.abs(x).max() + 1e-9
    d = np.tanh(g * x / pk) / np.tanh(g)
    a = math.exp(-2 * math.pi * 4500 / sr)
    from scipy.signal import lfilter
    return (0.5 * lfilter([1 - a], [1, -a], d)).astype(np.float32)


# ---- data ---------------------------------------------------------------------------------------

class Source:
    """Recordings of one manifest (tools/hrgt_data.py, tools/synth_riffs.py). Per recording, a
    manifest entry can say kd: 0 (no distillation: the original model can't hear it clean, e.g. a
    guitar recorded distorted) and norig: true (already through an amp: no rig on top)."""

    def __init__(self, d, weight, gt_w, snap, clean_only=False, band=0.0, kd=1.0, exclude_sets=()):
        man = [m for m in json.load(open(os.path.join(d, 'manifest.json'))) if m.get('set') not in exclude_sets]
        self.weight, self.gt_w, self.snap, self.clean_only, self.band = weight, gt_w, snap, clean_only, band
        self.audio = [np.load(os.path.join(d, m['audio']), mmap_mode='r') for m in man]
        self.notes = [np.array([n[:3] for n in m['notes']], np.float64).reshape(-1, 3) for m in man]
        self.names = [m['name'] for m in man]
        self.kd = [kd * float(m.get('kd', 1)) for m in man]
        self.norig = [bool(m.get('norig', False)) for m in man]
        dur = np.array([len(a) for a in self.audio], float)
        self.p = dur / dur.sum()
        self.hours = dur.sum() / SR / 3600

    def sample(self, r):
        """(clip, notes in the clip, kd weight, may go through a rig)"""
        i = r.choice(len(self.audio), p=self.p)
        a = self.audio[i]
        n = self.notes[i]
        if len(a) <= CLIP:
            s = 0
            x = np.zeros(CLIP, np.float32)
            x[:len(a)] = a[:] / 32000
        else:
            if len(n) and r.random() < 0.85:  # mostly where the part plays (song stems are often silent)
                s = int(np.clip((n[r.integers(len(n)), 0] - r.uniform(0.5, 9.5)) * SR, 0, len(a) - CLIP))
            else:
                s = int(r.integers(0, len(a) - CLIP))
            x = np.asarray(a[s:s + CLIP], np.float32) / 32000
        t0 = s / SR
        sel = n[(n[:, 1] > t0 - 0.1) & (n[:, 0] < t0 + CLIP / SR + 0.1)].copy()
        sel[:, :2] -= t0
        return x, sel, self.kd[i], not self.norig[i]


class Backing:
    """Band backing for mixing (tools/hrgt_data.py slakh: per song drums.npy, bass.npy, other.npy,
    16 kHz; no guitars): a random stretch of a random song, the parts at random levels."""

    def __init__(self, d):
        self.songs = []
        for s in sorted(os.listdir(d)):
            parts = [np.load(os.path.join(d, s, f), mmap_mode='r') for f in ('drums.npy', 'bass.npy', 'other.npy') if os.path.exists(os.path.join(d, s, f))]
            if parts:
                self.songs.append(parts)

    def sample(self, r, n=CLIP):
        parts = self.songs[r.integers(len(self.songs))]
        L = min(len(p) for p in parts)
        s = int(r.integers(0, max(1, L - n)))
        y = np.zeros(n, np.float32)
        for p in parts:
            if r.random() < 0.85:  # now and then a part sits out
                seg = np.asarray(p[s:s + n], np.float32) / 32000
                y[:len(seg)] += seg * 10 ** (r.uniform(-8, 2) / 20)
        return y


def band_mix(g, backing, r):
    """The (processed) guitar g in a band, isolated again by the simulated separator."""
    b = backing.sample(r, len(g))
    gr, br = np.sqrt(np.mean(g ** 2)) + 1e-9, np.sqrt(np.mean(b ** 2)) + 1e-9
    if br < 1e-5:
        return g
    b = b * (gr / br) * 10 ** (-r.uniform(-10, 3) / 20)  # guitar from 10 dB under the band to 3 dB over it
    return separate_sim(g, b, r)


def gt_targets(notes, teacher_onset=None, snap=False):
    """Kong et al.'s targets for one clip: onset/offset regression (1 - |dt| / 50 ms) and frames.
    snap: move each onset to the teacher's onset peak within 60 ms; drop short notes it doesn't hear."""
    on = np.zeros((T, 88), np.float32)
    off = np.zeros((T, 88), np.float32)
    fr = np.zeros((T, 88), np.float32)
    ft = np.arange(T) / FPS
    for t0, t1, p in notes:
        k = int(round(p)) - BEGIN
        if not 0 <= k < 88:
            continue
        if snap and teacher_onset is not None:
            c = int(round(t0 * FPS))
            lo, hi = max(0, c - 6), min(T, c + 7)
            if hi > lo:
                seg = teacher_onset[lo:hi, k]
                j = int(np.argmax(seg))
                if seg[j] > 0.25:
                    i = lo + j
                    sh = 0.0
                    if 0 < i < T - 1:
                        a, v, b = teacher_onset[i - 1, k], teacher_onset[i, k], teacher_onset[i + 1, k]
                        sh = (b - a) / (v - b) / 2 if a > b and v > b else ((b - a) / (v - a) / 2 if v > a else 0.0)
                    t1 += (i + sh) / FPS - t0
                    t0 = (i + sh) / FPS
                elif t1 - t0 < 0.06:
                    continue
        a, b = max(0, int(round(t0 * FPS)) - 6), min(T, int(round(t0 * FPS)) + 7)
        if b > a:
            on[a:b, k] = np.maximum(on[a:b, k], np.clip(1 - np.abs(ft[a:b] - t0) / 0.05, 0, 1))
        a, b = max(0, int(round(t1 * FPS)) - 6), min(T, int(round(t1 * FPS)) + 7)
        if b > a:
            off[a:b, k] = np.maximum(off[a:b, k], np.clip(1 - np.abs(ft[a:b] - t1) / 0.05, 0, 1))
        a, b = max(0, int(round(t0 * FPS))), min(T, int(round(t1 * FPS)) + 1)
        if b > a:
            fr[a:b, k] = 1
    return on, off, fr


# ---- model --------------------------------------------------------------------------------------

BRANCHES = ['frame_model', 'reg_onset_model', 'reg_offset_model', 'velocity_model']


def setup_student(net):
    """Trainable: convolution blocks 2-4, fc5/bn5 and the recurrent layers of the frame, onset and
    offset branches; the velocity branch's recurrent layers; the two conditioning GRUs. Everything
    else stays as trained, including the STFT and mel filterbank (torchlibrosa keeps them as
    parameters)."""
    for p in net.parameters():
        p.requires_grad = False
    train = []
    for b in BRANCHES[:3]:
        m = getattr(net, b)
        train += [m.conv_block2, m.conv_block3, m.conv_block4, m.fc5, m.bn5, m.gru, m.fc]
    train += [net.velocity_model.gru, net.velocity_model.fc, net.reg_onset_gru, net.reg_onset_fc, net.frame_gru, net.frame_fc]
    for m in train:
        for p in m.parameters():
            p.requires_grad = True
    return [p for p in net.parameters() if p.requires_grad]


def train_mode(net):
    net.train()
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def bce(p, t, w=None):
    p = p.clamp(1e-6, 1 - 1e-6)
    l = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
    if w is None:
        return l.mean(dim=(1, 2))
    return (l * w).sum(dim=(1, 2)) / (w.sum(dim=(1, 2)) + 1e-6)


# ---- validation -----------------------------------------------------------------------------------

def val_sets(gtechs_dir, gs, cabs, backing=None):
    sets = {}
    if gtechs_dir:
        man = [m for m in json.load(open(os.path.join(gtechs_dir, 'manifest.json'))) if m['set'] == 'P3_music' and m['kind'] == 'di']
        items = [(np.load(os.path.join(gtechs_dir, m['audio'])).astype(np.float32) / 32000, m['notes']) for m in man[:6]]
        sets['p3_clean'] = items
        sets['p3_heavy'] = [(rig(x, np.random.default_rng(1000 + i), cabs, style='high'), n) for i, (x, n) in enumerate(items)]
        if backing:
            sets['p3_band'] = [(band_mix(x, backing, np.random.default_rng(2000 + i)), n) for i, (x, n) in enumerate(items)]
    if gs:
        ann_path, audio_dir = gs
        ann = json.load(open(ann_path))
        names = sorted(ann)[1::3][::5][:8]  # not among the test takes (sorted(ann)[::3])
        items = []
        for nm in names:
            x, sr = sf.read(os.path.join(audio_dir, nm + '_mic.wav'), dtype='float32', always_2d=True)
            x = resample_poly(x.mean(1), SR // math.gcd(sr, SR), sr // math.gcd(sr, SR)).astype(np.float32)
            items.append((x, [(t, t + d, v) for t, d, v, *_ in ann[nm]['notes']]))
        sets['gs_clean'] = items
        sets['gs_tanh30'] = [(tanh_amp(x, 30), n) for x, n in items]
        if backing:
            sets['gs_band'] = [(band_mix(x, backing, np.random.default_rng(3000 + i)), n) for i, (x, n) in enumerate(items)]
    return sets


def validate(net, sets):
    net.eval()
    res = {}
    for k, items in sets.items():
        F, P, R, off = score(net, items)
        res[k] = round(100 * F, 2)
    return res


# ---- main ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--data', action='append', required=True,
                    help='<dir>:<weight>:<label weight>:<snap 0|1>[:<always clean 0|1>], or '
                         '<dir>,w=..,gt=..,snap=0|1,clean=0|1,band=<share mixed into a band>,kd=<distillation weight>')
    ap.add_argument('--init', help='start the new model from this checkpoint instead of the original (the teacher stays the original)')
    ap.add_argument('--backing', help='band backing (tools/hrgt_data.py slakh) for sources with band=...')
    ap.add_argument('--gs', help='<GuitarSet annotations.json>:<audio dir> for validation')
    ap.add_argument('--val-extra', action='append', default=[], help='<name>=<manifest dir>: more validation (e.g. synthetic parts through a bridge pickup)')
    ap.add_argument('--cabs', help='cabinet impulse responses (48 kHz): a folder of .wav files, or the app\'s assets/sound-cabs-*.bin')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--kd', type=float, default=1.0)
    ap.add_argument('--clean-p', type=float, default=0.12)
    ap.add_argument('--val-every', type=int, default=300)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--no-channels-last', action='store_true')
    ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    cabs = []
    if a.cabs and os.path.isfile(a.cabs):  # the app's packed cabinet responses (assets/sound-cabs-*.bin)
        import io
        import struct
        b = open(a.cabs, 'rb').read()
        n = struct.unpack('<I', b[4:8])[0]
        for at, ln in json.loads(b[8:8 + n]).values():
            ir, sr = sf.read(io.BytesIO(b[8 + n + at:8 + n + at + ln]), dtype='float64')
            cabs.append(ir / np.sqrt(np.sum(ir ** 2)))
    elif a.cabs:
        for f in sorted(os.listdir(a.cabs)):
            if f.endswith('.wav'):
                ir, sr = sf.read(os.path.join(a.cabs, f), dtype='float64')
                cabs.append(ir / np.sqrt(np.sum(ir ** 2)))
    sources = []
    gtechs_dir = None
    for spec in a.data:
        if '=' in spec:
            d, *kv = spec.split(',')
            o = dict(x.split('=') for x in kv)
            w, gw, snap, clean_only, band = float(o.get('w', 1)), float(o.get('gt', 0.5)), o.get('snap') == '1', o.get('clean') == '1', float(o.get('band', 0))
            kd_w = float(o.get('kd', 1))
        else:
            d, w, gw, snap, *more = spec.split(':')
            w, gw, snap, clean_only, band, kd_w = float(w), float(gw), snap == '1', bool(more) and more[0] == '1', 0.0, 1.0
        sources.append(Source(d, w, gw, snap, clean_only, band, kd_w, exclude_sets=('P3_music',)))
        if os.path.basename(os.path.normpath(d)) == 'gtechs':
            gtechs_dir = d
        print(f'data {d}: {len(sources[-1].names)} recordings, {sources[-1].hours:.2f} h, weight {w}, label weight {gw}, snap {snap}'
              f'{", always clean" if clean_only else ""}{f", {band:.0%} in a band" if band else ""}{f", distillation x{kd_w:g}" if kd_w != 1 else ""}', flush=True)
    sw = np.array([s.weight for s in sources])
    sw /= sw.sum()
    backing = Backing(a.backing) if a.backing else None
    if backing:
        print(f'backing: {len(backing.songs)} songs', flush=True)

    teacher = load_model(a.ckpt)
    for p in teacher.parameters():
        p.requires_grad = False
    student = load_model(a.init or a.ckpt)
    if not a.no_channels_last:  # about 25 % faster convolutions on the processor
        teacher = teacher.to(memory_format=torch.channels_last)
        student = student.to(memory_format=torch.channels_last)
    params = setup_student(student)
    print('trainable parameters', sum(p.numel() for p in params), 'of', sum(p.numel() for p in student.parameters()), flush=True)
    opt = torch.optim.Adam(params, lr=a.lr)
    step = 0
    last = os.path.join(a.out, 'last.pt')
    rng = np.random.default_rng(1)
    if os.path.exists(last):
        ck = torch.load(last, map_location='cpu', weights_only=False)
        student.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        step = ck['step']
        rng = np.random.default_rng(1000 + step)
        print('resumed at step', step, flush=True)
    sets = val_sets(gtechs_dir, a.gs.split(':', 1) if a.gs else None, cabs, backing) if not a.bench else {}
    for spec in a.val_extra:
        name, d = spec.split('=', 1)
        man = json.load(open(os.path.join(d, 'manifest.json')))
        sets[name] = [(np.load(os.path.join(d, m['audio'])).astype(np.float32) / 32000, m['notes']) for m in man]
    if step == 0 and sets:
        res = validate(teacher, sets)
        print('val original', res, flush=True)
        if a.init:
            res0 = validate(student, sets)
            print('val start', res0, flush=True)
        with open(os.path.join(a.out, 'log.jsonl'), 'a') as f:
            f.write(json.dumps({'step': 0, 'val': res}) + '\n')

    def lr_at(s):
        warm = 100
        if s < warm:
            return a.lr * (s + 1) / warm
        q = (s - warm) / max(1, a.steps - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, q))))

    names = {'onset': 'reg_onset_output', 'offset': 'reg_offset_output', 'frame': 'frame_output', 'velocity': 'velocity_output'}
    t_start = time.time()
    while step < a.steps:
        tb = time.time()
        clean, proc, gts, gtw, kdw, snaps = [], [], [], [], [], []
        for _ in range(a.batch):
            src = sources[rng.choice(len(sources), p=sw)]
            x, notes, kd_i, may_rig = src.sample(rng)
            clean.append(x)
            y = x if (src.clean_only or not may_rig or rng.random() < a.clean_p) else rig(x, rng, cabs)
            if backing and rng.random() < src.band:
                y = band_mix(y, backing, rng)
            proc.append(y * (10 ** (rng.uniform(-30, -10) / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9)))
            gts.append(notes)
            gtw.append(src.gt_w)
            kdw.append(kd_i)
            snaps.append(src.snap)
        t_data = time.time() - tb
        with torch.no_grad():
            tout = teacher(torch.from_numpy(np.stack(clean)))
        targets = [gt_targets(n, tout['reg_onset_output'][i].numpy(), s) for i, (n, s) in enumerate(zip(gts, snaps))]
        g_on = torch.from_numpy(np.stack([t[0] for t in targets]))
        g_off = torch.from_numpy(np.stack([t[1] for t in targets]))
        g_fr = torch.from_numpy(np.stack([t[2] for t in targets]))
        w_gt = torch.tensor(gtw, dtype=torch.float32)
        w_kd = a.kd * torch.tensor(kdw, dtype=torch.float32)
        train_mode(student)
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=a.bf16):
            out = student(torch.from_numpy(np.stack(proc)))
        out = {k: v.float() for k, v in out.items()}
        kd = (bce(out[names['onset']], tout[names['onset']]) + bce(out[names['offset']], tout[names['offset']]) +
              bce(out[names['frame']], tout[names['frame']]) + bce(out[names['velocity']], tout[names['velocity']], tout[names['onset']]))
        gt = bce(out[names['onset']], g_on) + bce(out[names['offset']], g_off) + bce(out[names['frame']], g_fr)
        loss = (w_kd * kd + w_gt * gt).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 3.0)
        opt.step()
        step += 1
        dt = time.time() - tb
        if step % 10 == 0 or a.bench:
            print(f'step {step} loss {loss.item():.4f} kd {kd.mean().item():.4f} gt {gt.mean().item():.4f} lr {lr_at(step):.2e} '
                  f'{dt:.1f} s/step (data {t_data:.1f} s) elapsed {(time.time() - t_start) / 3600:.2f} h', flush=True)
        if a.bench and step >= 3:
            import resource
            print(f'peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB', flush=True)
            return
        if step % 50 == 0 or step == a.steps:
            torch.save({'model': student.state_dict(), 'opt': opt.state_dict(), 'step': step}, last + '.tmp')
            os.replace(last + '.tmp', last)
        if sets and (step % a.val_every == 0 or step == a.steps):
            res = validate(student, sets)
            print(f'val step {step}', res, flush=True)
            torch.save({'model': student.state_dict(), 'step': step, 'val': res}, os.path.join(a.out, f'step_{step}.pt'))
            with open(os.path.join(a.out, 'log.jsonl'), 'a') as f:
                f.write(json.dumps({'step': step, 'val': res}) + '\n')


if __name__ == '__main__':
    main()
