"""The guitar note model in PyTorch, as tools/export_hrgt.py exports it: loading a checkpoint,
running it over a whole recording like the app does (10 s windows, half overlap, keeping the
middle of each), and scoring the notes against annotations. Used by tools/train_hrgt.py."""
import numpy as np
import torch
from piano_transcription_inference.models import Regress_onset_offset_frame_velocity_CRNN

from hrgt_notes import decode, match, f1

SR = 16000
WIN = 160000  # 10 s: 1001 frames
HOP = 80000


def load_model(ckpt):
    net = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=100, classes_num=88)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    net.load_state_dict(sd['model'] if 'model' in sd else sd, strict=True)
    return net.eval()


@torch.no_grad()
def infer(net, x, batch=4):
    """Frame outputs (T, 88) for mono 16 kHz audio x (numpy float32)."""
    T = len(x) // 160 + 1
    n = max(1, int(np.ceil((len(x) - WIN) / HOP)) + 1) if len(x) > WIN else 1
    pad = np.zeros((n - 1) * HOP + WIN, np.float32)
    pad[:len(x)] = x
    starts = [k * HOP for k in range(n)]
    out = {k: np.zeros((T, 88), np.float32) for k in ('onset', 'offset', 'frame', 'velocity')}
    names = {'onset': 'reg_onset_output', 'offset': 'reg_offset_output', 'frame': 'frame_output', 'velocity': 'velocity_output'}
    fh = HOP // 160  # 500 frames
    M = (1001 - 1 - fh) // 2  # 250
    for b in range(0, n, batch):
        seg = torch.from_numpy(np.stack([pad[s:s + WIN] for s in starts[b:b + batch]]))
        od = net(seg)
        for j, s in enumerate(starts[b:b + batch]):
            k = b + j
            f0 = s // 160
            a = 0 if k == 0 else M
            e = 1001 if k == n - 1 else M + fh
            lo, hi = f0 + a, min(T, f0 + e)
            if hi <= lo:
                continue
            for key, nm in names.items():
                out[key][lo:hi] = od[nm][j, a:a + hi - lo].numpy()
    return out


def score(net, items, thr=(0.4, 0.3, 0.1)):
    """items: [(audio16k, [[t0, t1, pitch, ...]])] -> (F1, P, R, mean onset offset in ms)"""
    TP = NE = NR = 0
    D = []
    for x, notes in items:
        est = decode(infer(net, x), *thr)
        ref = [(n[0], n[1], int(round(n[2]))) for n in notes]  # GuitarSet pitches are averaged (63.98)
        tp, d = match(est, ref)
        TP += tp
        NE += len(est)
        NR += len(ref)
        D += d
    F, P, R = f1(TP, NE, NR)
    return F, P, R, 1000 * float(np.mean(D)) if D else 0.0
