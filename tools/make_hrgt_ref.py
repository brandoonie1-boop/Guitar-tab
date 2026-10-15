"""Reference data for tests/test_hrgt.mjs: 16 kHz audio, torchlibrosa log-mel and the
original inference outputs (10 s segments, 50 % overlap) of a transcription checkpoint.
usage: python tools/make_hrgt_ref.py <ckpt> <audio> <out dir> [seconds]"""
import sys, os, json
import numpy as np, torch, librosa
from piano_transcription_inference.models import Regress_onset_offset_frame_velocity_CRNN
torch.set_grad_enabled(False)
ckpt, audio, out = sys.argv[1:4]
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 25
os.makedirs(out, exist_ok=True)
net = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=100, classes_num=88)
net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'], strict=True)
net.eval()
y, _ = librosa.load(audio, sr=16000, mono=True, duration=secs)
y = y.astype(np.float32)
y.tofile(f'{out}/audio16k.f32')
mel = net.logmel_extractor(net.spectrogram_extractor(torch.from_numpy(y[None])))[0, 0].numpy()
mel.astype(np.float32).tofile(f'{out}/mel.f32')
np.asarray(net.logmel_extractor.melW.detach().numpy(), np.float32).tofile(f'{out}/melW.f32')
SEG = 160000
n = len(y)
pad = int(np.ceil(n / SEG)) * SEG - n
a = np.concatenate([y, np.zeros(pad, np.float32)])
segs, p = [], 0
while p + SEG <= len(a):
    segs.append(a[p:p + SEG]); p += SEG // 2
outs = {}
for s in segs:
    o = net(torch.from_numpy(s[None]))
    for k, v in o.items(): outs.setdefault(k, []).append(v[0].numpy())
T = mel.shape[0]
for k, v in outs.items():
    x = np.stack(v)
    if x.shape[0] == 1: r = x[0]
    else:
        x = x[:, :-1]; L = x.shape[1]
        r = np.concatenate([x[0, :int(L * .75)]] + [x[i, int(L * .25):int(L * .75)] for i in range(1, x.shape[0] - 1)] + [x[-1, int(L * .25):]])
    r[:T].astype(np.float32).tofile(f'{out}/{k}.f32')
from piano_transcription_inference.utilities import RegressionPostProcessor
pp = RegressionPostProcessor(100, classes_num=88, onset_threshold=0.3, offset_threshold=0.3, frame_threshold=0.1, pedal_offset_threshold=0.2)
ev, _ = pp.output_dict_to_midi_events({k: np.fromfile(f'{out}/{k}.f32', np.float32).reshape(-1, 88) for k in outs})
json.dump({'T': int(T), 'n': int(n), 'notes': [[float(e['onset_time']), float(e['offset_time']), int(e['midi_note'])] for e in ev]}, open(f'{out}/ref.json', 'w'))
print('T', T, 'notes', len(ev))
