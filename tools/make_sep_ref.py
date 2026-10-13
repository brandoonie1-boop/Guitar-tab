"""Reference separation with the original PyTorch demucs apply_model, for checking the JS port."""
import sys, json, numpy as np, torch
from fractions import Fraction
from safetensors.torch import load_file
from demucs.htdemucs import HTDemucs
from demucs.apply import apply_model

torch.set_grad_enabled(False)
meta = json.load(open('models/5c90dfd2.json'))
kw = json.loads(meta['kwargs'])
kw['segment'] = Fraction(kw['segment']['numerator'], kw['segment']['denominator'])
model = HTDemucs(**kw)
model.load_state_dict(load_file('models/htdemucs_6s.safetensors'))
model.eval()

sr = 44100
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
n = int(sr * dur)
rng = np.random.default_rng(1)

def ks_pluck(freq, length, decay=0.996):
    p = int(sr / freq)
    buf = rng.uniform(-1, 1, p)
    out = np.zeros(length)
    for i in range(length):
        out[i] = buf[i % p]
        buf[i % p] = decay * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out

mixL = np.zeros(n); mixR = np.zeros(n)
# guitar arpeggio (left-ish) and chords (right-ish)
notes = [52, 59, 64, 67, 71, 64, 67, 59]
t = 0.0
k = 0
while t < dur - 1:
    f = 440 * 2 ** ((notes[k % len(notes)] - 69) / 12)
    s = int(t * sr)
    L = min(int(1.2 * sr), n - s)
    pl = ks_pluck(f, L) * 0.3
    mixL[s:s + L] += pl * 0.8
    mixR[s:s + L] += pl * 0.3
    t += 0.25
    k += 1
# bass
tt = np.arange(n) / sr
bass = 0.25 * np.sin(2 * np.pi * 82.4 * tt) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 1 * tt)))
mixL += bass; mixR += bass
# drums: kick + hats
for b in np.arange(0, dur, 0.5):
    s = int(b * sr); L = min(int(0.15 * sr), n - s)
    env = np.exp(-np.arange(L) / (0.03 * sr))
    kick = 0.6 * np.sin(2 * np.pi * 55 * np.arange(L) / sr * (1 + 2 * env)) * env
    mixL[s:s + L] += kick; mixR[s:s + L] += kick
for b in np.arange(0.25, dur, 0.5):
    s = int(b * sr); L = min(int(0.05 * sr), n - s)
    hat = 0.15 * rng.standard_normal(L) * np.exp(-np.arange(L) / (0.01 * sr))
    mixL[s:s + L] += hat; mixR[s:s + L] += hat
mix = np.stack([mixL, mixR]).astype(np.float32)
mix /= np.abs(mix).max() * 1.1
mix.tofile('models/sep_in.f32')

wav = torch.from_numpy(mix)
ref = wav.mean(0)
mean, std = ref.mean(), ref.std() + 1e-8
out = apply_model(model, ((wav - mean) / std)[None], shifts=0, split=True, overlap=0.25, progress=False)
out = out * std + mean
out = out[0].numpy()  # [S, 2, n]
for i, s in enumerate(model.sources):
    out[i].astype(np.float32).tofile(f'models/sep_ref_{s}.f32')
print('done', mix.shape, 'sources', model.sources)
