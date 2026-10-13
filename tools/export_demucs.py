"""Export htdemucs_6s to an ONNX "core" graph with STFT/iSTFT moved outside.

Inputs:  mag  [1, 4, 2048, 336]   complex-as-channels spectrogram of the stereo mix
         mix  [1, 2, 343980]      stereo waveform segment (7.8 s @ 44.1 kHz)
Outputs: spec [1, 6, 4, 2048, 336] per-source spectrogram (complex-as-channels)
         wave [1, 6, 2, 343980]    per-source time-branch waveform
final source waveform = istft(spec) + wave
"""
import json, sys, math, time
from fractions import Fraction
import numpy as np
import torch
from safetensors.torch import load_file
from demucs.htdemucs import HTDemucs

torch.set_grad_enabled(False)
torch.backends.mha.set_fastpath_enabled(False)
meta = json.load(open('models/5c90dfd2.json')) if False else None


def load_model(path_json, path_st):
    meta = json.load(open(path_json))
    kw = json.loads(meta['kwargs'])
    seg = kw['segment']
    kw['segment'] = Fraction(seg['numerator'], seg['denominator'])
    model = HTDemucs(**kw)
    sd = load_file(path_st)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('missing', missing, 'unexpected', unexpected)
    model.eval()
    return model


class Core(torch.nn.Module):
    def __init__(self, m: HTDemucs):
        super().__init__()
        self.m = m

    def forward(self, mag, mix):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None]  # broadcast over time (no expand -> small constant)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            x, xt = m.crosstransformer(x, xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        L = mix.shape[-1]
        xt = xt.view(B, S, -1, L)
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


# ---------- numpy reference STFT / iSTFT (mirrors torch + demucs exactly) ----------
NFFT, HOP = 4096, 1024
WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float64)  # periodic hann


def np_spec(x):  # x: [C, L]
    L = x.shape[-1]
    le = int(math.ceil(L / HOP))
    pad = HOP // 2 * 3
    xp = np.pad(x, ((0, 0), (pad, pad + le * HOP - L)), mode='reflect')
    xp = np.pad(xp, ((0, 0), (NFFT // 2, NFFT // 2)), mode='reflect')  # center=True
    nfr = 1 + (xp.shape[-1] - NFFT) // HOP
    assert nfr == le + 4, (nfr, le)
    out = np.zeros((x.shape[0], NFFT // 2 + 1, nfr), np.complex128)
    for t in range(nfr):
        fr = xp[:, t * HOP: t * HOP + NFFT] * WIN
        out[:, :, t] = np.fft.rfft(fr, axis=-1) / math.sqrt(NFFT)
    z = out[:, :-1, 2:2 + le]
    return z  # [C, 2048, le]


def np_ispec(z, length):  # z: [..., 2048, le] complex
    lead = z.shape[:-2]
    zz = z.reshape(-1, z.shape[-2], z.shape[-1])
    zz = np.pad(zz, ((0, 0), (0, 1), (2, 2)))  # +1 freq bin, +2 frames each side
    pad = HOP // 2 * 3
    le = HOP * int(math.ceil(length / HOP)) + 2 * pad
    nfr = zz.shape[-1]
    olen = (nfr - 1) * HOP + NFFT
    y = np.zeros((zz.shape[0], olen))
    env = np.zeros(olen)
    for t in range(nfr):
        fr = np.fft.irfft(zz[:, :, t] * math.sqrt(NFFT), n=NFFT, axis=-1) * WIN
        y[:, t * HOP: t * HOP + NFFT] += fr
        env[t * HOP: t * HOP + NFFT] += WIN ** 2
    start = NFFT // 2
    y = y[:, start:start + le] / env[start:start + le]
    y = y[:, pad:pad + length]
    return y.reshape(*lead, length)


def cac(z):  # [C, F, T] complex -> [C*2, F, T]
    C, F_, T = z.shape
    m = np.stack([z.real, z.imag], axis=1)  # [C, 2, F, T]
    return m.reshape(C * 2, F_, T)


def uncac(m, S, C):  # [S, C*2, F, T] -> [S, C, F, T] complex
    m = m.reshape(S, C, 2, m.shape[-2], m.shape[-1])
    return m[:, :, 0] + 1j * m[:, :, 1]


if __name__ == '__main__':
    model = load_model('models/5c90dfd2.json', 'models/htdemucs_6s.safetensors')
    nparams = sum(p.numel() for p in model.parameters())
    print('params', nparams, 'sources', model.sources, 'segment', model.segment)
    L = int(model.segment * model.samplerate)
    print('training length', L)
    torch.manual_seed(0)
    np.random.seed(0)
    # test signal: some sines + noise burst, stereo
    t = np.arange(L) / 44100
    sig = np.stack([
        0.3 * np.sin(2 * np.pi * 196 * t) * np.exp(-((t % 0.5) * 4)) + 0.05 * np.random.randn(L),
        0.3 * np.sin(2 * np.pi * 246.9 * t) * np.exp(-((t % 0.25) * 6)) + 0.05 * np.random.randn(L),
    ]).astype(np.float32)
    mix = torch.from_numpy(sig)[None]
    t0 = time.time()
    ref = model(mix)[0].numpy()  # [S, 2, L]
    print('torch full forward', time.time() - t0, 's')

    z = np_spec(sig.astype(np.float64))
    mag = torch.from_numpy(cac(z).astype(np.float32))[None]
    core = Core(model).eval()
    spec, wave = core(mag, mix)
    out = np_ispec(uncac(spec[0].numpy().astype(np.float64), 6, 2), L) + wave[0].numpy()
    err = np.abs(out - ref).max()
    print('numpy-STFT + core vs torch full: max abs err', err, 'ref max', np.abs(ref).max())

    if '--export' in sys.argv:
        t0 = time.time()
        torch.onnx.export(core, (mag, mix), 'models/htdemucs_6s_core.onnx', opset_version=17,
                          input_names=['mag', 'mix'], output_names=['spec', 'wave'], dynamo=False,
                          do_constant_folding=True)
        print('exported in', time.time() - t0)
        np.save('models/test_sig.npy', sig)
        np.save('models/test_ref.npy', ref)
