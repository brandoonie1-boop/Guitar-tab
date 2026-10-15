"""Export a high-resolution guitar transcription checkpoint to ONNX parts for the app.

The model is Kong et al.'s high-resolution transcription CRNN (piano_transcription_inference,
MIT) trained for guitar by Riley, Edwards and Dixon (GAPS / Francois Leduc checkpoints,
huggingface.co/xavriley/midi-transcription-models, MIT).

Parts (float32 graphs, packed for the app by tools/pack_demucs.py --name hrgt.chain):
  conv_frame, conv_onset, conv_offset, conv_velocity
      mel [1, T, 229] log-mel (dB) of 16 kHz audio, n_fft 2048, hop 160 (100 frames/s)
      -> feat [1, T, 768]   (bn0, four conv blocks, fc5/bn5/relu)
      The convolutions only see +-8 frames in time, so the app runs these on chunks with
      8 frames of context on each side and keeps the middle.
  head
      feat_frame, feat_onset, feat_offset, feat_velocity [1, T, 768]
      -> onset, offset, frame, velocity [1, T, 88]   (the recurrent layers)
      Run on 10 s windows (1001 frames) with 50 % overlap, like the original inference.

usage: python tools/export_hrgt.py <checkpoint.pth> <out dir> [--check audio.wav]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from piano_transcription_inference.models import Regress_onset_offset_frame_velocity_CRNN

torch.set_grad_enabled(False)
BRANCHES = ['frame', 'onset', 'offset', 'velocity']
MODELS = {'frame': 'frame_model', 'onset': 'reg_onset_model', 'offset': 'reg_offset_model', 'velocity': 'velocity_model'}


class Conv(torch.nn.Module):
    def __init__(self, net, branch):
        super().__init__()
        self.bn0 = net.bn0
        self.m = getattr(net, MODELS[branch])

    def forward(self, mel):
        x = mel[:, None]  # (1, 1, T, 229)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        m = self.m
        x = m.conv_block1(x, pool_size=(1, 2), pool_type='avg')
        x = m.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = m.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = m.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = x.transpose(1, 2).flatten(2)
        return F.relu(m.bn5(m.fc5(x).transpose(1, 2)).transpose(1, 2))


class Head(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def branch(self, name, feat):
        m = getattr(self.net, MODELS[name])
        x, _ = m.gru(feat)
        return torch.sigmoid(m.fc(x))

    def forward(self, feat_frame, feat_onset, feat_offset, feat_velocity):
        n = self.net
        frame = self.branch('frame', feat_frame)
        onset = self.branch('onset', feat_onset)
        offset = self.branch('offset', feat_offset)
        velocity = self.branch('velocity', feat_velocity)
        x = torch.cat((onset, (onset ** 0.5) * velocity), dim=2)
        x, _ = n.reg_onset_gru(x)
        onset = torch.sigmoid(n.reg_onset_fc(x))
        x = torch.cat((frame, onset, offset), dim=2)
        x, _ = n.frame_gru(x)
        frame = torch.sigmoid(n.frame_fc(x))
        return onset, offset, frame, velocity


def logmel(net, audio):
    x = net.spectrogram_extractor(audio)
    return net.logmel_extractor(x)[:, 0]  # (1, T, 229)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--check')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    net = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=100, classes_num=88)
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=False)['model'], strict=True)
    net.eval()
    T = 1001
    mel = torch.randn(1, T, 229) * 10 - 40
    parts = []
    feats = {}
    for b in BRANCHES:
        conv = Conv(net, b).eval()
        f = f'conv_{b}'
        torch.onnx.export(conv, (mel,), os.path.join(a.out, f + '.onnx'), input_names=['mel'], output_names=['feat'],
                          dynamic_axes={'mel': {1: 'T'}, 'feat': {1: 'T'}}, opset_version=17, dynamo=False)
        feats[b] = conv(mel)
        parts.append({'file': f, 'in': {'mel': 'mel'}, 'out': {'feat': f'feat_{b}'}})
    head = Head(net).eval()
    names = [f'feat_{b}' for b in BRANCHES]
    torch.onnx.export(head, tuple(feats[b] for b in BRANCHES), os.path.join(a.out, 'head.onnx'), input_names=names,
                      output_names=['onset', 'offset', 'frame', 'velocity'],
                      dynamic_axes={**{n: {1: 'T'} for n in names}, **{n: {1: 'T'} for n in ['onset', 'offset', 'frame', 'velocity']}},
                      opset_version=17, dynamo=False)
    parts.append({'file': 'head', 'in': {n: n for n in names}, 'out': {n: n for n in ['onset', 'offset', 'frame', 'velocity']}})
    json.dump({'segment': 'hrgt', 'model': os.path.basename(a.ckpt), 'sr': 16000, 'nfft': 2048, 'hop': 160, 'nMels': 229,
               'fmin': 30, 'fmax': 8000, 'window': 1001, 'context': 8, 'parts': parts},
              open(os.path.join(a.out, 'chain.json'), 'w'), indent=1)

    # check: the parts reproduce the full model on a 10 s window, and the chunked convs are exact
    import onnxruntime as ort
    so = ort.SessionOptions()
    sess = {p['file']: ort.InferenceSession(os.path.join(a.out, p['file'] + '.onnx'), so, providers=['CPUExecutionProvider']) for p in parts}
    if a.check:
        import librosa
        y, _ = librosa.load(a.check, sr=16000, mono=True)
        audio = torch.from_numpy(y[:160000][None].astype(np.float32))
    else:
        audio = torch.randn(1, 160000) * 0.05
    ref = net(audio)
    m = logmel(net, audio).numpy()
    fe = {}
    for b in BRANCHES:
        full = sess[f'conv_{b}'].run(None, {'mel': m})[0]
        # chunked with 8 frames of context
        out = np.zeros_like(full)
        C, K = 200, 8
        for s in range(0, m.shape[1], C):
            lo, hi = max(0, s - K), min(m.shape[1], s + C + K)
            o = sess[f'conv_{b}'].run(None, {'mel': np.ascontiguousarray(m[:, lo:hi])})[0]
            out[:, s:min(s + C, m.shape[1])] = o[:, s - lo:s - lo + min(C, m.shape[1] - s)]
        print(b, 'chunked vs full max diff', float(np.abs(out - full).max()))
        fe[f'feat_{b}'] = full
    got = sess['head'].run(None, fe)
    for name, g in zip(['reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output'], got):
        print(name, 'onnx vs torch max diff', float(np.abs(g - ref[name].numpy()).max()))


if __name__ == '__main__':
    main()
