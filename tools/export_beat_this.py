"""Export a Beat This! checkpoint (CPJKU/beat_this, MIT) to ONNX for the app.

Input  spect [1, 1500, 128]: log-mel spectrogram, 50 frames/s (see engine/beatnet.js)
Output beat [1, 1500], downbeat [1, 1500]: logits

usage: python tools/export_beat_this.py small0.ckpt out.onnx [queries per attention block]
"""
import sys

import numpy as np
import torch
from beat_this.inference import load_model
from beat_this.model import roformer

torch.set_grad_enabled(False)
ckpt, out = sys.argv[1], sys.argv[2]
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 150


def chunked_attend(self, q, k, v):
    """scaled_dot_product_attention computed for blocks of queries: same result, and the
    attention matrices (32 x 1500 x 1500 in the front end) never exist all at once."""
    scale = (self.scale if self.scale is not None else q.shape[-1] ** -0.5)
    kt = k.transpose(-2, -1) * scale
    outs = []
    for s0 in range(0, q.shape[-2], CHUNK):
        a = torch.softmax(torch.matmul(q[..., s0:s0 + CHUNK, :], kt), dim=-1)
        outs.append(torch.matmul(a, v))
    return torch.cat(outs, dim=-2)


model = load_model(ckpt, 'cpu').eval()
x0 = torch.randn(1, 1500, 128).abs()
ref0 = model(x0)
roformer.Attend.forward = chunked_attend
got0 = model(x0)
print('chunked attention vs original: max diff', float((ref0['beat'] - got0['beat']).abs().max()))


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, spect):
        o = self.m(spect)
        return o['beat'], o['downbeat']


w = Wrap(model).eval()
x = torch.randn(1, 1500, 128).abs()
ref = w(x)
torch.onnx.export(w, (x,), out, opset_version=17, input_names=['spect'], output_names=['beat', 'downbeat'], dynamo=False, do_constant_folding=True)
import onnxruntime as ort  # noqa: E402
s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
got = s.run(None, {'spect': x.numpy()})
for r, g, n in zip(ref, got, ['beat', 'downbeat']):
    print(n, 'max abs diff vs torch', float(np.abs(r.numpy() - g).max()))
print('params', sum(p.numel() for p in model.parameters()))
