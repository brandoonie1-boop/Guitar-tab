"""Blend a fine-tuned note-model checkpoint (tools/train_hrgt.py) with the original, weight by
weight (WiSE-FT, Wortsman et al., "Robust fine-tuning of zero-shot models", CVPR 2022):
w = (1 - alpha) * original + alpha * fine-tuned. Fine-tuning for distortion costs some accuracy
on clean acoustic guitar; the blend gets nearly all of it back and keeps most of the gain.
`average` takes the mean of checkpoints from one run first (less noise from the last steps).

usage: python tools/blend_hrgt.py <guitar-fl.pth> <fine-tuned .pt> <alpha> <out .pt>
       python tools/blend_hrgt.py average <a .pt> <b .pt> [...] <out .pt>
"""
import sys

import torch


def weights(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return d.get('model', d)


def main():
    if sys.argv[1] == 'average':
        srcs, out = sys.argv[2:-1], sys.argv[-1]
        ws = [weights(p) for p in srcs]
        assert all(w.keys() == ws[0].keys() for w in ws), 'not the same model'
        avg = {k: (sum(w[k] for w in ws) / len(ws)).contiguous() if ws[0][k].dtype.is_floating_point else ws[0][k] for k in ws[0]}
        torch.save({'model': avg, 'average': srcs}, out)
        print(f'wrote {out}: mean of {len(srcs)} checkpoints')
        return
    orig_path, ft_path, alpha, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    orig, ft = weights(orig_path), weights(ft_path)
    assert orig.keys() == ft.keys(), 'not the same model'
    blend = {k: ((1 - alpha) * orig[k] + alpha * ft[k]).contiguous() if orig[k].dtype.is_floating_point else orig[k] for k in orig}
    torch.save({'model': blend, 'blend': {'original': orig_path, 'fine_tuned': ft_path, 'alpha': alpha}}, out)
    print(f'wrote {out}: {alpha:g} fine-tuned + {1 - alpha:g} original')


if __name__ == '__main__':
    main()
