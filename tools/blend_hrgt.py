"""Blend a fine-tuned note-model checkpoint (tools/train_hrgt.py) with the original, weight by
weight (WiSE-FT, Wortsman et al., "Robust fine-tuning of zero-shot models", CVPR 2022):
w = (1 - alpha) * original + alpha * fine-tuned. Fine-tuning for distortion costs some accuracy
on clean acoustic guitar; the blend gets nearly all of it back and keeps most of the gain.

usage: python tools/blend_hrgt.py <guitar-fl.pth> <fine-tuned .pt> <alpha> <out .pt>
"""
import sys

import torch


def main():
    orig_path, ft_path, alpha, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    orig = torch.load(orig_path, map_location='cpu', weights_only=False)
    orig = orig.get('model', orig)
    ft = torch.load(ft_path, map_location='cpu', weights_only=False)['model']
    assert orig.keys() == ft.keys(), 'not the same model'
    blend = {k: ((1 - alpha) * orig[k] + alpha * ft[k]).contiguous() if orig[k].dtype.is_floating_point else orig[k] for k in orig}
    torch.save({'model': blend, 'blend': {'original': orig_path, 'fine_tuned': ft_path, 'alpha': alpha}}, out)
    print(f'wrote {out}: {alpha:g} fine-tuned + {1 - alpha:g} original')


if __name__ == '__main__':
    main()
