"""Export htdemucs_6s as a chain of smaller ONNX graphs that run one after another:
encoder -> transformer -> decoder layer 0 -> ... -> decoder layer 3.

Same inputs/outputs and the same maths as export_demucs.py's single "core" graph, but
the tensors handed between parts leave the WebAssembly heap between runs, so the
heap only has to hold one part's working memory at a time (about half the peak of the
single graph). The parts are written with a chain.json describing how to wire them.

usage (from the folder holding models/5c90dfd2.json and models/htdemucs_6s.safetensors):
  python export_demucs_parts.py --segment 343980 --out models/parts343980
then for each part: ORT basic optimisation + fp16 storage (see build notes in README).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_demucs as ed  # noqa: E402

torch.set_grad_enabled(False)
torch.backends.mha.set_fastpath_enabled(False)


import torch.nn.functional as F  # noqa: E402


def _rows_dconv(layer, y):
    """DConv over each frequency row (rows are independent: exact on any subset of rows)."""
    if not layer.dconv:
        return y
    B, C, Fr, T = y.shape
    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
    y = layer.dconv(y)
    return y.view(B, Fr, C, T).permute(0, 2, 1, 3)


def _check_identity_norms(layer):
    for n in ('norm1', 'norm2'):
        assert isinstance(getattr(layer, n, torch.nn.Identity()), torch.nn.Identity), 'chunking needs layers without GroupNorm'


def enc_freq_chunked(layer, x, rows_per_chunk, post=None):
    """HEncLayer.forward (frequency branch, no inject) computed a band of output rows at a
    time, so the big intermediate tensors are a fraction of the full size. Exact."""
    _check_identity_norms(layer)
    K, S, P = layer.kernel_size, layer.stride, layer.pad
    Fin = x.shape[2]
    Fout = (Fin + 2 * P - K) // S + 1
    xp = F.pad(x, (0, 0, P, P))
    outs = []
    for r0 in range(0, Fout, rows_per_chunk):
        r1 = min(Fout, r0 + rows_per_chunk)
        xi = xp[:, :, r0 * S:(r1 - 1) * S + K, :]
        y = F.conv2d(xi, layer.conv.weight, layer.conv.bias, stride=(S, 1))
        y = F.gelu(y)
        y = _rows_dconv(layer, y)
        z = F.glu(layer.rewrite(y), dim=1) if layer.rewrite is not None else y
        if post is not None:
            z = post(z, r0, r1)
        outs.append(z)
    return torch.cat(outs, dim=2)


def dec_freq_chunked(layer, x, skip, rows_per_chunk, frames_per_chunk, post=None):
    """HDecLayer.forward (frequency branch) in bands of rows (rewrite conv + DConv) and then
    in blocks of frames (transposed conv along frequency). Exact."""
    _check_identity_norms(layer)
    assert not layer.empty and layer.rewrite is not None
    B, C, Fq, T = x.shape
    rw = layer.rewrite
    kf, kt = rw.kernel_size
    pf, pt = rw.padding
    ys = []
    for r0 in range(0, Fq, rows_per_chunk):
        r1 = min(Fq, r0 + rows_per_chunk)
        a, b = max(r0 - pf, 0), min(r1 + pf, Fq)
        xi = x[:, :, a:b, :] + skip[:, :, a:b, :]
        xi = F.pad(xi, (pt, pt, pf - (r0 - a), pf - (b - r1)))
        y = F.glu(F.conv2d(xi, rw.weight, rw.bias), dim=1)
        ys.append(_rows_dconv(layer, y))
    y = torch.cat(ys, dim=2)
    outs = []
    P = layer.pad
    for t0 in range(0, T, frames_per_chunk):
        t1 = min(T, t0 + frames_per_chunk)
        z = layer.conv_tr(y[..., t0:t1])
        if P:
            z = z[..., P:-P, :]
        if not layer.last:
            z = F.gelu(z)
        if post is not None:
            z = post(z)
        outs.append(z)
    return torch.cat(outs, dim=-1), y


class EncLayer(torch.nn.Module):
    """Encoder layer idx (frequency branch + paired time branch). Layer 0 also normalises the inputs."""

    def __init__(self, m, idx, rows=0):
        super().__init__()
        self.m, self.idx, self.rows = m, idx, rows

    def forward(self, x, xt):
        m, idx = self.m, self.idx
        if idx == 0:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) / (1e-5 + std)
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)
        inject = None
        if idx < len(m.tencoder):
            tenc = m.tencoder[idx]
            xt = tenc(xt)
            assert not tenc.empty, 'merged time/frequency branches are not supported here'
        emb = None
        if idx == 0 and m.freq_emb is not None:
            frs = torch.arange((x.shape[-2] + 2 * m.encoder[0].pad - m.encoder[0].kernel_size) // m.encoder[0].stride + 1)
            emb = m.freq_emb_scale * m.freq_emb(frs).t()[None, :, :, None]
        if self.rows and inject is None:
            post = (lambda z, r0, r1: z + emb[:, :, r0:r1]) if emb is not None else None
            x = enc_freq_chunked(m.encoder[idx], x, self.rows, post)
        else:
            x = m.encoder[idx](x, inject)
            if emb is not None:
                x = x + emb
        if idx == 0:
            return x, xt, mean, std, meant, stdt
        return x, xt


class EncFreq(torch.nn.Module):
    """Frequency branch of encoder layer idx (layer 0 also normalises its input)."""

    def __init__(self, m, idx, rows=0):
        super().__init__()
        self.m, self.idx, self.rows = m, idx, rows

    def forward(self, x):
        m, idx = self.m, self.idx
        if idx == 0:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) / (1e-5 + std)
        emb = None
        if idx == 0 and m.freq_emb is not None:
            frs = torch.arange((x.shape[-2] + 2 * m.encoder[0].pad - m.encoder[0].kernel_size) // m.encoder[0].stride + 1)
            emb = m.freq_emb_scale * m.freq_emb(frs).t()[None, :, :, None]
        if self.rows:
            post = (lambda z, r0, r1: z + emb[:, :, r0:r1]) if emb is not None else None
            x = enc_freq_chunked(m.encoder[idx], x, self.rows, post)
        else:
            x = m.encoder[idx](x, None)
            if emb is not None:
                x = x + emb
        return (x, mean, std) if idx == 0 else x


class EncTime(torch.nn.Module):
    """Time branch of encoder layer idx (layer 0 also normalises its input)."""

    def __init__(self, m, idx):
        super().__init__()
        self.m, self.idx = m, idx

    def forward(self, xt):
        m, idx = self.m, self.idx
        if idx == 0:
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)
        assert not m.tencoder[idx].empty
        xt = m.tencoder[idx](xt)
        return (xt, meant, stdt) if idx == 0 else xt


class DecFreq(torch.nn.Module):
    """Frequency branch of decoder layer idx; the last one also rescales and picks sources."""

    def __init__(self, m, idx, length, last, fq, rows=0, frames=0):
        super().__init__()
        self.m, self.idx, self.length, self.last, self.fq = m, idx, length, last, fq
        self.rows, self.frames = rows, frames

    def forward(self, x, skip, *rest):
        m = self.m
        post = None
        if self.last:
            mean, std, sources = rest
            S = len(m.sources)
            post = lambda z: torch.index_select(z.view(z.shape[0], S, -1, self.fq, z.shape[-1]), 1, sources) * std[:, None] + mean[:, None]  # noqa: E731
        if self.rows:
            x, _ = dec_freq_chunked(m.decoder[self.idx], x, skip, self.rows, self.frames or x.shape[-1], post)
        else:
            x, _ = m.decoder[self.idx](x, skip, self.length)
            if post is not None:
                x = post(x)
        return x


class DecTime(torch.nn.Module):
    """Time branch of decoder layer idx; the last one also rescales and picks sources."""

    def __init__(self, m, idx, length_t, last):
        super().__init__()
        self.m, self.idx, self.length_t, self.last = m, idx, length_t, last

    def forward(self, xt, skip_t, *rest):
        m = self.m
        tdec = m.tdecoder[self.idx - (m.depth - len(m.tdecoder))]
        assert not tdec.empty
        xt, _ = tdec(xt, skip_t, self.length_t)
        if self.last:
            meant, stdt, sources = rest
            S = len(m.sources)
            xt = torch.index_select(xt.view(xt.shape[0], S, -1, xt.shape[-1]), 1, sources) * stdt[:, None] + meant[:, None]
        return xt


class Tr(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, xt):
        return self.m.crosstransformer(x, xt)


class Dec(torch.nn.Module):
    """Decoder layer idx (frequency branch, plus the paired time-branch layer if any)."""

    def __init__(self, m, idx, length, length_t, uses_skip_t, last, fq, rows=0, frames=0):
        super().__init__()
        self.m, self.idx, self.length, self.length_t = m, idx, length, length_t
        self.uses_skip_t, self.last, self.fq = uses_skip_t, last, fq
        self.rows, self.frames = rows, frames

    def forward(self, x, skip, xt, *rest):
        m, idx = self.m, self.idx
        rest = list(rest)
        post = None
        if self.last:
            mean, std, sources = rest[-5], rest[-4], rest[-1]
            S = len(m.sources)
            # only the requested sources leave the model (less memory to hand back)
            post = lambda z: torch.index_select(z.view(z.shape[0], S, -1, self.fq, z.shape[-1]), 1, sources) * std[:, None] + mean[:, None]  # noqa: E731
        if self.rows:
            x, pre = dec_freq_chunked(m.decoder[idx], x, skip, self.rows, self.frames or x.shape[-1], post)
        else:
            x, pre = m.decoder[idx](x, skip, self.length)
            if post is not None:
                x = post(x)
        offset = m.depth - len(m.tdecoder)
        if idx >= offset:
            tdec = m.tdecoder[idx - offset]
            if tdec.empty:
                pre = pre[:, :, 0]
                xt, _ = tdec(pre, None, self.length_t)
            else:
                skip_t = rest.pop(0)
                xt, _ = tdec(xt, skip_t, self.length_t)
        if not self.last:
            return x, xt
        mean, std, meant, stdt, sources = rest
        S = len(m.sources)
        xt = torch.index_select(xt.view(xt.shape[0], S, -1, xt.shape[-1]), 1, sources)
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


def export(module, args, in_names, out_names, path, dynamic_axes=None):
    torch.onnx.export(module, args, path, opset_version=17, input_names=in_names, output_names=out_names,
                      dynamo=False, do_constant_folding=True, dynamic_axes=dynamic_axes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segment', type=int, default=343980)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', action='store_true', help='compare the chained parts with the single-graph Core')
    ap.add_argument('--rows', type=int, default=64, help='frequency rows per block in the outer layers (0 = no blocking)')
    ap.add_argument('--frames', type=int, default=48, help='frames per block for the outer transposed convs')
    ap.add_argument('--min-rows', type=int, default=128, help='only block layers with at least this many frequency rows')
    ap.add_argument('--split-layers', type=int, default=2, help='outer layers whose frequency and time branches run as separate parts')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.nn.MultiheadAttention.forward = ed.chunked_mha_forward
    model = ed.load_model('models/5c90dfd2.json', 'models/htdemucs_6s.safetensors')
    SEG = a.segment
    FR = -(-SEG // 1024)
    torch.manual_seed(0)
    mag = torch.randn(1, 4, 2048, FR) * 0.1
    mix = torch.randn(1, 2, SEG) * 0.1

    parts = []
    saved, saved_t = [], []
    x, xt = mag, mix
    for idx in range(len(model.encoder)):
        fout = x.shape[2] // model.encoder[idx].stride
        rows = a.rows if a.rows and fout >= a.min_rows else 0
        if idx < a.split_layers:
            # frequency and time branches as two parts: each needs its working memory on its own
            ef, et = EncFreq(model, idx, rows).eval(), EncTime(model, idx).eval()
            rf, rt = ef(x), et(xt)
            if idx == 0:
                export(ef, (x,), ['x'], ['x_next', 'mean', 'std'], f'{a.out}/enc{idx}f.onnx')
                parts.append({'file': f'enc{idx}f', 'in': {'x': 'mag'}, 'out': {'x_next': 's0', 'mean': 'mean', 'std': 'std'}})
                export(et, (xt,), ['xt'], ['xt_next', 'meant', 'stdt'], f'{a.out}/enc{idx}t.onnx')
                parts.append({'file': f'enc{idx}t', 'in': {'xt': 'mix'}, 'out': {'xt_next': 't0', 'meant': 'meant', 'stdt': 'stdt'}})
                (x, mean, std), (xt, meant, stdt) = rf, rt
            else:
                export(ef, (x,), ['x'], ['x_next'], f'{a.out}/enc{idx}f.onnx')
                parts.append({'file': f'enc{idx}f', 'in': {'x': f's{idx - 1}'}, 'out': {'x_next': f's{idx}'}})
                export(et, (xt,), ['xt'], ['xt_next'], f'{a.out}/enc{idx}t.onnx')
                parts.append({'file': f'enc{idx}t', 'in': {'xt': f't{idx - 1}'}, 'out': {'xt_next': f't{idx}'}})
                x, xt = rf, rt
            saved.append(x)
            saved_t.append(xt)
            print('exported enc', idx, '(two parts)', flush=True)
            continue
        enc = EncLayer(model, idx, rows).eval()
        res = enc(x, xt)
        if idx == 0:
            in_w = {'x': 'mag', 'xt': 'mix'}
            out_names = ['x_next', 'xt_next', 'mean', 'std', 'meant', 'stdt']
            out_w = {'x_next': 's0', 'xt_next': 't0', 'mean': 'mean', 'std': 'std', 'meant': 'meant', 'stdt': 'stdt'}
            mean, std, meant, stdt = res[2:]
        else:
            in_w = {'x': f's{idx - 1}', 'xt': f't{idx - 1}'}
            out_names = ['x_next', 'xt_next']
            out_w = {'x_next': f's{idx}', 'xt_next': f't{idx}'}
        export(enc, (x, xt), ['x', 'xt'], out_names, f'{a.out}/enc{idx}.onnx')
        parts.append({'file': f'enc{idx}', 'in': in_w, 'out': out_w})
        x, xt = res[0], res[1]
        saved.append(x)
        saved_t.append(xt)
        print('exported enc', idx, flush=True)
    n_saved, n_saved_t = len(saved), len(saved_t)

    # lengths the decoder layers crop to (recorded the same way HTDemucs.forward does)
    lengths, lengths_t = [], []
    xx, xxt = (mag - mean) / (1e-5 + std), (mix - meant) / (1e-5 + stdt)
    for idx, encode in enumerate(model.encoder):
        lengths.append(xx.shape[-1])
        inject = None
        if idx < len(model.tencoder):
            lengths_t.append(xxt.shape[-1])
            tenc = model.tencoder[idx]
            xxt = tenc(xxt)
            if tenc.empty:
                inject = xxt
        xx = encode(xx, inject)

    tr = Tr(model).eval()
    export(tr, (x, xt), ['x', 'xt'], ['x_next', 'xt_next'], f'{a.out}/tr.onnx')
    x, xt = tr(x, xt)
    parts.append({'file': 'tr', 'in': {'x': f's{n_saved - 1}', 'xt': f't{n_saved_t - 1}'}, 'out': {'x_next': 'x', 'xt_next': 'xt'}})
    print('exported tr', flush=True)

    saved_names = [f's{i}' for i in range(n_saved)]
    saved_t_names = [f't{i}' for i in range(n_saved_t)]
    offset = model.depth - len(model.tdecoder)
    for idx in range(len(model.decoder)):
        skip = saved.pop(-1)
        skip_name = saved_names.pop(-1)
        length = lengths.pop(-1)
        length_t = None
        uses_skip_t = False
        skip_t = skip_t_name = None
        if idx >= offset:
            length_t = lengths_t.pop(-1)
            if not model.tdecoder[idx - offset].empty:
                uses_skip_t = True
                skip_t = saved_t.pop(-1)
                skip_t_name = saved_t_names.pop(-1)
        last = idx == len(model.decoder) - 1
        blocked = a.rows and x.shape[2] >= a.min_rows
        if idx >= len(model.decoder) - a.split_layers and uses_skip_t:
            df = DecFreq(model, idx, length, last, mag.shape[2], a.rows if blocked else 0, a.frames if blocked else 0).eval()
            dt = DecTime(model, idx, length_t, last).eval()
            srcs = torch.arange(len(model.sources))
            if last:
                fa, ta = (x, skip, mean, std, srcs), (xt, skip_t, meant, stdt, srcs)
                export(df, fa, ['x', 'skip', 'mean', 'std', 'sources'], ['spec'], f'{a.out}/dec{idx}f.onnx', {'sources': {0: 'k'}, 'spec': {1: 'k'}})
                parts.append({'file': f'dec{idx}f', 'in': {'x': 'x', 'skip': skip_name, 'mean': 'mean', 'std': 'std', 'sources': 'sources'}, 'out': {'spec': 'spec'}})
                export(dt, ta, ['xt', 'skip_t', 'meant', 'stdt', 'sources'], ['wave'], f'{a.out}/dec{idx}t.onnx', {'sources': {0: 'k'}, 'wave': {1: 'k'}})
                parts.append({'file': f'dec{idx}t', 'in': {'xt': 'xt', 'skip_t': skip_t_name, 'meant': 'meant', 'stdt': 'stdt', 'sources': 'sources'}, 'out': {'wave': 'wave'}})
            else:
                fa, ta = (x, skip), (xt, skip_t)
                export(df, fa, ['x', 'skip'], ['x_next'], f'{a.out}/dec{idx}f.onnx')
                parts.append({'file': f'dec{idx}f', 'in': {'x': 'x', 'skip': skip_name}, 'out': {'x_next': 'x'}})
                export(dt, ta, ['xt', 'skip_t'], ['xt_next'], f'{a.out}/dec{idx}t.onnx')
                parts.append({'file': f'dec{idx}t', 'in': {'xt': 'xt', 'skip_t': skip_t_name}, 'out': {'xt_next': 'xt'}})
            x, xt = df(*fa), dt(*ta)
            print('exported dec', idx, '(two parts)', flush=True)
            continue
        dec = Dec(model, idx, length, length_t, uses_skip_t, last, mag.shape[2],
                  a.rows if blocked else 0, a.frames if blocked else 0).eval()
        args = [x, skip, xt]
        in_names = ['x', 'skip', 'xt']
        wiring = {'x': 'x', 'skip': skip_name, 'xt': 'xt'}
        if uses_skip_t:
            args.append(skip_t)
            in_names.append('skip_t')
            wiring['skip_t'] = skip_t_name
        dyn = None
        if last:
            args += [mean, std, meant, stdt, torch.arange(len(model.sources))]
            in_names += ['mean', 'std', 'meant', 'stdt', 'sources']
            wiring.update({'mean': 'mean', 'std': 'std', 'meant': 'meant', 'stdt': 'stdt', 'sources': 'sources'})
            out_names, out_wiring = ['spec', 'wave'], {'spec': 'spec', 'wave': 'wave'}
            dyn = {'sources': {0: 'k'}, 'spec': {1: 'k'}, 'wave': {1: 'k'}}
        else:
            out_names, out_wiring = ['x_next', 'xt_next'], {'x_next': 'x', 'xt_next': 'xt'}
        res = dec(*args)
        export(dec, tuple(args), in_names, out_names, f'{a.out}/dec{idx}.onnx', dyn)
        parts.append({'file': f'dec{idx}', 'in': wiring, 'out': out_wiring})
        x, xt = res
        print('exported dec', idx, flush=True)

    json.dump({'segment': SEG, 'frames': FR, 'sources': list(model.sources), 'select': True, 'parts': parts},
              open(f'{a.out}/chain.json', 'w'), indent=1)
    if a.check:
        core = ed.Core(model).eval()
        spec, wave = core(mag, mix)
        print('chain vs core: spec max err', float((spec - x).abs().max()), 'wave max err', float((wave - xt).abs().max()))


if __name__ == '__main__':
    main()
