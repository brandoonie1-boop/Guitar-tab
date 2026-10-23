"""Train the string classifier used by engine/strings.js.

Input: feature files written by tools/string_data.mjs: engine/strings.js stringFeatures(), one
row per note, with [take, pitch, string, fret, copy(, tuning)] metadata. The network scores each
place on the neck where a note could have been played (string + fret from the nut); a softmax
over those places gives P(place | sound).

usage:
  python tools/train_strings.py cv    <data prefix> [<data prefix> ...]   # leave-one-player-out check
  python tools/train_strings.py cvfit <folds dir> <data prefix> [...]     # same, keeping fold_<player>.json
  python tools/train_strings.py test  <train prefixes...> --test <test prefixes...>
  python tools/train_strings.py fit   <out.json> <data prefix> [...]       # final model, weights as JSON
  python tools/train_strings.py pack  <model.json> engine/strings-model.js # float16 weights for the app
Each data prefix names <prefix>.f32 (float32 rows) and <prefix>.json
({nfeat, names, meta, tunings?, group?}); takes are grouped by player (first two characters of
the GuitarSet name) unless the file gives a group per take. Rows with copy 0 (the recordings as
they are) are the test rows; jittered and tone-augmented copies only train.
Trains with PyTorch when it is installed (several times faster), otherwise with the NumPy code
below. Environment: EPOCHS (25), HIDDEN (256,128,64), SEED, THREADS, ONLY_GROUP (cv: one fold).
"""
import json
import os
import sys

import numpy as np

STD = [40, 45, 50, 55, 59, 64]
MAXF = 22
H1, H2, H3 = (int(v) for v in os.environ.get('HIDDEN', '256,128,64').split(','))
NC = 8


def cand_enc(s, f):
    c = np.zeros(NC, np.float32)
    c[s] = 1
    c[6] = f / 12
    c[7] = 1.0 if f == 0 else 0.0
    return c


def load(prefixes):
    X, rows = [], []
    for pi, p in enumerate(prefixes):
        meta = json.load(open(p + '.json'))
        x = np.fromfile(p + '.f32', np.float32).reshape(-1, meta['nfeat'])
        names = meta['names']
        tunings = meta.get('tunings')
        for m in meta['meta']:
            take, pitch, string, fret, copy = m[:5]
            tun = tunings[m[5]] if tunings and len(m) > 5 else STD
            group = meta['group'][take] if 'group' in meta else names[take][:2]
            rows.append((pi, names[take], group, pitch, string, fret, copy, tun))
        X.append(x)
    return np.concatenate(X), rows


def build(rows):
    """candidate encodings C [N, 6, NC], mask M [N, 6], target y [N]"""
    N = len(rows)
    C = np.zeros((N, 6, NC), np.float32)
    M = np.zeros((N, 6), bool)
    y = np.zeros(N, np.int64)
    for i, (pi, name, group, pitch, string, fret, copy, tun) in enumerate(rows):
        for s in range(6):
            f = pitch - tun[s]
            if 0 <= f <= MAXF:
                C[i, s] = cand_enc(s, f)
                M[i, s] = True
        y[i] = string
        if not M[i, string]:
            C[i, string] = cand_enc(string, min(MAXF, max(0, pitch - tun[string])))
            M[i, string] = True
    return C, M, y


class Net:
    """x -> relu(W1) -> relu(W2) = h;  score(place) = w4 . relu(W3h h + W3c c + b3) + b4"""

    def __init__(self, d, rng):
        def he(i, o):
            return (rng.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(np.float32)
        self.p = {
            'W1': he(d, H1), 'b1': np.zeros(H1, np.float32),
            'W2': he(H1, H2), 'b2': np.zeros(H2, np.float32),
            'W3h': he(H2, H3), 'W3c': he(NC, H3), 'b3': np.zeros(H3, np.float32),
            'w4': (rng.standard_normal(H3) * np.sqrt(1.0 / H3)).astype(np.float32), 'b4': np.zeros(1, np.float32),
        }

    def forward(self, x, C, M, train=False, rng=None, drop=0.2):
        p = self.p
        a1 = x @ p['W1'] + p['b1']
        h1 = np.maximum(a1, 0)
        if train and drop > 0:
            keep = (rng.random(h1.shape) >= drop).astype(np.float32) / (1 - drop)
            h1 = h1 * keep
        else:
            keep = None
        a2 = h1 @ p['W2'] + p['b2']
        h = np.maximum(a2, 0)
        a3 = (h @ p['W3h'])[:, None, :] + C @ p['W3c'] + p['b3']  # [N, 6, H3]
        z = np.maximum(a3, 0)
        s = z @ p['w4'] + p['b4'][0]  # [N, 6]
        s = np.where(M, s, -1e9)
        cache = (x, a1, h1, keep, a2, h, a3, z, C, M)
        return s, cache

    def backward(self, ds, cache):
        x, a1, h1, keep, a2, h, a3, z, C, M = cache
        p = self.p
        g = {}
        ds = np.where(M, ds, 0)
        g['b4'] = np.array([ds.sum()], np.float32)
        g['w4'] = np.einsum('nk,nkh->h', ds, z)
        dz = ds[:, :, None] * p['w4'][None, None, :]
        da3 = dz * (a3 > 0)
        g['b3'] = da3.sum((0, 1))
        g['W3c'] = np.einsum('nkc,nkh->ch', C, da3)
        dh_sum = da3.sum(1)  # [N, H3]
        g['W3h'] = h.T @ dh_sum
        dh = dh_sum @ p['W3h'].T
        da2 = dh * (a2 > 0)
        g['W2'] = h1.T @ da2
        g['b2'] = da2.sum(0)
        dh1 = da2 @ p['W2'].T
        if keep is not None:
            dh1 = dh1 * keep
        da1 = dh1 * (a1 > 0)
        g['W1'] = x.T @ da1
        g['b1'] = da1.sum(0)
        return g


def log_softmax(s):
    m = s.max(1, keepdims=True)
    e = np.exp(s - m)
    return s - m - np.log(e.sum(1, keepdims=True))


def train(X, C, M, y, epochs=25, seed=0, lr=2e-3, wd=1e-4, bs=512, verbose=False):
    rng = np.random.default_rng(seed)
    net = Net(X.shape[1], rng)
    mom = {k: np.zeros_like(v) for k, v in net.p.items()}
    vel = {k: np.zeros_like(v) for k, v in net.p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(y)
    steps = epochs * ((n + bs - 1) // bs)
    t = 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            s, cache = net.forward(X[idx], C[idx], M[idx], train=True, rng=rng)
            lp = log_softmax(s)
            tot += -lp[np.arange(len(idx)), y[idx]].sum()
            ds = np.exp(lp)
            ds[np.arange(len(idx)), y[idx]] -= 1
            ds /= len(idx)
            g = net.backward(ds, cache)
            t += 1
            # one-cycle-ish schedule: warm up, then cosine down
            frac = t / steps
            cur = lr * (frac / 0.1 if frac < 0.1 else 0.5 * (1 + np.cos(np.pi * (frac - 0.1) / 0.9)))
            for k in net.p:
                gk = g[k] + (wd * net.p[k] if net.p[k].ndim > 1 else 0)
                mom[k] = b1 * mom[k] + (1 - b1) * gk
                vel[k] = b2 * vel[k] + (1 - b2) * gk * gk
                mh = mom[k] / (1 - b1 ** t)
                vh = vel[k] / (1 - b2 ** t)
                net.p[k] -= (cur * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        if verbose:
            print(f'  epoch {ep + 1}: loss {tot / n:.4f}', flush=True)
    return net


def train_torch(X, C, M, y, epochs=25, seed=0, lr=2e-3, wd=1e-4, bs=512, verbose=False, drop=0.2):
    """Same network and schedule as train(), on PyTorch (several times faster); returns a Net."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get('THREADS', 4)))
    net = Net(X.shape[1], np.random.default_rng(seed))
    P = {k: torch.nn.Parameter(torch.from_numpy(v.copy())) for k, v in net.p.items()}
    opt = torch.optim.AdamW([{'params': [v for v in P.values() if v.ndim > 1], 'weight_decay': wd},
                             {'params': [v for v in P.values() if v.ndim <= 1], 'weight_decay': 0.0}], lr=lr)
    n = len(y)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / (0.1 * steps) if t < 0.1 * steps else 0.5 * (1 + np.cos(np.pi * (t - 0.1 * steps) / (0.9 * steps))))
    Xt, Ct, Mt, yt = torch.from_numpy(X), torch.from_numpy(C), torch.from_numpy(M), torch.from_numpy(y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            h1 = torch.relu(Xt[idx] @ P['W1'] + P['b1'])
            h1 = torch.nn.functional.dropout(h1, drop, True)
            h = torch.relu(h1 @ P['W2'] + P['b2'])
            z = torch.relu((h @ P['W3h'])[:, None, :] + Ct[idx] @ P['W3c'] + P['b3'])
            sc = (z @ P['w4'] + P['b4']).masked_fill(~Mt[idx], -1e9)
            loss = torch.nn.functional.cross_entropy(sc, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if verbose:
            print(f'  epoch {ep + 1}: loss {tot / n:.4f}', flush=True)
    for k in net.p:
        net.p[k] = P[k].detach().numpy().astype(np.float32)
    return net


def fit(X, C, M, y, **kw):
    """PyTorch when it is installed (TORCH=0 forces the NumPy version)"""
    if os.environ.get('TORCH', '1') != '0':
        try:
            import torch  # noqa: F401
            return train_torch(X, C, M, y, **kw)
        except ImportError:
            pass
    return train(X, C, M, y, **kw)


def save(net, d, outp):
    json.dump({'nfeat': int(d), 'h': [H1, H2, H3, NC],
               'weights': {k: np.round(v.astype(np.float64), 6).tolist() for k, v in net.p.items()}}, open(outp, 'w'))


def predict(net, X, C, M, bs=4096):
    out = np.zeros((len(X), 6), np.float32)
    for b in range(0, len(X), bs):
        s, _ = net.forward(X[b:b + bs], C[b:b + bs], M[b:b + bs])
        out[b:b + bs] = log_softmax(s)
    return out


def main():
    mode = sys.argv[1]
    if mode in ('cv', 'cvfit'):
        # cvfit <out dir> <prefixes...>: also keeps each held-out group's model (fold_<group>.json)
        outdir = sys.argv[2] if mode == 'cvfit' else None
        prefixes = [a for a in sys.argv[3 if outdir else 2:] if not a.startswith('--')]
        epochs = int(os.environ.get('EPOCHS', 25))
        X, rows = load(prefixes)
        C, M, y = build(rows)
        groups = sorted(set(r[2] for r in rows))
        out = np.zeros((len(y), 6), np.float32)
        test = np.array([r[6] == 0 for r in rows])
        for g in groups:
            if os.environ.get('ONLY_GROUP') and g != os.environ['ONLY_GROUP']:
                continue
            tr = np.array([r[2] != g for r in rows])
            te = ~tr & test
            net = fit(X[tr], C[tr], M[tr], y[tr], epochs=epochs)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                save(net, X.shape[1], os.path.join(outdir, f'fold_{g}.json'))
            lp = predict(net, X[te], C[te], M[te])
            out[te] = lp
            multi = M[te].sum(1) > 1
            acc = lp.argmax(1) == y[te]
            print(f'group {g}: test notes {te.sum()} acc {acc.mean() * 100:.1f}% (notes with 2+ places {acc[multi].mean() * 100:.1f}%, n={multi.sum()})', flush=True)
        acc = out[test].argmax(1) == y[test]
        multi = M[test].sum(1) > 1
        print(f'ALL: acc {acc.mean() * 100:.1f}%  notes with 2+ places {acc[multi].mean() * 100:.1f}%')
        for pi, p in enumerate(prefixes):
            sel = test & np.array([r[0] == pi for r in rows])
            print(f'  {p}: acc {(out[sel].argmax(1) == y[sel]).mean() * 100:.1f}%')
        np.save(prefixes[0] + '.cvlogp.npy', out[test])
        json.dump([[r[0], r[1], r[3], r[4]] for r, t in zip(rows, test) if t], open(prefixes[0] + '.cvrows.json', 'w'))
    elif mode == 'test':
        # train on the prefixes before '--test', report on the ones after it
        i = sys.argv.index('--test')
        trp, tep = sys.argv[2:i], sys.argv[i + 1:]
        epochs = int(os.environ.get('EPOCHS', 25))
        Xa, ra = load(trp)
        Ca, Ma, ya = build(ra)
        net = fit(Xa, Ca, Ma, ya, epochs=epochs, seed=int(os.environ.get('SEED', 0)))
        if os.environ.get('SAVE'):
            save(net, Xa.shape[1], os.environ['SAVE'])
        for p in tep:
            Xb, rb = load([p])
            Cb, Mb, yb = build(rb)
            te = np.array([r[6] == 0 for r in rb])
            lp = predict(net, Xb[te], Cb[te], Mb[te])
            acc = lp.argmax(1) == yb[te]
            multi = Mb[te].sum(1) > 1
            print(f'{p}: acc {acc.mean() * 100:.1f}% (notes with 2+ places {acc[multi].mean() * 100:.1f}%, n={te.sum()})', flush=True)
            np.save(p + os.environ.get('TAG', '') + '.testlogp.npy', lp)
            json.dump([[0, r[1], r[3], r[4]] for r, t in zip(rb, te) if t], open(p + '.testrows.json', 'w'))
    elif mode == 'pack':
        # pack <model.json> <engine/strings-model.js>: float16 weights as base64 in a JS module
        import base64
        w = json.load(open(sys.argv[2]))
        names = ['W1', 'b1', 'W2', 'b2', 'W3h', 'W3c', 'b3', 'w4', 'b4']
        arrs = [np.asarray(w['weights'][k], np.float32).ravel() for k in names]
        data = base64.b64encode(np.concatenate(arrs).astype('<f2').tobytes()).decode()
        shapes = ',\n'.join(f"    ['{k}', {a.size}]" for k, a in zip(names, arrs))
        src = ("// Weights of the string classifier (engine/strings.js), written by tools/train_strings.py pack.\n"
               f"// {os.environ.get('MODEL_NOTE', 'Trained on GuitarSet and Guitar-TECHS (CC BY 4.0).')}\n"
               "export const MODEL = {\n"
               f"  nfeat: {w['nfeat']},\n  h: {json.dumps(w['h'])},\n  shapes: [\n{shapes},\n  ],\n"
               f"  data: '{data}',\n}};\n")
        open(sys.argv[3], 'w').write(src)
        print('wrote', sys.argv[3], len(src) // 1000, 'kB')
    elif mode == 'fit':
        outp = sys.argv[2]
        X, rows = load(sys.argv[3:])
        C, M, y = build(rows)
        net = fit(X, C, M, y, epochs=int(os.environ.get('EPOCHS', 25)), verbose=True)
        save(net, X.shape[1], outp)
        print('saved', outp)


if __name__ == '__main__':
    main()
