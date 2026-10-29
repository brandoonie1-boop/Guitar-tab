"""Train the classifier of how each note began (engine/techniques.js): picked, legato (hammer-on or
pull-off), slide, or the rest of a bend the note detection split; from tools/onset_data.mjs rows.

usage:
  python tools/train_onset.py cv <train.json> [...] [--model lr|mlp|gb]       # train on one pickup, test on the others
  python tools/train_onset.py test <train.json> [...] --test <test.json> [...] [--model mlp]
  python tools/train_onset.py fit <out.json> <train.json> [...]              # weights for the app (small MLP)
  python tools/train_onset.py pack <model.json> engine/onset-model.js        # into the app
options: --drop feature,feature (leave features out)
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

CLASSES = ['pick', 'legato', 'slide', 'bend']
HIDDEN = (32, 16)


def arg(k, d=None):
    if k in sys.argv:
        i = sys.argv.index(k)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return d


DROP = []


def load(paths):
    rows, feats = [], None
    for p in paths:
        d = json.load(open(p))
        feats = d['features']
        rows += d['rows']
    keep = [i for i, f in enumerate(feats) if f not in DROP]
    feats = [feats[i] for i in keep]
    X = np.array([[r[5 + i] for i in keep] for r in rows], np.float64)
    y = np.array([CLASSES.index(r[3]) for r in rows])
    group = np.array([r[1] for r in rows])
    kind = np.array([r[4] for r in rows])
    src = np.array([r[2] for r in rows])
    return X, y, group, kind, src, feats


def balance(y):
    cnt = np.bincount(y, minlength=len(CLASSES)).astype(float)
    return (len(y) / (len(CLASSES) * np.maximum(cnt, 1)))[y]


def make(kind, seed=0):
    if kind == 'lr':
        return LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')
    if kind == 'gb':
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=seed)
    return MLPClassifier(HIDDEN, alpha=1e-3, max_iter=2000, early_stopping=True, n_iter_no_change=30, random_state=seed)


class Model:
    def __init__(self, kind='mlp', seed=0):
        self.kind, self.m = kind, make(kind, seed)

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        if self.kind == 'mlp':  # no class weights in sklearn's MLP: resample to balance
            rng = np.random.default_rng(0)
            p = balance(y)
            idx = rng.choice(len(y), size=2 * len(y), p=p / p.sum())
            self.m.fit(Xs[idx], y[idx])
        else:
            self.m.fit(Xs, y)
        return self

    def proba(self, X):
        P = np.zeros((len(X), len(CLASSES)))
        P[:, self.m.classes_] = self.m.predict_proba((X - self.mu) / self.sd)
        return P


def report(y, P, title):
    pred = P.argmax(1)
    print(title)
    print('            ' + ' '.join(f'{c[:8]:>8s}' for c in CLASSES) + '   recall')
    for i, c in enumerate(CLASSES):
        row = [(pred[y == i] == j).sum() for j in range(len(CLASSES))]
        if sum(row):
            print(f'  {c:9s} ' + ' '.join(f'{v:8d}' for v in row) + f'   {100 * row[i] / sum(row):5.1f}%')
    # what the app needs most: legato marks that are right, and how many legato moves get one
    lg = CLASSES.index('legato')
    for thr in (0.5, 0.7, 0.85):
        mark = P[:, lg] >= thr
        tp = (mark & (y == lg)).sum()
        print(f'  legato at p>={thr}: precision {100 * tp / max(1, mark.sum()):5.1f}%  recall {100 * tp / max(1, (y == lg).sum()):5.1f}%'
              f'  (picks marked: {(mark & (y == 0)).sum()} of {(y == 0).sum()})')


def main():
    model = arg('--model', 'mlp')
    DROP.extend([f for f in arg('--drop', '').split(',') if f])
    cmd = sys.argv[1]
    if cmd == 'cv':
        X, y, group, kind, src, feats = load(sys.argv[2:])
        for g in sorted(set(group)):
            tr = group != g
            m = Model(model).fit(X[tr], y[tr])
            report(y[~tr], m.proba(X[~tr]), f'== {model}: trained without {g}, tested on {g} ({(~tr).sum()} moves)')
    elif cmd == 'test':
        i = sys.argv.index('--test')
        X, y, *_ = load(sys.argv[2:i])
        Xt, yt, gt, kt, st, _ = load(sys.argv[i + 1:])
        m = Model(model).fit(X, y)
        P = m.proba(Xt)
        report(yt, P, f'== {model}: trained on {len(X)}, tested on {len(Xt)}')
        for s in sorted(set(st)):
            report(yt[st == s], P[st == s], f'  -- {s} only')
    elif cmd == 'fit':
        out = sys.argv[2]
        X, y, group, kind, src, feats = load(sys.argv[3:])
        m = Model('mlp').fit(X, y)
        net = m.m
        json.dump({'classes': CLASSES, 'features': feats, 'mean': m.mu.round(6).tolist(), 'scale': m.sd.round(6).tolist(),
                   'layers': [{'W': W.round(5).tolist(), 'b': b.round(5).tolist()} for W, b in zip(net.coefs_, net.intercepts_)]},
                  open(out, 'w'))
        print('wrote', out)
    elif cmd == 'pack':
        m = json.load(open(sys.argv[2]))
        r = lambda v: [r(x) for x in v] if isinstance(v, list) else float(f'{v:.4g}')
        m = {k: r(v) if k in ('mean', 'scale') else v for k, v in m.items()}
        m['layers'] = [{'W': r(l['W']), 'b': r(l['b'])} for l in m['layers']]
        with open(sys.argv[3], 'w') as f:
            f.write('// Weights of the onset classifier (engine/techniques.js), written by tools/train_onset.py pack.\n'
                    '// Trained on EG-IPT (Fiorini, Brochec, Borg, Pasini 2025; CC BY 4.0): real hammer-ons, pull-offs,\n'
                    '// trills, slides and bends, and picked moves made from its plain notes.\n')
            f.write('export const ONSET_MODEL = ' + json.dumps(m, separators=(',', ':')) + ';\n')
        print('wrote', sys.argv[3])


if __name__ == '__main__':
    main()
