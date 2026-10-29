"""Train the note technique classifier used by engine/techniques.js: plain, bend, vibrato,
natural/pinch harmonic, palm mute, from engine/techfeat.js features of the notes the note model
finds (tools/tech_data.mjs: EG-IPT's single gestures, Guitar-TECHS's technique takes).

usage:
  python tools/train_tech.py cv <data.json> [...] [--model lr|mlp|gb]        # train without one group (player / pickup), test on it
  python tools/train_tech.py test <train.json> [...] --test <test.json> [...] [--model mlp]
  python tools/train_tech.py fit <out.json> <data.json> [...]                  # weights for the app (small MLP)
  python tools/train_tech.py pack <model.json> engine/tech-model.js
options: --drop feature,feature (leave features out)
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

CLASSES = ['plain', 'bend', 'vibrato', 'harmonic', 'palmMute']
ALIAS = {'pinch': 'harmonic'}
HIDDEN = (32, 16)
DROP = []


def arg(k, d=None):
    if k in sys.argv:
        i = sys.argv.index(k)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return d


def load(paths):
    rows, feats = [], None
    for p in paths:
        d = json.load(open(p))
        feats = d['features']
        rows += d['rows']
    keep = [i for i, f in enumerate(feats) if f not in DROP]
    feats = [feats[i] for i in keep]
    X = np.array([[r[4 + i] for i in keep] for r in rows], np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([CLASSES.index(ALIAS.get(r[3], r[3])) for r in rows])
    group = np.array([r[1] for r in rows])
    src = np.array([str(r[2]).split(':')[0] for r in rows])
    return X, y, group, src, feats


def balance(y):
    cnt = np.bincount(y, minlength=len(CLASSES)).astype(float)
    return (len(y) / (len(CLASSES) * np.maximum(cnt, 1)))[y]


class Model:
    def __init__(self, kind='mlp', seed=0):
        self.kind = kind
        if kind == 'lr':
            self.m = LogisticRegression(max_iter=3000, class_weight='balanced')
        elif kind == 'gb':
            self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=seed)
        else:
            self.m = MLPClassifier(HIDDEN, alpha=1e-3, max_iter=2000, early_stopping=True, n_iter_no_change=30, random_state=seed)

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        if self.kind == 'mlp':  # sklearn's MLP has no class weights: resample to balance
            p = balance(y)
            idx = np.random.default_rng(0).choice(len(y), size=2 * len(y), p=p / p.sum())
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
    # a mark is worth showing when it's right: precision and recall of each mark at a few thresholds
    for i, c in enumerate(CLASSES[1:], 1):
        if not (y == i).any():
            continue
        out = []
        for thr in (0.5, 0.7, 0.85):
            mark = P[:, i] >= thr
            tp = (mark & (y == i)).sum()
            out.append(f'p>={thr}: P {100 * tp / max(1, mark.sum()):5.1f}% R {100 * tp / max(1, (y == i).sum()):5.1f}%')
        print(f'  {c:9s} ' + '   '.join(out))


def main():
    model = arg('--model', 'mlp')
    DROP.extend([f for f in arg('--drop', '').split(',') if f])
    cmd = sys.argv[1]
    if cmd == 'cv':
        X, y, group, src, feats = load(sys.argv[2:])
        for g in sorted(set(group)):
            tr = group != g
            m = Model(model).fit(X[tr], y[tr])
            report(y[~tr], m.proba(X[~tr]), f'== {model}: trained without {g}, tested on {g} ({(~tr).sum()} notes)')
    elif cmd == 'test':
        i = sys.argv.index('--test')
        X, y, *_ = load(sys.argv[2:i])
        Xt, yt, gt, st, _ = load(sys.argv[i + 1:])
        m = Model(model).fit(X, y)
        P = m.proba(Xt)
        report(yt, P, f'== {model}: trained on {len(X)}, tested on {len(Xt)}')
        for s in sorted(set(st)):
            report(yt[st == s], P[st == s], f'  -- {s} only')
    elif cmd == 'fit':
        out = sys.argv[2]
        X, y, group, src, feats = load(sys.argv[3:])
        m = Model('mlp').fit(X, y)
        json.dump({'classes': CLASSES, 'features': feats, 'mean': m.mu.round(6).tolist(), 'scale': m.sd.round(6).tolist(),
                   'layers': [{'W': W.round(5).tolist(), 'b': b.round(5).tolist()} for W, b in zip(m.m.coefs_, m.m.intercepts_)]},
                  open(out, 'w'))
        print('wrote', out)
    elif cmd == 'pack':
        m = json.load(open(sys.argv[2]))
        r = lambda v: [r(x) for x in v] if isinstance(v, list) else float(f'{v:.4g}')
        m = {k: r(v) if k in ('mean', 'scale') else v for k, v in m.items()}
        m['layers'] = [{'W': r(l['W']), 'b': r(l['b'])} for l in m['layers']]
        with open(sys.argv[3], 'w') as f:
            f.write('// Weights of the note technique classifier (engine/techniques.js), written by tools/train_tech.py pack.\n'
                    '// Trained on EG-IPT (Fiorini, Brochec, Borg, Pasini 2025) and Guitar-TECHS (Pedroza et al. 2025), both CC BY 4.0.\n')
            f.write('export const TECH_MODEL = ' + json.dumps(m, separators=(',', ':')) + ';\n')
        print('wrote', sys.argv[3])


if __name__ == '__main__':
    main()
