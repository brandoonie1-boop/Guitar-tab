"""Steel-string acoustic notes for the tab player, one per string and fret, from GuitarSet.

GuitarSet recorded each string through a hexaphonic pickup and the whole guitar with a microphone at
the same time. A filter per string, fitted over many takes, turns that string's pickup signal into
what the microphone heard of it (the body, the room, the mic). Each single note picked from a
string's pickup track and run through its filter is then that note as the microphone would have
heard it on its own.

usage: python tools/guitarset_samples.py <GuitarSet dir with annotation/, audio_mono-mic/,
         audio_hex-pickup_debleeded/> <out dir> [--fit-takes 80]
writes <out>/s<string>_f<fret>.f32 (mono float32, 48 kHz) and <out>/index.json
"""
import glob, json, os, subprocess, sys
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
FIT = int(sys.argv[sys.argv.index('--fit-takes') + 1]) if '--fit-takes' in sys.argv else 80
SR = 44100
OPEN = [40, 45, 50, 55, 59, 64]
N, HOP = 4096, 2048
os.makedirs(OUT, exist_ok=True)


def load(p, channels):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-f', 'f32le', '-ac', str(channels), '-ar', str(SR), '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels).T.astype(np.float64)


def lag_of(X, y):
    """How many samples the mic lags the pickups (they differ per take, by up to ~15 ms):
    GCC-PHAT on 5 s windows, the median."""
    hs = X.sum(0)
    n = min(len(hs), len(y))
    W = 5 * SR
    lags = []
    for i in range(0, n - W, W):
        A = np.fft.rfft(hs[i:i + W], 2 * W); Bm = np.fft.rfft(y[i:i + W], 2 * W)
        R = Bm * np.conj(A)
        c = np.fft.irfft(R / (np.abs(R) + 1e-9))
        c = np.concatenate([c[-2000:], c[:2000]])
        lags.append(int(np.argmax(c)) - 2000)
    return int(np.median(lags)) if lags else 0


def aligned(t):
    """pickups and mic of a take, the mic shifted onto the pickups' timing (plus 2 ms so the
    fitted filters stay causal)"""
    X = load(hexp(t), 6); y = load(micp(t), 1)[0]
    lag = lag_of(X, y) - int(0.002 * SR)
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = np.concatenate([np.zeros(-lag), y])
    n = min(X.shape[1], len(y))
    return X[:, :n], y[:n]


takes = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(SRC, 'annotation', '*.jams')))
hexp = lambda t: os.path.join(SRC, 'audio_hex-pickup_debleeded', t + '_hex_cln.wav')
micp = lambda t: os.path.join(SRC, 'audio_mono-mic', t + '_mic.wav')
takes = [t for t in takes if os.path.exists(hexp(t)) and os.path.exists(micp(t))]
print('takes with both recordings', len(takes))

# ---- 1. one filter per string: mic(f) = sum_s H_s(f) hex_s(f), least squares over STFT frames
rng = np.random.default_rng(1)
order = rng.permutation(len(takes))
fit, held = [takes[i] for i in order[:FIT]], [takes[i] for i in order[FIT: FIT + 10]]
w = np.hanning(N)
B = N // 2 + 1
A = np.zeros((B, 6, 6), complex)
b = np.zeros((B, 6), complex)
for t in fit:
    X, y = aligned(t)
    n = len(y)
    for i in range(0, n - N, HOP):
        F = np.fft.rfft(X[:, i:i + N] * w, axis=1)  # (6, B)
        Y = np.fft.rfft(y[i:i + N] * w)
        A += np.einsum('sb,tb->bst', np.conj(F), F)
        b += (np.conj(F) * Y).T
lam = 1e-6 * np.trace(A, axis1=1, axis2=2).real.mean() / 6
H = np.linalg.solve(A + lam * np.eye(6)[None], b[..., None])[..., 0]  # (B, 6)
h = np.fft.irfft(H.T, n=N, axis=1)  # (6, N) impulse responses
h = np.roll(h, 256, axis=1)[:, :3072]  # a little pre-ringing kept, 70 ms
h *= np.concatenate([np.hanning(512)[:256], np.ones(3072 - 256 - 1024), np.hanning(2048)[1024:]])[None]


def predict(X):
    return sum(np.convolve(X[s], h[s])[256: 256 + X.shape[1]] for s in range(6))


snrs = []
for t in held:
    X, y = aligned(t)
    n = len(y)
    p = predict(X[:, :n])
    g = (p @ y[:n]) / (p @ p)
    snrs.append(10 * np.log10((y[:n] ** 2).sum() / ((y[:n] - g * p) ** 2).sum()))
print('mic predicted from the pickups on unseen takes: SNR', np.round(snrs, 1), 'mean', round(float(np.mean(snrs)), 1), 'dB')
np.save(os.path.join(OUT, 'string_filters.npy'), h)

# ---- 2. the best single note per string and fret
cands = {}
for t in takes:
    d = json.load(open(os.path.join(SRC, 'annotation', t + '.jams')))
    per = {}
    for a in d['annotations']:
        if a['namespace'] == 'note_midi':
            s = int(a['annotation_metadata']['data_source'])
            per[s] = sorted((n['time'], n['duration'], n['value']) for n in a['data'])
    for s, ns in per.items():
        for i, (t0, du, v) in enumerate(ns):
            nxt = ns[i + 1][0] if i + 1 < len(ns) else t0 + 6
            room = nxt - t0  # how long the string rings before it is played again
            fret = int(round(v)) - OPEN[s]
            if not 0 <= fret <= 20 or abs(v - round(v)) > 0.25 or room < 0.6:
                continue
            key = (s, fret)
            score = min(room, 3.0) + 0.3 * min(du, 2.0)
            if key not in cands or score > cands[key][0]:
                cands[key] = (score, t, t0, room, v)
print('string/fret positions found', len(cands))

index = []
cache = {}
for (s, fret), (score, t, t0, room, v) in sorted(cands.items()):
    if t not in cache:
        cache.clear()
        cache[t] = load(hexp(t), 6)
    X = cache[t]
    a = max(0, int((t0 - 0.03) * SR))
    L = min(int(min(room + 0.03, 3.0) * SR), X.shape[1] - a)
    if L < int(0.5 * SR):
        continue
    y = np.convolve(X[s, a: a + L], h[s])[256: 256 + L]
    # the note is cut where the string was played again: fade there
    f = int(0.06 * SR)
    y[-f:] *= np.linspace(1, 0, f)
    out = np.zeros(int(len(y) * 48000 / SR))
    out = np.interp(np.arange(len(out)) * SR / 48000, np.arange(len(y)), y).astype(np.float32)
    name = f's{s}_f{fret}.f32'
    out.tofile(os.path.join(OUT, name))
    index.append({'s': s, 'f': fret, 'file': name, 'len': round(len(out) / 48000, 3), 'cents': round(100 * (v - round(v)), 1), 'take': t})
json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=0)
print('notes written', len(index), 'mean length', round(float(np.mean([r['len'] for r in index])), 2), 's')
