"""Note decoding of the guitar note model's frame outputs, as the app does it (engine/hrgt.js
hrNotes, then the pitch range of engine/notes.js cleanNotes), and note scoring (onset F1: same
pitch, onset within 50 ms, maximum matching, like mir_eval). Used by tools/train_hrgt.py."""
import numpy as np

FPS = 100
BEGIN = 21  # MIDI pitch of class 0
GUITAR_MIN, GUITAR_MAX = 33, 91


def _binarize(reg, threshold, neighbour):
    """Peaks of one pitch's regression output: (bool mask, sub-frame shift)."""
    T = len(reg)
    on = np.zeros(T, bool)
    shift = np.zeros(T, np.float32)
    cand = np.where(reg > threshold)[0]
    for n in cand:
        if n < neighbour or n >= T - neighbour:
            continue
        mono = True
        for i in range(neighbour):
            if reg[n - i] < reg[n - i - 1] or reg[n + i] < reg[n + i + 1]:
                mono = False
                break
        if not mono:
            continue
        on[n] = True
        v, a, b = reg[n], reg[n - 1], reg[n + 1]
        shift[n] = (b - a) / (v - b) / 2 if a > b else (b - a) / (v - a) / 2
    return on, shift


def decode(act, onset_thr=0.4, offset_thr=0.3, frame_thr=0.1):
    """act: dict of (T, 88) arrays onset/offset/frame/velocity -> [(t0, t1, pitch, velocity)]"""
    onset, offset, frame, vel = act['onset'], act['offset'], act['frame'], act['velocity']
    T, C = onset.shape
    notes = []
    for k in range(C):
        p = k + BEGIN
        if p < GUITAR_MIN or p > GUITAR_MAX or onset[:, k].max() <= onset_thr:
            continue
        on, on_s = _binarize(onset[:, k], onset_thr, 2)
        off, off_s = _binarize(offset[:, k], offset_thr, 4)
        fr = frame[:, k]
        bgn, gone, off_at = -1, -1, -1

        def emit(fin, s):
            notes.append(((bgn + on_s[bgn]) / FPS, (fin + s) / FPS, p, float(vel[bgn, k])))

        for i in range(T):
            if on[i]:
                if bgn >= 0:
                    emit(max(i - 1, 0), 0.0)
                    gone, off_at = -1, -1
                bgn = i
            if bgn >= 0 and i > bgn:
                if fr[i] <= frame_thr and gone < 0:
                    gone = i
                if off[i] and off_at < 0:
                    off_at = i
                if gone >= 0:
                    fin = off_at if off_at >= 0 and off_at - bgn > gone - off_at else gone
                    emit(fin, float(off_s[fin]))
                    bgn, gone, off_at = -1, -1, -1
                if bgn >= 0 and (i - bgn >= 600 or i == T - 1):
                    emit(i, float(off_s[i]))
                    bgn, gone, off_at = -1, -1, -1
    notes = [n for n in notes if n[1] > n[0]]
    notes.sort()
    return notes


def match(est, ref, tol=0.05):
    """Maximum matching of onsets per pitch. est/ref: [(t0, ..., pitch at index 2)].
    Returns (number matched, list of (est t0 - ref t0) for the matched pairs)."""
    by = {}
    for i, r in enumerate(ref):
        by.setdefault(r[2], ([], []))[0].append(i)
    for j, e in enumerate(est):
        if e[2] in by:
            by[e[2]][1].append(j)
    tp, diffs = 0, []
    for ri, ej in by.values():
        adj = [[j for j in ej if abs(est[j][0] - ref[i][0]) <= tol] for i in ri]
        owner = {}

        def aug(k, seen):
            for j in adj[k]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in owner or aug(owner[j], seen):
                    owner[j] = k
                    return True
            return False

        for k in range(len(ri)):
            aug(k, set())
        tp += len(owner)
        diffs += [est[j][0] - ref[ri[k]][0] for j, k in owner.items()]
    return tp, diffs


def f1(tp, n_est, n_ref):
    p, r = tp / max(1, n_est), tp / max(1, n_ref)
    return 2 * p * r / (p + r) if p + r else 0.0, p, r
