"""Cache basic-pitch model outputs for GuitarSet and sweep note-creation parameters.
usage: python eval_bp_gs.py cache  [every]
       python eval_bp_gs.py sweep  [every]
"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, '/home/claude/work/basic-pitch-src')
import librosa, mir_eval
from basic_pitch import inference, note_creation as nc

AUD = '/home/claude/work/data/guitarset/audio'
ANN = json.load(open('/home/claude/work/data/gs/guitarset_annotations.json'))
CACHE = '/home/claude/work/data/gs/bpcache'
os.makedirs(CACHE, exist_ok=True)
mode = sys.argv[1]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 3
names = sorted(ANN.keys())[::every]

def model_out(name):
    p = f'{CACHE}/{name}.npz'
    if os.path.exists(p):
        d = np.load(p)
        return {k: d[k] for k in d.files}
    y, _ = librosa.load(f'{AUD}/{name}_mic.wav', sr=22050, mono=True)
    model = model_out.model
    overlap_len = 30 * 256
    hop = 43844 - overlap_len
    audio = np.concatenate([np.zeros(overlap_len // 2, np.float32), y])
    out = {'note': [], 'onset': [], 'contour': []}
    for w, _ in inference.window_audio_file(audio, hop):
        for k, v in model.predict(w[None].astype(np.float32)).items():
            out[k].append(v)
    un = {k: inference.unwrap_output(np.concatenate(out[k]), len(y), 30, hop) for k in out}
    np.savez_compressed(p, **un)
    return un

def frame_time(f):
    w = np.floor(f / 142)
    return (f * 256 - w * (142 * 256 - 36164)) / 22050

def evaluate(params, subset=None):
    on_f, onoff_f, P, R = [], [], [], []
    for name in (subset or names):
        un = model_out(name)
        notes = nc.output_to_notes_polyphonic(un['note'], un['onset'], onset_thresh=params['on'], frame_thresh=params['fr'],
                                              min_note_len=params['ml'], infer_onsets=params['inf'], max_freq=None, min_freq=None,
                                              melodia_trick=params['mel'], energy_tol=params.get('et', 11))
        if not notes:
            on_f.append(0); continue
        est_int = np.array([[frame_time(a), frame_time(b)] for a, b, p, amp in notes])
        est_p = np.array([librosa.midi_to_hz(p) for a, b, p, amp in notes])
        ref = ANN[name]['notes']
        ref_int = np.array([[t, t + d] for t, d, v, s in ref])
        ref_p = np.array([librosa.midi_to_hz(v) for t, d, v, s in ref])
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_int, ref_p, est_int, est_p, offset_ratio=None)
        on_f.append(f); P.append(p); R.append(r)
        if params.get('offsets'):
            _, _, f2, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_int, ref_p, est_int, est_p)
            onoff_f.append(f2)
    return float(np.mean(on_f)), float(np.mean(P)), float(np.mean(R)), (float(np.mean(onoff_f)) if onoff_f else None)

if mode == 'cache':
    model_out.model = inference.Model('/home/claude/work/models/basic_pitch.onnx')
    for i, n in enumerate(names):
        model_out(n)
        if i % 20 == 0: print('cached', i, n, flush=True)
    print('done')
else:
    base = {'on': 0.5, 'fr': 0.3, 'ml': 11, 'inf': True, 'mel': True}
    print('default', evaluate({**base, 'offsets': True}), flush=True)
    best = dict(base); bf = evaluate(best)[0]
    grid = {'on': [0.3, 0.4, 0.5, 0.6, 0.7], 'fr': [0.2, 0.25, 0.3, 0.4, 0.5], 'ml': [3, 5, 7, 9, 11], 'inf': [True, False], 'mel': [True, False], 'et': [5, 8, 11, 15]}
    for rnd in range(2):
        for k, vals in grid.items():
            for v in vals:
                if best.get(k) == v: continue
                c = {**best, k: v}
                f = evaluate(c)[0]
                if f > bf + 1e-4:
                    best, bf = c, f
                    print(f'  {k}={v} -> F1 {f:.4f}', flush=True)
        print('round', rnd, best, bf, flush=True)
    print('best with offsets', evaluate({**best, 'offsets': True}))
