"""Run basic-pitch's own inference + note creation (ONNX model) on an audio file and dump results."""
import sys, json, numpy as np
sys.path.insert(0, '/home/claude/work/basic-pitch-src')
import librosa
from basic_pitch import inference, note_creation as nc

wav_path, out_prefix = sys.argv[1], sys.argv[2]
y, _ = librosa.load(wav_path, sr=22050, mono=True)
y.astype(np.float32).tofile(out_prefix + '_audio22k.f32')
model = inference.Model('/home/claude/work/models/basic_pitch.onnx')
# replicate run_inference but from array
n_overlapping_frames = inference.DEFAULT_OVERLAPPING_FRAMES
overlap_len = n_overlapping_frames * 256
hop_size = 43844 - overlap_len
original_length = y.shape[0]
audio = np.concatenate([np.zeros((overlap_len // 2,), dtype=np.float32), y])
output = {"note": [], "onset": [], "contour": []}
for window, _ in inference.window_audio_file(audio, hop_size):
    for k, v in model.predict(window[None].astype(np.float32)).items():
        output[k].append(v)
un = {k: inference.unwrap_output(np.concatenate(output[k]), original_length, n_overlapping_frames, hop_size) for k in output}
for k in un:
    un[k].astype(np.float32).tofile(f'{out_prefix}_{k}.f32')
notes = nc.output_to_notes_polyphonic(un['note'], un['onset'], onset_thresh=0.5, frame_thresh=0.3,
                                      min_note_len=11, infer_onsets=True, max_freq=None, min_freq=None,
                                      melodia_trick=True)
notes_pb = nc.get_pitch_bends(un['contour'], notes)
json.dump([[int(a), int(b), int(p), float(amp), [int(x) for x in pb]] for a, b, p, amp, pb in notes_pb],
          open(out_prefix + '_notes.json', 'w'))
print('frames', un['note'].shape, 'notes', len(notes))
