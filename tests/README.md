# Tests and evaluation scripts

Development scripts used to check and tune the engine. They expect a local setup:

- `npm install onnxruntime-node onnxruntime-web playwright-core` (Node 22)
- GuitarSet audio and annotations (https://zenodo.org/records/3371780), converted with the snippet in `tools/eval_bp_gs.py`
- paths at the top of each script point at the original development machine; adjust them.

| Script | What it checks |
|---|---|
| `test_demucs.mjs` | JS STFT/iSTFT + ONNX Demucs vs. the PyTorch reference (should be ~80 dB SNR) |
| `test_hrgt.mjs` | guitar transcription model: JS log-mel, ONNX parts and note decoding vs. PyTorch (reference from `tools/make_hrgt_ref.py`) |
| `eval_hrgt_gs.mjs` | note accuracy of the guitar transcription model on GuitarSet, through the app's JS code |
| `analyze_note_errors.mjs` | note errors by kind (missed re-attacks, octave/harmonic ghosts, ...) for basic-pitch or (`--hr`) the transcription model |
| `test_basicpitch.mjs` | JS basic-pitch runner + note decoding vs. the Python package (identical notes) |
| `test_fretboard.mjs`, `synthetic_cases.mjs` | tuning/capo detection and fingering on synthetic tabs (DAEABE capo 7, FACGCE, open D, drop D, capo 2/3, ...) |
| `eval_fingering_gs.mjs` | string/fret accuracy on GuitarSet ground-truth notes |
| `eval_gs_pipeline.mjs` | full pipeline on GuitarSet audio (optionally with drums and bass mixed in) |
| `eval_tempo.mjs` | tempo and beat tracking on GuitarSet |
| `sweep_*.mjs` | parameter searches used to tune the cost weights |
| `helper_e2e.mjs` | the paste-a-link flow through the link helper (`helper/helper.ps1` running), in Chromium |
| `browser_e2e.mjs` | runs the real app in headless Chromium / WebKit (iPhone engine) |
| `run_file.mjs` | transcribe a WAV from the command line and print the tab |
