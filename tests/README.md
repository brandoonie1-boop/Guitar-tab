# Tests and evaluation scripts

Development scripts used to check and tune the engine. They expect a local setup:

- `npm install onnxruntime-node onnxruntime-web playwright-core` (Node 22)
- GuitarSet audio and annotations (https://zenodo.org/records/3371780); `python tools/guitarset_json.py <annotation folder> guitarset_annotations.json` turns the JAMS files into the one JSON file the scripts read (the same works for EGSet12)
- for the string classifier also Guitar-TECHS (https://zenodo.org/records/14963133) and EGSet12 (https://zenodo.org/records/11406378)
- paths at the top of the older scripts point at the original development machine; adjust them. The newer ones (`eval_strings.mjs`, `cache_activations.mjs`, `tools/string_data.mjs`, `tools/train_strings.py`) take their paths as arguments.

String classifier, from data to numbers:

```
python tools/guitarset_json.py GuitarSet/annotation gs.json          # and EGSet12 the same way
node tools/string_data.mjs guitarset gs.json GuitarSet/audio sf/gs_mic --version mic --copies 2
node tools/string_data.mjs guitarset gs.json GuitarSet/audio sf/gs_mix --version mix --copies 2
node tools/string_data.mjs guitarset gs.json GuitarSet/audio sf/aug_gs_mix --version mix --copies 0 --augment 2
node tools/string_data.mjs guitarset gs.json GuitarSet/audio sf/aug_gs_mic --version mic --copies 0 --augment 1
node tools/string_data.mjs gtechs-lags Guitar-TECHS
node tools/string_data.mjs gtechs Guitar-TECHS sf/gt_micamp --kind micamp
node tools/string_data.mjs gtechs Guitar-TECHS sf/gt_di --kind directinput
node tools/string_data.mjs gtechs Guitar-TECHS sf/aug_gt_di --kind directinput --copies 0 --augment 2 --groups P1,P2
node tools/string_data.mjs gtechs Guitar-TECHS sf/aug_gt_mic --kind micamp --copies 0 --augment 1 --groups P1,P2
python tools/train_strings.py cvfit sf/folds sf/gs_mic sf/gs_mix sf/aug_gs_mix sf/aug_gs_mic sf/gt_micamp sf/gt_di sf/aug_gt_di sf/aug_gt_mic
python tools/train_strings.py fit sf/model.json <the same data>
python tools/train_strings.py pack sf/model.json engine/strings-model.js
node tests/cache_activations.mjs models acts/gs GuitarSet/audio/*_mic.wav
node tests/eval_strings.mjs guitarset gs.json GuitarSet/audio acts/gs sf/folds --detect
node tests/eval_strings.mjs egset egset.json EGSet12 acts/egset sf/model.json --detect
```

| Script | What it checks |
|---|---|
| `test_demucs.mjs` | JS STFT/iSTFT + ONNX Demucs vs. the PyTorch reference (should be ~80 dB SNR) |
| `test_bsroformer.mjs` | BS-RoFormer wrapper (`engine/bsroformer.js`) with a stand-in model that hands the mix back: STFT/iSTFT, chunking, padding and cross-fades must give the input back exactly |
| `test_onnxprune.mjs` | the on-device model cutter (`engine/onnxprune.js`) on a small hand-made model: only what the kept outputs need stays, weights unchanged, the same bytes however the download arrives in pieces |
| `test_guitarsynth.mjs` | the tab player (`app/guitarsynth.js`): note timing from the tab, ties, slides, bends, hammer-ons, loudness |
| `synth_render.mjs` | renders a tab with every tab-player sound to WAV files through the real page (headless Chromium), to listen to |
| `test_hrgt.mjs` | guitar transcription model: JS log-mel, ONNX parts and note decoding vs. PyTorch (reference from `tools/make_hrgt_ref.py`) |
| `eval_hrgt_gs.mjs` | note accuracy of the guitar transcription model on GuitarSet, through the app's JS code |
| `analyze_note_errors.mjs` | note errors by kind (missed re-attacks, octave/harmonic ghosts, ...) for basic-pitch or (`--hr`) the transcription model |
| `test_basicpitch.mjs` | JS basic-pitch runner + note decoding vs. the Python package (identical notes) |
| `test_fretboard.mjs`, `synthetic_cases.mjs` | tuning/capo detection and fingering on synthetic tabs (DAEABE capo 7, FACGCE, open D, drop D, capo 2/3, ...) |
| `eval_fingering_gs.mjs` | string/fret accuracy on GuitarSet ground-truth notes (from the notes alone) |
| `eval_strings.mjs` | tab accuracy after note detection (notes found, right string/fret, notes completely right, tuning/capo), with and without the sound of each note: GuitarSet with every player judged by a string model that never heard them, EGSet12 |
| `cache_activations.mjs` | runs the note model over WAV files and keeps its outputs for `eval_strings.mjs` |
| `eval_gs_pipeline.mjs` | full pipeline on GuitarSet audio (optionally with drums and bass mixed in) |
| `eval_tempo.mjs` | tempo and beat tracking on GuitarSet |
| `sweep_*.mjs` | parameter searches used to tune the cost weights |
| `helper_e2e.mjs` | the paste-a-link flow through the link helper (`helper/helper.ps1` running), in Chromium |
| `browser_e2e.mjs` | runs the real app in headless Chromium / WebKit (iPhone engine) |
| `run_file.mjs` | transcribe a WAV from the command line and print the tab |
