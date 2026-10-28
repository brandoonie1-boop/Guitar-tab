# Tests and evaluation scripts

Development scripts used to check and tune the engine. They expect a local setup:

- `npm install onnxruntime-node onnxruntime-web playwright-core` (Node 22)
- GuitarSet audio and annotations (https://zenodo.org/records/3371780); `python tools/guitarset_json.py <annotation folder> guitarset_annotations.json` turns the JAMS files into the one JSON file the scripts read (the same works for EGSet12)
- for the string classifier also Guitar-TECHS (https://zenodo.org/records/14963133) and EGSet12 (https://zenodo.org/records/11406378)
- paths at the top of the older scripts point at the original development machine; adjust them. The newer ones (`eval_strings.mjs`, `eval_notes.mjs`, `cache_activations.mjs`, `tools/string_data.mjs`, `tools/train_strings.py`, `tools/train_hrgt.py` and its data tools) take their paths as arguments.

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

Note model fine-tuned for distortion, from data to numbers (PyTorch CPU build, `pip install torch piano_transcription_inference audioread matplotlib soundfile scipy pedalboard`; `guitar-fl.pth` from https://huggingface.co/xavriley/midi-transcription-models):

```
python tools/hrgt_data.py gtechs Guitar-TECHS nd/gtechs                # recordings with their MIDI lined up
python tools/hrgt_data.py gs-anchor gs.json GuitarSet/audio nd/gs_anchor  # clean acoustic, no annotations
python tools/gtechs_notes.py Guitar-TECHS gt_notes                     # one recorded note per string and fret
python tools/synth_riffs.py gt_notes nd/synth --minutes 150            # riffs, chords and lines with exact notes
# the shipped model: one run in three stages, each resuming from runs/ft/last.pt (about 14 s a step)
C="--cabs assets/sound-cabs-<sha>.0.bin --gs gs.json:GuitarSet/audio --batch 2 --steps 1200 --val-every 200"
python tools/train_hrgt.py guitar-fl.pth runs/ft $C --lr 5e-5 --data nd/gtechs:0.6:0.5:1 --data nd/synth:0.4:1:0      # to step 250
python tools/train_hrgt.py guitar-fl.pth runs/ft $C --lr 5e-5 --data nd/gtechs:0.6:0.5:1 --data nd/synth:0.4:1:0 \
  --data nd/gs_anchor:0.25:0:0:1                                                                                  # to step 400
python tools/train_hrgt.py guitar-fl.pth runs/ft $C --lr 3e-5 --data nd/gtechs:0.6:0.25:1 --data nd/synth:0.4:0.5:0 \
  --data nd/gs_anchor:0.6:0:0:1                                                                                   # to step 800
python tools/blend_hrgt.py average runs/ft/step_600.pt runs/ft/step_800.pt runs/ft/avg_600_800.pt
python tools/blend_hrgt.py guitar-fl.pth runs/ft/avg_600_800.pt 0.7 hrgt_ft.pt
python tools/export_hrgt.py hrgt_ft.pt models/hrgt_ft --check some_guitar.wav
python tools/finish_parts.py models/hrgt_ft
python tools/pack_demucs.py --out models/packed --weights hrgt.weights.bin --name hrgt.chain --f16 models/hrgt_ft
node tests/eval_notes.mjs models gs.json GuitarSet/audio --suffix _mic --every 3                # clean
node tests/eval_notes.mjs models gs.json GuitarSet/audio --suffix _mic --every 6 --fx tanh:30   # heavy distortion
node tests/eval_notes.mjs models egset.json EGSet12 --fx amp:20 --write-wav egset_amp20
node tests/eval_strings.mjs egset egset.json egset_amp20 models/acts/egset_amp20_h500 default --detect
```

Training validates on takes the tests don't use (Guitar-TECHS P3, which training leaves out, and 8 GuitarSet takes outside `--every 3`), and the stages, the checkpoints and the blend share were chosen on those; EGSet12 is never used for training or choosing. Why the stages: on electric data alone, clean acoustic guitar slipped (validation 84.9 -> 82.9 after 200 steps) while distortion improved fast; the anchor of clean acoustic recordings (distillation only) at weight 0.6 brought it back to 83.8 and the 70/30 blend to 84.7.

Second round (a guitar in a band, pickups, Slakh guitar parts), from the weights above:

```
python tools/hrgt_data.py slakh babyslakh_16k nd/slakh_gtr nd/backing          # BabySlakh (https://zenodo.org/records/4603870): its guitar parts with their MIDI, and the rest of each song as backing
python tools/synth_riffs.py gt_notes nd/synth2 --minutes 200 --seed 2 --pickup any                # riffs through bridge, middle and neck pickups
python tools/synth_riffs.py gt_notes val/synth_bridge --minutes 10 --seed 99 --pickup bridge     # validation only
C="--cabs assets/sound-cabs-<sha>.0.bin --gs gs.json:GuitarSet/audio --backing nd/backing --val-extra synth_bridge=val/synth_bridge"
python tools/train_hrgt.py guitar-fl.pth runs/r3 --init hrgt_ft.pt $C --batch 2 --steps 1000 --lr 3e-5 --val-every 200 \
  --data nd/gtechs,w=0.45,gt=0.25,snap=1,band=0.4 --data nd/synth2,w=0.35,gt=1,band=0.4,kd=0.3 \
  --data nd/slakh_gtr,w=0.3,gt=1,snap=1,band=0.7 --data nd/gs_anchor,w=0.45,gt=0,clean=1,band=0.4
python tools/blend_hrgt.py average runs/r3/step_800.pt runs/r3/step_1000.pt hrgt_r3.pt         # then export and pack as above
```

`band=` is the share of a source's clips that are mixed into a band (another song's drums, bass and other instruments, or a Slakh part's own song, in time) and taken out again by a simulated separator. The simulator was set so that it costs the note model what BS-RoFormer itself does on the same mixes (GuitarSet validation takes in MUSDB18 backing: notes found 80.4 after either, 83.8 clean); the ideal ratio mask costs more (77.9), the first setting much more (68.0). The checkpoint and the new onset threshold (0.3) were chosen on validation sets only: the trainer's (Guitar-TECHS P3 and 8 GuitarSet takes, clean, heavy distortion, in a simulated band; synthetic bridge-pickup parts) plus the same with heavy distortion in a band, and the GuitarSet validation takes isolated from MUSDB18 backing by the real BS-RoFormer.

Band test bench: the test takes mixed into real drums, bass and vocals from MUSDB18's test songs (the guitar 3-9 dB under the band; each take always gets the same song, place and level) and isolated by the app's BS-RoFormer cut, run through the app's separation code (the isolated guitars are cached, so comparing note models only reruns the note model):

```
node tools/cut_bsr.mjs bs_roformer_sw_6stem_fp16.onnx bsr_guitar.bin guitar     # the cut the app makes on the device
node tests/eval_band.mjs models gs.json GuitarSet/audio musdb18_test bsr_guitar.bin --suffix _mic --every 12 --limit 16
node tests/eval_band.mjs models egset.json EGSet12 musdb18_test bsr_guitar.bin [--fx amp:20]
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
| `eval_notes.mjs` | notes found by the note model on any annotated set (GuitarSet, EGSet12), clean or through a simulated amp (`--fx tanh:G`, `--fx amp:G`), through the app's JS code; caches the model outputs per take |
| `eval_band.mjs` | notes found in a band: the test takes mixed into MUSDB18 drums, bass and vocals and isolated again by BS-RoFormer through the app's separation code (`--fx` as above); caches the isolated guitars |
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
