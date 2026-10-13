# Guitar Tab

Turn any song into guitar tab. Drop in an audio or video file and the app:

1. pulls the guitar out of the mix (and splits two guitars that are panned left/right),
2. hears the notes, including bends,
3. finds the beat and the bar lines,
4. works out the tuning, the capo position and how far the recording is from A440,
5. picks strings and frets the way a player would, and writes the tab.

Everything runs on your own device, in the browser. Nothing is uploaded anywhere.

**Open it:** https://guitar-tab.onrender.com/ (a free Render static site built from this repo's `main` branch; after pushing changes, press *Manual Deploy* in the Render dashboard, or connect the repo through Render's GitHub app so pushes deploy automatically)

## Install it like an app

- **iPhone (Safari):** open the link, tap the Share button, then **Add to Home Screen**.
- **Windows 10 (Chrome or Edge):** open the link, then click the install icon at the right end of the address bar (or menu → *Apps* → *Install this site as an app*).

The first song downloads about 100 MB of models and engine files once; after that the app works offline.

## Using it

- **Make the tab:** choose a file, then *Make the tab*. A 3-minute song takes roughly 2–5 minutes on a laptop and longer on a phone. Pick a section (*Part of the song*) to go faster. On a phone, keep the screen on while it works.
- **Tuning and capo:** the best match is shown with other possibilities and how likely each one is. Tap another one and the whole tab is re-fingered for it. Add the tunings you play to *Tunings you use* so they get picked more readily.
- **Tempo and bars:** tap the tempo to fix half-time / double-time mix-ups or shift where bars start.
- **Fix a note:** tap it in the tab to move it to another string, nudge the pitch, or delete it. *Undo* reverts.
- **Practice:** play the original recording, the isolated guitar, or the tab itself (synth), at 50–100% speed, with looping. The cursor follows along.
- **Export:** text tab (.txt), Guitar Pro (.gp, opens in Guitar Pro / TuxGuitar / MuseScore / Songsterr), MIDI, and the isolated guitar as .wav.
- **Saved tabs** stay on the device you made them on.

## How accurate is it?

It gets you most of the way, not a perfect 1:1 copy. Expect to check it by ear and fix some notes.

Measured on [GuitarSet](https://guitarset.weebly.com/) (real recordings with the true tab):

| | result |
|---|---|
| Notes found, guitar on its own (onset F1) | 81% on average: chord comping 75%, single-note lines 87% |
| Notes found with drums and bass mixed in, after isolation | about 77% |
| String/fret matches what the player used (true notes / transcribed notes) | 75% / 70% |
| Tuning + capo correct on standard-tuning takes | about 19 in 20 |
| Tuning + capo correct on test riffs in DAEABE, FACGCE, open D, drop D, Eb, capo 2/3/7 | 9 of 10 (DADGAD capo 2 was mistaken for standard) |
| Tempo correct (with drums) | 32 of 40 takes; the rest were half or double speed, fixable with one tap |

Things it cannot know for sure:

- **The capo.** A capo can't be heard. Many tuning/capo combinations sound exactly the same (standard with capo 2 = D standard with capo 4), and the tab numbers come out identical for those, so you can play it either way. The app picks the most likely setup and shows the others.
- **Which string.** The same note can often be played in 2–4 places. The fingering is a best guess at what's comfortable; tap a note to move it.
- **Low guitar notes under a bass guitar** can get pulled into the bass during isolation and go missing.
- Two guitars that sit in the middle of the mix can't be pulled apart; they come out as one part.

## How it works

| Step | What does it |
|---|---|
| Guitar isolation | [Demucs v4](https://github.com/adefossez/demucs) `htdemucs_6s` (has a dedicated guitar stem), exported to ONNX with the STFT done in JS and attention computed in blocks to save memory. Runs on the graphics chip (WebGPU) when there is one, checks its output, and falls back to the processor. Matches the PyTorch model to 80+ dB SNR on both paths. |
| Two guitars | stereo-position split of the guitar stem |
| Notes | Spotify [basic-pitch](https://github.com/spotify/basic-pitch) (ONNX) + a JS port of its note decoding, thresholds re-tuned on GuitarSet |
| Tempo / beats | onset envelope + dynamic-programming beat tracker |
| Pitch reference | spectral peak histogram (cents from A440); corrected before note detection |
| Strings, frets, tuning, capo | beam search over hand positions, ringing strings, stretches and open strings; every candidate tuning × capo is scored by how naturally the music lays out on it |
| Rendering, playback, Guitar Pro | [alphaTab](https://www.alphatab.net/) |
| Runtime | [ONNX Runtime Web](https://onnxruntime.ai/) (WebAssembly, multi-threaded when the service worker enables cross-origin isolation) |

Code layout:

- `engine/` – the transcription engine (plain JS, runs in browsers and Node)
- `app/` – the interface, processing worker and storage
- `assets/` – model and runtime files, split into ≤14 MB parts
- `vendor/` – ONNX Runtime Web and alphaTab
- `tools/` – build script, local server, model export scripts
- `tests/` – accuracy tests and evaluation scripts

## Building

The repo is ready to serve as-is: Render serves the root of `main` as a static site. GitHub Pages works too (Settings → Pages → Deploy from branch `main`, folder `/`). After changing code, run:

```
npm install onnxruntime-web@1.30.0
node tools/build.mjs --ort node_modules/onnxruntime-web/dist
node tools/serve.mjs 8080
```

`tools/export_demucs.py` and `tools/to_fp16_storage.py` regenerate the Demucs ONNX model from the official weights (`adefossez/HTDemucs-6s` on Hugging Face).

## Credits and licenses

This app's own code is MIT licensed. It includes third-party work under their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
