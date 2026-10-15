# Guitar Tab

Turn any song into guitar tab. Drop in an audio or video file, paste a link, or record the song while it plays, and the app:

1. pulls the guitar out of the mix (and splits two guitars that are panned left/right),
2. hears the notes, including bends,
3. finds the beat and the bar lines (including songs that change meter),
4. works out the tuning, the capo position and how far the recording is from A440,
5. picks strings and frets the way a player would, and writes the tab.

Everything runs on your own device, in the browser. Nothing is uploaded anywhere.

**Open it:** https://guitar-tab.onrender.com/ (a free Render static site built from this repo's `main` branch; after pushing changes, press *Manual Deploy* in the Render dashboard, or connect the repo through Render's GitHub app so pushes deploy automatically)

## Install it like an app

- **iPhone (Safari):** open the link, tap the Share button, then **Add to Home Screen**.
- **Windows 10 (Chrome or Edge):** open the link, then click the install icon at the right end of the address bar (or menu → *Apps* → *Install this site as an app*).

The first song downloads about 120 MB of models and engine files once (about 135 MB when it uses the graphics chip); after that the app works offline.

## Using it

- **Get a song in:**
  - *Choose a file*: any audio or video file (MP3, M4A, WAV, FLAC, a phone video...).
  - *Paste a link* to an audio/video file and it downloads (if the site allows it).
  - *YouTube link* (computer, Chrome or Edge): the video opens inside the app; press *Record the song*, choose *This tab* and keep *Share tab audio* on. The video plays from the start and the recording stops when it ends. YouTube doesn't let apps download its audio, so the app records it while it plays.
  - *Spotify, SoundCloud, Bandcamp and other pages* (computer): open the song in another tab, press *Record*, and pick that tab with tab audio on (or on Windows, *Entire screen* with *Share system audio* for a desktop app).
  - *On iPhone*: use iPhone *Screen Recording* while the song plays in its app, then pick the recording with *Choose a file*. Or press *Record* / *Listen* and let the phone hear the song playing on another device or speaker.
  - Recordings stay on your device. Only record music you're allowed to use.
- **Make the tab:** choose a file, then *Make the tab*. A 3-minute song takes roughly 3–7 minutes on a laptop's processor (much less when the browser can use the graphics chip) and longer on a phone. Pick a section (*Part of the song*) to go faster. On a phone, keep the screen on while it works.
- **Tuning and capo:** the best match is shown with other possibilities and how likely each one is. Tap another one and the whole tab is re-fingered for it. Add the tunings you play to *Tunings you use* so they get picked more readily.
- **Tempo and bars:** tap the tempo to fix half-time / double-time mix-ups or shift where bars start.
- **Fix a note:** tap it in the tab to move it to another string, nudge the pitch, or delete it. *Undo* reverts.
- **Practice:** play the original recording, the isolated guitar, or the tab itself (synth), at 50–100% speed, with looping. The cursor follows along.
- **Export:** text tab (.txt), Guitar Pro (.gp, opens in Guitar Pro / TuxGuitar / MuseScore / Songsterr), MIDI, and the isolated guitar as .wav.
- **Saved tabs** stay on the device you made them on.

## If a run stops or the page reloads

Separating a whole song is memory-hungry, and phones end a page that uses too much. The app is built to stay well inside those limits (about 0.9 GB in total for a 4–5 minute song on a phone, 1.1–1.3 GB on a computer), and it recovers by itself if a run still fails:

- If the graphics chip fails or gives bad output, the run restarts on the processor.
- If memory runs out, the run restarts with the **light isolation model** (about 40% less memory, same notes found on our tests). Phones and computers with 4 GB of memory or less use it from the start. You can switch models under *Guitar isolation model*.
- If the browser ends the page in the middle of a run, the app notices on the next visit, says so, and picks lighter settings for the next try.
- Long songs on a phone: pick a section under *Part of the song* if it still stops.
- On a phone, keep the app open and the screen on until it finishes. Switching apps pauses the work and can end it.

## How accurate is it?

It gets you most of the way, not a perfect 1:1 copy. Expect to check it by ear and fix some notes.

Measured on [GuitarSet](https://guitarset.weebly.com/) (real recordings with the true tab):

| | result |
|---|---|
| Notes found, guitar on its own (onset F1, 40 takes) | 90%: chord comping 89.5%, single-note lines 91.6% (basic-pitch, the previous note detector: 79%) |
| Notes found with drums and bass mixed in, after isolation (16 takes) | 88.7% (basic-pitch: 78.4%) |
| String/fret matches what the player used (true notes / transcribed notes with a band) | 80% / 68% |
| Tuning + capo correct on standard-tuning takes | about 19 in 20 (116 of 120 from the true notes, 15 of 16 transcribed with a band) |
| DAEABE capo 7 test song (Brave Bird style arpeggios) | tuning and capo right; notes 99% and string/fret 97% for the guitar alone, notes 85% and string/fret 100% for the full band mix |
| Tuning + capo correct on test riffs in DAEABE, FACGCE, open D, drop D, Eb, capo 2/3/7 | 9 of 10 (DADGAD capo 2 was mistaken for standard) |
| Tempo correct (with drums and bass) | 37 of 40 takes (was 32 before the beat model); beat F-measure 0.98 |
| Bar lines (downbeats) in the right place | F-measure 0.78 (was 0.30) |

Things it cannot know for sure:

- **The capo.** A capo can't be heard. Many tuning/capo combinations sound exactly the same (standard with capo 2 = D standard with capo 4), and the tab numbers come out identical for those, so you can play it either way. The app picks the most likely setup and shows the others.
- **Which string.** The same note can often be played in 2–4 places. The fingering is a best guess at what's comfortable; tap a note to move it.
- **Low guitar notes under a bass guitar** can get pulled into the bass during isolation and go missing.
- Two guitars that sit in the middle of the mix can't be pulled apart; they come out as one part.

## How it works

| Step | What does it |
|---|---|
| Guitar isolation | [Demucs v4](https://github.com/adefossez/demucs) `htdemucs_6s` (has a dedicated guitar stem), exported to ONNX with the STFT done in JS. To fit in a phone's memory the model runs as a chain of 13 small graphs (encoder layers, transformer, decoder layers; the outer layers' frequency and time branches separately), the outer layers work through the spectrogram in bands, attention is computed in blocks, and only the stems that are needed come back. Same maths as the original: the engine needs about 450 MB instead of 2 GB (260 MB for the light model, which uses 3.9 s segments instead of 7.8 s). Runs on the graphics chip (WebGPU) when there is one, checks its output, and falls back to the processor. |
| Two guitars | stereo-position split of the guitar stem |
| Notes | a high-resolution guitar transcription model ([Riley, Edwards & Dixon, ICASSP 2024](https://arxiv.org/abs/2402.15258), weights trained on the François Leduc guitar recordings, never on GuitarSet) on [Kong et al.'s](https://github.com/qiuqiangkong/piano_transcription_inference) onset/offset-regression network: exported to ONNX as four convolution parts and a recurrent head, log-mel front end and note decoding in JS. The convolutions run once per frame in short chunks (on the graphics chip when there is one); the recurrent head runs on overlapping 10 s windows as it was trained. Spotify [basic-pitch](https://github.com/spotify/basic-pitch) adds the pitch-bend tracks, and is the light fallback |
| Beats and bar lines | [Beat This!](https://github.com/CPJKU/beat_this) small model (ISMIR 2024) on the mix, with its log-mel front end in JS; bar lines by a Viterbi over bar lengths 3–7 scored by its downbeat output (falls back to an onset-envelope tracker) |
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

The Demucs model files are generated from the official weights (`adefossez/HTDemucs-6s` on Hugging Face), from a folder holding `models/5c90dfd2.json` and `models/htdemucs_6s.safetensors`:

```
python tools/export_demucs_parts.py --segment 343980 --out models/chain343980 --check
python tools/export_demucs_parts.py --segment 171990 --out models/chain171990 --check
python tools/finish_parts.py models/chain343980
python tools/finish_parts.py models/chain171990
python tools/pack_demucs.py --out models/packed models/chain343980 models/chain171990
```

then copy `demucs.weights.bin`, `demucs-343980.chain` and `demucs-171990.chain` into `models/` and run the build. `tests/test_demucs.mjs` checks a packed chain against PyTorch's output.

The note model is `guitar-fl.pth` from [xavriley/midi-transcription-models](https://huggingface.co/xavriley/midi-transcription-models) (`pip install piano_transcription_inference torch onnx onnxruntime`):

```
python tools/export_hrgt.py guitar-fl.pth models/hrgt_fl --check some_guitar.wav
python tools/finish_parts.py models/hrgt_fl
python tools/pack_demucs.py --out models/packed --weights hrgt.weights.bin --name hrgt.chain --f16 models/hrgt_fl
```

then copy `hrgt.weights.bin` and `hrgt.chain` into `models/`. `tests/test_hrgt.mjs` checks the JS front end, the ONNX parts and the note decoding against PyTorch (reference data from `tools/make_hrgt_ref.py`).

## Credits and licenses

This app's own code is MIT licensed. It includes third-party work under their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
