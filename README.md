# Guitar Tab

Turn any song into guitar tab. Drop in an audio or video file, paste a link, or record the song while it plays, and the app:

1. pulls the guitar out of the mix with BS-RoFormer, one of the best music separation models there is (and splits two guitars that are panned left/right),
2. hears the notes, including bends,
3. finds the beat and the bar lines (including songs that change meter),
4. works out the tuning, the capo position and how far the recording is from A440,
5. listens to which string each note was played on (a note sounds different on a thicker string higher up the neck), weighs that against what's comfortable to play, and writes the tab.

Then it plays the tab back like Songsterr does: with real recorded guitar notes, a cursor that follows along, slow-down, loops, a metronome and a fretboard.

Everything runs on your own device, in the browser. Nothing is uploaded anywhere.

**Open it:** https://guitar-tab.onrender.com/ (a free Render static site built from this repo's `main` branch; after pushing changes, press *Manual Deploy* in the Render dashboard, or connect the repo through Render's GitHub app so pushes deploy automatically)

## Install it like an app

- **iPhone (Safari):** open the link, tap the Share button, then **Add to Home Screen**.
- **Windows 10 (Chrome or Edge):** open the link, then click the install icon at the right end of the address bar (or menu → *Apps* → *Install this site as an app*).

The first song downloads the models once: about 65 MB of the app's own files (75 MB when it uses the graphics chip) and the 353 MB guitar isolation model, which comes straight from Hugging Face and is cut down on your device to the 104 MB the app keeps. With the *Standard* isolation model it is about 120 MB in all. After that the app works offline.

## Using it

- **Get a song in:**
  - *Choose a file*: any audio or video file (MP3, M4A, WAV, FLAC, a phone video...).
  - *Paste a link* (Windows computer): with the **Guitar Tab link helper** running, YouTube, YouTube Music, SoundCloud, Bandcamp, Vimeo and most other sites go straight in, and so do Spotify, Apple Music, Deezer and Tidal links (those services encrypt their audio, so the helper finds the same recording on YouTube by artist and title). Paste the link and the tab starts by itself. The first time, the app shows the one-time setup (one PowerShell line, about a minute). See [Link helper](#link-helper) below.
  - Links straight to an audio/video file download without the helper when the site allows it.
  - Without the helper (or on a Mac): press *Record* and pick the tab that plays the song with tab audio on (or on Windows, *Entire screen* with *Share system audio* for a desktop app).
  - *On iPhone*: use iPhone *Screen Recording* while the song plays in its app, then pick the recording with *Choose a file*. Or press *Record* / *Listen* and let the phone hear the song playing on another device or speaker.
  - Recordings stay on your device. Only record music you're allowed to use.
- **Make the tab:** choose a file, then *Make the tab* (a pasted link starts by itself). With the *Best* isolation model (the default) a 3-minute song takes a few minutes when the browser can use the graphics chip, but on the processor alone it can take half an hour or more; when a run would take long, the app offers to switch to the *Standard* model, which takes roughly 2–5 minutes on a laptop's processor (longer on a phone). Pick a section (*Part of the song*) to go faster. *Speed: Balanced* (the default) does about a third less work than *Best quality* and found the same notes in our tests. On a phone, keep the screen on while it works.
- **Tuning and capo:** worked out from the whole song, all guitar parts together. The best match is shown with other possibilities and how likely each one is. Tap another one and the whole tab is re-fingered for it. Add the tunings you play to *Tunings you use* so they get picked more readily.
- **Tempo and bars:** tap the tempo to fix half-time / double-time mix-ups or shift where bars start.
- **Fix a note:** tap it in the tab to move it to another string, nudge the pitch, or delete it. *Undo* reverts.
- **Play along, like Songsterr:** *Tab* plays the tab itself with real recorded guitar notes, every string and fret (steel-string acoustic, nylon, clean electric, crunch or distortion through a modelled amp and speaker; the app starts with the sound that matches the recording), including slides, bends, vibrato, hammer-ons, pull-offs, palm mutes and how hard each note was played. Or play the original recording or the isolated guitar in sync with the tab. Speed 25–125%, looping (the whole song, or select bars in the tab to loop just those), a metronome and count-in, and a fretboard under the tab that shows each note as it plays. The cursor follows along; tap the tab to jump there.
- **Export:** text tab (.txt), Guitar Pro (.gp, opens in Guitar Pro / TuxGuitar / MuseScore / Songsterr), MIDI, and the isolated guitar as .wav.
- **Saved tabs** stay on the device you made them on.

## Link helper

Web pages can't download audio from YouTube and similar sites (those sites don't hand their audio to other pages), and doing it from a web server gets blocked and breaks the hosting rules. So the download happens on your own computer instead, with [yt-dlp](https://github.com/yt-dlp/yt-dlp):

- **Set up (Windows 10/11, no admin rights):** in the app, paste a link and press *Copy setup command*, then paste it into PowerShell. Or run this in PowerShell:
  ```
  [Net.ServicePointManager]::SecurityProtocol=3072; iex ((New-Object Net.WebClient).DownloadString('https://guitar-tab.onrender.com/helper/install.ps1'))
  ```
  It puts `helper.ps1`, `yt-dlp.exe` and `deno.exe` (yt-dlp needs a JavaScript runtime for YouTube) in `%LOCALAPPDATA%\GuitarTabHelper`, adds a *Guitar Tab helper* shortcut to the desktop and Start menu, and starts it. Run the same line again to update.
- **Use:** start the helper from the shortcut (a minimized window), then paste links in the app. The first time, the browser asks whether the site may use apps on this device: choose *Allow*.
- **Safety:** it listens only on 127.0.0.1 (this computer), answers only the Guitar Tab page (every call needs a header that forces the browser's cross-site check, and the Host header is checked), takes only web links, and keeps only the last song. yt-dlp updates itself each time the helper starts. Only download music you're allowed to.
- **Remove:** close the helper and delete `%LOCALAPPDATA%\GuitarTabHelper` and the two shortcuts.

The helper is a plain PowerShell script ([helper/helper.ps1](helper/helper.ps1)); it also runs with PowerShell 7 on macOS/Linux next to a `yt-dlp` binary.

## If a run stops or the page reloads

Separating a whole song is memory-hungry, and phones end a page that uses too much. The app is built to stay inside those limits: the Best isolation model runs as 14 small parts one after another, so its engine needs about 0.9 GB instead of 3 GB (the whole page peaks at about 1.6 GB on a computer); the standard model needs about 0.45 GB. If a run still fails, it recovers by itself:

- If the graphics chip fails or gives bad output (it is first tested on a known input), the run restarts on the processor.
- If memory runs out, the run restarts with the **standard** isolation model, and if that runs out too, with the **light** one (about 40% less memory again, same notes found on our tests). You can pick the model yourself under *Guitar isolation model*.
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
| Notes found on overdriven guitar (GuitarSet run through a simulated amp, 24 takes) | mild 81%, heavy 65% (basic-pitch: 74%, 60%) |
| Notes found, real electric guitar ([EGSet12](https://zenodo.org/records/11406378): a Telecaster through an amp, 12 takes) | 86.9% |
| String/fret matches what the player used, from the notes alone → now also from how each note sounds (all 360 takes; every player judged by a string model that never heard them) | true notes 75.4% → **91.3%**; transcribed notes 72.7% → **90.5%** |
| Same, with drums and bass mixed in, after isolation (40 takes) | 72.0% → **88.9%** |
| Same, real electric guitar (EGSet12, never used for training) | true notes 73.4% → **83.3%**; transcribed notes 61.8% → **74.2%** |
| Notes that come out completely right in the tab (pitch, timing, string and fret; F1) | GuitarSet 62.3 → **78.3**; with drums and bass 61.3 → **79.3**; EGSet12 53.6 → **64.5** |
| Tuning + capo correct on standard-tuning takes, transcribed | GuitarSet 343 → **350** of 360; with drums and bass 38 → **40** of 40; EGSet12 12 of 12 |
| Overdriven guitar (GuitarSet through a simulated amp, 20 takes): tuning + capo correct / notes completely right (F1) | mild 19 → **20** of 20 / 64.1 → **73.1**; heavy 8 → **16** of 20 / 22.7 → **44.4** (heavy distortion's made-up low notes used to pass for drop D) |
| Brave Bird, *Too Late Now* (the real recording, DAEABE capo 7) | tuning and capo right for all three guitar parts, 96% sure (measured with the previous version) |
| DAEABE capo 7 test song (Brave Bird style arpeggios, `tools/synth_song.py`) | tuning and capo right for the guitar alone (the previous version said standard tuning) and for the full band mix; string/fret 97% / 96% |
| Tuning + capo correct on test riffs in DAEABE, FACGCE, open D, drop D, Eb, capo 2/3/7 | 9 of 10 (DADGAD capo 2 was mistaken for standard) |
| Tempo correct (with drums and bass) | 37 of 40 takes (was 32 before the beat model); beat F-measure 0.98 |
| Bar lines (downbeats) in the right place | F-measure 0.78 (was 0.30) |

Things it cannot know for sure:

- **The capo.** A capo can't be heard. Many tuning/capo combinations sound exactly the same (standard with capo 2 = D standard with capo 4), and the tab numbers come out identical for those, so you can play it either way. The app picks the most likely setup and shows the others.
- **Which string, every time.** The same note can often be played in 2–4 places. The app listens for it (about 9 in 10 notes land where the player had them on the acoustic guitar recordings above, 3 in 4 on the electric ones) and otherwise goes by what's comfortable; tap a note to move it. Electric guitars are harder: their lighter strings and plain G string sound different from the acoustic recordings most of the training data comes from, so the app also learns each recording's strings from the song itself.
- **Low guitar notes under a bass guitar** can get pulled into the bass during isolation and go missing.
- **Heavy distortion on full chords** creates tones nobody played (differences between the chord's notes), and some come out as extra low notes, an octave below a power chord. The tuning and capo are judged only from notes both note detectors agree on, leaving out low notes that look like such ghosts (no fifth right above them, as a real drop-tuned low note has), but expect more cleanup on very distorted parts: about two in three notes are found there.
- Two guitars that sit in the middle of the mix can't be pulled apart; they come out as one part.

## How it works

| Step | What does it |
|---|---|
| Guitar isolation | **Best** (the default): [BS-RoFormer](https://arxiv.org/abs/2309.02612) SW, a band-split transformer with six stems, one of them guitar, from its public ONNX export on Hugging Face. The browser downloads it once (353 MB) and cuts it down as it arrives (`engine/onnxprune.js`, checked against the cut's SHA-256): only the guitar's mask estimator is kept, the attention over the 62 frequency bands is split into 6 slices (a WebGPU shader may use at most 8 buffers), and the graph is split into 14 parts, one per transformer layer, so the engine needs about 0.9 GB instead of 3.1 GB. STFT/iSTFT and the 4 s chunks with overlapping fades run in JS (`engine/bsroformer.js`); on a graphics chip, a known input is separated first and checked against the expected output. **Standard / Light:** [Demucs v4](https://github.com/adefossez/demucs) `htdemucs_6s` (has a dedicated guitar stem), exported to ONNX with the STFT done in JS. To fit in a phone's memory the model runs as a chain of 13 small graphs (encoder layers, transformer, decoder layers; the outer layers' frequency and time branches separately), the outer layers work through the spectrogram in bands, attention is computed in blocks, and only the stems that are needed come back. Same maths as the original: the engine needs about 450 MB instead of 2 GB (260 MB for the light model, which uses 3.9 s segments instead of 7.8 s). Both run on the graphics chip (WebGPU) when there is one, check their output, and fall back to the processor. A recording of a guitar on its own (separating it would only take some of it away: what is left after taking the guitar out is far quieter than the guitar) is used as it is |
| Two guitars | stereo-position split of the guitar stem |
| Notes | a high-resolution guitar transcription model ([Riley, Edwards & Dixon, ICASSP 2024](https://arxiv.org/abs/2402.15258), weights trained on the François Leduc guitar recordings, never on GuitarSet) on [Kong et al.'s](https://github.com/qiuqiangkong/piano_transcription_inference) onset/offset-regression network: exported to ONNX as four convolution parts and a recurrent head, log-mel front end and note decoding in JS. The convolutions run once per frame in short chunks (on the graphics chip when there is one); the recurrent head runs on overlapping 10 s windows as it was trained. Spotify [basic-pitch](https://github.com/spotify/basic-pitch) adds the pitch-bend tracks, and is the light fallback |
| Beats and bar lines | [Beat This!](https://github.com/CPJKU/beat_this) small model (ISMIR 2024) on the mix, with its log-mel front end in JS; bar lines by a Viterbi over bar lengths 3–7 scored by its downbeat output (falls back to an onset-envelope tracker) |
| Pitch reference | spectral peak histogram (cents from A440); corrected before note detection |
| Which string, from the sound | for every note, its first 24 overtones in three snapshots after the attack (level, how sharp each runs, a fitted inharmonicity coefficient), skipping overtones that collide with other notes; a small network (`engine/strings.js`, trained by `tools/train_strings.py` on [GuitarSet](https://guitarset.weebly.com/) and [Guitar-TECHS](https://zenodo.org/records/14963133) with random tone changes) scores each place on the neck the note could be played. After a first fingering pass, each string's stiffness is fitted from the recording itself and the notes are fingered again (electric guitars with a plain G string) |
| Strings, frets, tuning, capo | beam search over hand positions, ringing strings, stretches and open strings, plus what the sound says about each note's string; every candidate tuning × capo is scored by how naturally the music lays out on it and how well it fits the sound |
| Rendering, Guitar Pro | [alphaTab](https://www.alphatab.net/) |
| Tab playback | `app/guitarsynth.js`: a recording of every string and fret (from [Guitar-TECHS](https://zenodo.org/records/14963133)'s direct-input recordings for the electric and steel-string sounds, the steel-string one equalised string by string to [GuitarSet](https://guitarset.weebly.com/)'s microphone recordings; the [University of Iowa](https://theremin.music.uiowa.edu/MISguitar.html) samples for nylon), played the way a guitar works: one note per string at a time, strums spread across the strings, hammer-ons and pull-offs without a new pick attack, slides, bends and vibrato as pitch glides, palm mutes darker and shorter, each note as loud as it was heard. The electric sounds go through a modelled amp (drive, tone stack) and speaker cabinets measured from Guitar-TECHS's amp-microphone recordings, then a small room, a compressor and a limiter. It acts like a media element, so alphaTab's cursor follows it the same way it follows the recording |
| Runtime | [ONNX Runtime Web](https://onnxruntime.ai/) (WebAssembly, multi-threaded when the service worker enables cross-origin isolation) |

Code layout:

- `engine/` – the transcription engine (plain JS, runs in browsers and Node)
- `app/` – the interface, processing worker and storage
- `assets/` – model, runtime and tab-player sound files, split into ≤14 MB parts (the Best isolation model is not here: it comes from Hugging Face)
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

The Best isolation model needs nothing at build time: the browser downloads it from Hugging Face and cuts it itself (`app/models.js`; the cut's size and SHA-256 are pinned there; `tests/test_onnxprune.mjs` and `tests/test_bsroformer.mjs` check the cutter and the wrapper).

The tab player's sounds (`sounds/*.bin`; not in the repo, the build reuses the copies in `assets/`) are made from the sample datasets ([Guitar-TECHS](https://zenodo.org/records/14963133) `P1_singlenotes` and `P2_singlenotes`, [GuitarSet](https://zenodo.org/records/3371780), the [Iowa guitar samples](https://huggingface.co/datasets/sondalex/university-iowa-guitar-samples); numpy and ffmpeg):

```
python tools/gtechs_notes.py Guitar-TECHS gt_notes                   # one note per string and fret
python tools/build_guitar_sounds.py electric gt_notes snd
python tools/build_guitar_sounds.py acoustic gt_notes,GuitarSet snd  # each string matched to GuitarSet's microphone sound
python tools/build_guitar_sounds.py nylon iowa-guitar-samples snd
python tools/cab_ir.py Guitar-TECHS P1 cab_P1.npy && python tools/build_guitar_sounds.py ir cab_P1.npy snd/cab-bright.wav
python tools/cab_ir.py Guitar-TECHS P2 cab_P2.npy && python tools/build_guitar_sounds.py ir cab_P2.npy snd/cab-dark.wav
for b in electric nylon acoustic; do python tools/build_guitar_sounds.py pack sounds/$b.bin mp3=snd/$b.mp3 json=snd/$b.json; done
python tools/build_guitar_sounds.py pack sounds/cabs.bin bright=snd/cab-bright.wav dark=snd/cab-dark.wav
```

`tests/test_guitarsynth.mjs` checks the player's note scheduling; `tests/synth_render.mjs` renders a tab to a WAV through the real page to listen to.

## Credits and licenses

This app's own code is MIT licensed. It includes third-party work under their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
