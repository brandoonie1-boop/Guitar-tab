# Third-party notices

| Component | Used for | License |
|---|---|---|
| [Demucs](https://github.com/adefossez/demucs) `htdemucs_6s` weights and model code (Meta AI / Alexandre Défossez) | guitar isolation (`assets/demucs-*.bin`, exported to ONNX as a chain of parts) | MIT |
| [Beat This!](https://github.com/CPJKU/beat_this) small model (Foscarin, Schlüter, Widmer; Johannes Kepler University Linz) | beats and bar lines (`assets/beat-this-*.bin`, exported to ONNX); `engine/beatthis.js` reproduces its log-mel front end and peak picking | MIT, Copyright 2024 Institute of Computational Perception, JKU Linz |
| Guitar transcription model: [high-resolution guitar transcription](https://huggingface.co/xavriley/midi-transcription-models) weights `guitar-fl.pth` (Xavier Riley, Drew Edwards, Simon Dixon; "High Resolution Guitar Transcription via Domain Adaptation", ICASSP 2024) on the CRNN of [piano_transcription_inference](https://github.com/qiuqiangkong/piano_transcription_inference) (Qiuqiang Kong / ByteDance, "High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times") | note detection (`assets/hrgt-*.bin`, exported to ONNX parts by `tools/export_hrgt.py`); `engine/hrgt.js` reproduces its log-mel front end (torchlibrosa) and ports its note post-processing | MIT (weights and code) |
| [basic-pitch](https://github.com/spotify/basic-pitch) (Spotify) | pitch-bend tracks and attack strengths for the detected notes, and note detection when the model above is off (`assets/basic-pitch-*.bin`); `engine/basicpitch.js` ports its note-creation code | Apache License 2.0 |
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (Microsoft) | running the models (`vendor/ort/`, `assets/ort-wasm*.bin`) | MIT |
| [alphaTab](https://github.com/CoderLine/alphaTab) (Daniel Kuschny and contributors) | tab rendering, playback, Guitar Pro export (`vendor/alphatab/`) | Mozilla Public License 2.0 |
| Bravura music font (Steinberg) | notation glyphs (`vendor/alphatab/font/`) | SIL Open Font License 1.1 |
| Sonivox soundfont | synth playback (`vendor/alphatab/soundfont/`) | Apache License 2.0 (see `vendor/alphatab/soundfont/LICENSE`) |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | downloads songs from links, on the user's own computer; fetched by `helper/install.ps1`, not part of the app | The Unlicense (public domain) |
| [Deno](https://deno.com/) | JavaScript runtime yt-dlp uses for YouTube; fetched by `helper/install.ps1`, not part of the app | MIT |
| [GuitarSet](https://guitarset.weebly.com/) (Xi, Bittner, et al.) | training data for the string classifier (`engine/strings-model.js` holds weights learned from it; no recordings included) and evaluation | CC BY 4.0 |
| [Guitar-TECHS](https://zenodo.org/records/14963133) (Pedroza, Abreu, Corey, Roman; ICASSP 2025) | training data for the string classifier (weights only; no recordings included) | CC BY 4.0 |
| [EGSet12](https://zenodo.org/records/11406378) | evaluation only, not included | CC BY 4.0 |

Unmodified copies of the alphaTab and soundfont license files are in `vendor/alphatab/`.

The basic-pitch note-creation port in `engine/basicpitch.js` is a derivative of Apache-2.0 code:
Copyright 2022 Spotify AB, licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
