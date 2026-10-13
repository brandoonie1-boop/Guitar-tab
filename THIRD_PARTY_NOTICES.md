# Third-party notices

| Component | Used for | License |
|---|---|---|
| [Demucs](https://github.com/adefossez/demucs) `htdemucs_6s` weights and model code (Meta AI / Alexandre Défossez) | guitar isolation (`assets/demucs-6s-*.bin`, exported to ONNX) | MIT |
| [basic-pitch](https://github.com/spotify/basic-pitch) (Spotify) | note detection model (`assets/basic-pitch-*.bin`); `engine/basicpitch.js` ports its note-creation code | Apache License 2.0 |
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (Microsoft) | running the models (`vendor/ort/`, `assets/ort-wasm*.bin`) | MIT |
| [alphaTab](https://github.com/CoderLine/alphaTab) (Daniel Kuschny and contributors) | tab rendering, playback, Guitar Pro export (`vendor/alphatab/`) | Mozilla Public License 2.0 |
| Bravura music font (Steinberg) | notation glyphs (`vendor/alphatab/font/`) | SIL Open Font License 1.1 |
| Sonivox soundfont | synth playback (`vendor/alphatab/soundfont/`) | Apache License 2.0 (see `vendor/alphatab/soundfont/LICENSE`) |
| [GuitarSet](https://guitarset.weebly.com/) (Xi, Bittner, et al.) | evaluation only, not included | CC BY 4.0 |

Unmodified copies of the alphaTab and soundfont license files are in `vendor/alphatab/`.

The basic-pitch note-creation port in `engine/basicpitch.js` is a derivative of Apache-2.0 code:
Copyright 2022 Spotify AB, licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
