"""Speaker-cabinet + amp + mic response from simultaneous DI and amp-mic recordings (Wiener estimate),
for the tab player's electric sounds (then tools/build_guitar_sounds.py ir / pack).

usage: python tools/cab_ir.py <Guitar-TECHS dir> <player: P1|P2> <out.npy>
"""
import subprocess, sys, numpy as np
SR = 48000
def load(p, ch=None):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)
SRC, player, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
b = f'{SRC}/{player}_singlenotes/audio'
di = load(f'{b}/directinput/directinput_allsinglenotes.wav'); mic = load(f'{b}/micamp/micamp_allsinglenotes.wav')
n = min(len(di), len(mic)); di, mic = di[:n], mic[:n]
# delay between the two
seg = slice(0, SR * 60)
c = np.fft.irfft(np.fft.rfft(mic[seg], 2 ** 23) * np.conj(np.fft.rfft(di[seg], 2 ** 23)))
lag = int(np.argmax(np.abs(np.concatenate([c[-2000:], c[:2000]])))) - 2000
print(player, 'mic lags DI by', lag, 'samples')
N = 8192; hop = N // 2; w = np.hanning(N)
Sxy = np.zeros(N // 2 + 1, complex); Sxx = np.zeros(N // 2 + 1)
for i in range(0, n - N, hop):
    X = np.fft.rfft(di[i:i + N] * w); Y = np.fft.rfft(mic[i:i + N] * w)
    Sxy += Y * np.conj(X); Sxx += np.abs(X) ** 2
H = Sxy / (Sxx + 1e-3 * Sxx.mean())
h = np.fft.irfft(H)
h = np.roll(h, 256)[:4096]  # keep a little pre-ringing, 85 ms total
h *= np.concatenate([np.hanning(512)[:256], np.ones(4096 - 256 - 1024), np.hanning(2048)[1024:]])
# how well a linear filter explains the amp mic signal (on a held-out stretch)
test = slice(SR * 300, SR * 360)
pred = np.convolve(di[test], h)[256: 256 + (test.stop - test.start)]
y = mic[test]
g = (pred @ y) / (pred @ pred)
snr = 10 * np.log10((y ** 2).sum() / ((y - g * pred) ** 2).sum())
print('linear model of the amp mic: SNR', round(snr, 1), 'dB')
np.save(OUT, h.astype(np.float32))
fr = np.fft.rfftfreq(len(h) * 4, 1 / SR); Hm = 20 * np.log10(np.abs(np.fft.rfft(h, len(h) * 4)) + 1e-9)
for f in [80, 150, 300, 600, 1000, 2000, 3000, 4000, 6000, 8000, 12000]:
    print(f'  {f:6d} Hz {Hm[np.argmin(np.abs(fr - f))] - Hm.max():6.1f} dB')
