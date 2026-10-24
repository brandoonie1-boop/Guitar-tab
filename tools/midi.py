"""Minimal Standard MIDI File reader: notes with absolute times (seconds), per track, with program."""
import struct

def _vlq(b, i):
    v = 0
    while True:
        c = b[i]; i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80: return v, i

def read_midi(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'MThd'
    hl = struct.unpack('>I', b[4:8])[0]
    fmt, ntrk, div = struct.unpack('>HHH', b[8:14])
    i = 8 + hl
    tracks = []
    tempo_events = []  # (tick, us_per_beat)
    for t in range(ntrk):
        assert b[i:i+4] == b'MTrk', b[i:i+4]
        ln = struct.unpack('>I', b[i+4:i+8])[0]
        j, end = i + 8, i + 8 + ln
        tick, status = 0, 0
        evs = []
        name = ''
        while j < end:
            dt, j = _vlq(b, j); tick += dt
            c = b[j]
            if c == 0xFF:
                typ = b[j+1]; l, k = _vlq(b, j+2); data = b[k:k+l]; j = k + l
                if typ == 0x51: tempo_events.append((tick, int.from_bytes(data, 'big')))
                elif typ == 0x03: name = data.decode('latin1', 'replace')
                continue
            if c in (0xF0, 0xF7):
                l, k = _vlq(b, j+1); j = k + l; continue
            if c & 0x80: status = c; j += 1
            hi = status & 0xF0; ch = status & 0x0F
            if hi in (0xC0, 0xD0):
                evs.append((tick, hi, ch, b[j], 0)); j += 1
            else:
                evs.append((tick, hi, ch, b[j], b[j+1])); j += 2
        tracks.append((name, evs))
        i = end
    tempo_events.sort()
    if not tempo_events or tempo_events[0][0] != 0: tempo_events.insert(0, (0, 500000))
    def t2s(tick):
        s, last_t, last_tempo = 0.0, 0, tempo_events[0][1]
        for tt, tempo in tempo_events[1:]:
            if tt >= tick: break
            s += (tt - last_t) * last_tempo / 1e6 / div; last_t, last_tempo = tt, tempo
        return s + (tick - last_t) * last_tempo / 1e6 / div
    out = []
    for name, evs in tracks:
        program, chan = None, None
        on = {}
        notes = []
        for tick, hi, ch, a, v in evs:
            if hi == 0xC0 and program is None: program = a
            if hi == 0x90 and v > 0:
                chan = ch
                on.setdefault((ch, a), []).append((tick, v))
            elif hi == 0x80 or (hi == 0x90 and v == 0):
                st = on.get((ch, a))
                if st:
                    t0, vel = st.pop(0)
                    notes.append((t2s(t0), t2s(tick), a, vel))
        if notes:
            notes.sort()
            out.append({'name': name, 'program': program if program is not None else 0, 'drum': chan == 9, 'notes': notes})
    return out
