"""Pack exported Demucs chains (export_demucs_parts.py + finish_parts.py) for the app.

All chains share one weights file. Every large initializer is stored once (identical
tensors in different chains or parts are deduplicated), as float16 when that is lossless
(true for all of htdemucs_6s's published weights) and as float32 otherwise.

The part graphs keep float32 initializers that point to a per-part external data file
("p0.bin", ...). The app builds that file in memory just before creating the part's
session (float16 -> float32 conversion in JS), so ONNX Runtime only has to copy the
weights in: no Cast nodes to fold, which would roughly double the engine's memory peak.

Output:
  <out>/demucs.weights.bin        weight data
  <out>/demucs-<segment>.chain     'DMCH' | u32 header length | header JSON | part graphs
  header.parts[i].data = {"path": "p<i>.bin", "size": bytes,
                          "tensors": [[src offset, src bytes, dst offset, "h"=f16->f32 | "c"=copy], ...]}

usage: python tools/pack_demucs.py --out models/packed models/chain343980 models/chain171990
       python tools/pack_demucs.py --out models/packed --weights hrgt.weights.bin --name hrgt.chain --f16 models/hrgt
       (--f16 stores every float32 weight as float16 even when that rounds it; --name sets
        the chain file name for a single chain)
"""
import argparse
import hashlib
import json
import os
import struct

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

WEIGHTS = 'demucs.weights.bin'
MIN_BYTES = 1024
ALIGN = 64


def pad_to(buf, align):
    buf.extend(b'\0' * ((-len(buf)) % align))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--suffix', default='.opt.onnx', help='part files to pack (float32 initializers)')
    ap.add_argument('--weights', default=WEIGHTS, help='name of the shared weights file')
    ap.add_argument('--name', help='chain file name (one chain only)')
    ap.add_argument('--f16', action='store_true', help='round float32 weights to float16 (lossy)')
    ap.add_argument('chains', nargs='+')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    weights = bytearray()
    index = {}
    stats = {'h': 0, 'c': 0}
    for d in a.chains:
        chain = json.load(open(os.path.join(d, 'chain.json')))
        blobs, parts = [], []
        offset = 0
        for pi, p in enumerate(chain['parts']):
            m = onnx.load(os.path.join(d, p['file'] + a.suffix))
            onnx.checker.check_model(m)
            path = f'p{pi}.bin'
            tensors, size = [], 0
            for t in m.graph.initializer:
                assert t.data_location != TensorProto.EXTERNAL
                arr = numpy_helper.to_array(t)
                raw = arr.tobytes()
                if len(raw) < MIN_BYTES:
                    continue
                kind = 'c'
                src = raw
                if arr.dtype == np.float32:
                    h = arr.astype(np.float16)
                    if a.f16 or np.array_equal(h.astype(np.float32), arr, equal_nan=True):
                        kind, src = 'h', h.tobytes()
                key = hashlib.sha1(kind.encode() + str(arr.dtype).encode() + str(list(arr.shape)).encode() + src).hexdigest()
                if key not in index:
                    pad_to(weights, ALIGN)
                    index[key] = len(weights)
                    weights.extend(src)
                    stats[kind] += 1
                size += (-size) % ALIGN
                tensors.append([index[key], len(src), size, kind])
                for f in ('raw_data', 'float_data', 'int32_data', 'int64_data', 'double_data', 'uint64_data', 'string_data'):
                    t.ClearField(f)
                t.data_location = TensorProto.EXTERNAL
                del t.external_data[:]
                for k, v in (('location', path), ('offset', str(size)), ('length', str(len(raw)))):
                    e = t.external_data.add()
                    e.key, e.value = k, v
                size += len(raw)
            blob = m.SerializeToString()
            parts.append({'file': p['file'], 'in': p['in'], 'out': p['out'], 'offset': offset, 'length': len(blob),
                          'data': {'path': path, 'size': size, 'tensors': tensors}})
            blobs.append(blob)
            offset += len(blob)
        header = {k: v for k, v in chain.items() if k != 'parts'}
        header['weights'] = a.weights
        header['parts'] = parts
        hb = json.dumps(header, separators=(',', ':')).encode()
        out = os.path.join(a.out, a.name or f"demucs-{chain['segment']}.chain")
        with open(out, 'wb') as f:
            f.write(b'DMCH' + struct.pack('<I', len(hb)) + hb + b''.join(blobs))
        print(out, os.path.getsize(out) // 1000, 'kB')
    out = os.path.join(a.out, a.weights)
    with open(out, 'wb') as f:
        f.write(weights)
    print(out, len(weights) // 1000, 'kB;', stats['h'], 'float16 tensors,', stats['c'], 'stored as is')


if __name__ == '__main__':
    main()
