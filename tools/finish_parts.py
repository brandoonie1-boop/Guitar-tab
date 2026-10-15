"""Optimise each exported Demucs part with ONNX Runtime's basic level, done offline so the
browser has less to do. Writes <part>.opt.onnx (what pack_demucs.py packs) and
<part>.f16.onnx (float16 weights with Cast nodes, a self-contained file for tests).

usage: python tools/finish_parts.py models/chain343980
"""
import argparse
import json
import os
import subprocess
import sys

import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    a = ap.parse_args()
    chain = json.load(open(os.path.join(a.dir, 'chain.json')))
    for p in chain['parts']:
        src = os.path.join(a.dir, p['file'] + '.onnx')
        opt = os.path.join(a.dir, p['file'] + '.opt.onnx')
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = opt
        ort.InferenceSession(src, so, providers=['CPUExecutionProvider'])
        cur = opt
        dst = os.path.join(a.dir, p['file'] + '.f16.onnx')
        subprocess.check_call([sys.executable, os.path.join(HERE, 'to_fp16_storage.py'), cur, dst])
        print(p['file'], os.path.getsize(dst) // 1000, 'kB', flush=True)


if __name__ == '__main__':
    main()
