import sys, time, math
import numpy as np
import onnx, onnxruntime as ort
from collections import Counter
sys.path.insert(0, '.')
from export_demucs import np_spec, np_ispec, cac, uncac

path = sys.argv[1] if len(sys.argv) > 1 else 'models/htdemucs_6s_core.onnx'
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 2
m = onnx.load(path, load_external_data=True)
ops = Counter(n.op_type for n in m.graph.node)
print('nodes', len(m.graph.node), 'ops', dict(ops))
inits = sorted(m.graph.initializer, key=lambda t: -np.prod(t.dims))
print('n initializers', len(inits))
for t in inits[:8]:
    print('  ', t.name[:60], t.dims, onnx.TensorProto.DataType.Name(t.data_type))

sig = np.load('models/test_sig.npy')
ref = np.load('models/test_ref.npy')
L = sig.shape[-1]
z = np_spec(sig.astype(np.float64))
mag = cac(z).astype(np.float32)[None]
so = ort.SessionOptions()
so.intra_op_num_threads = threads
t0 = time.time()
sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
print('load', round(time.time() - t0, 2), 's')
for i in range(2):
    t0 = time.time()
    spec, wave = sess.run(None, {'mag': mag, 'mix': sig[None]})
    print('run', i, round(time.time() - t0, 2), 's')
out = np_ispec(uncac(spec[0].astype(np.float64), 6, 2), L) + wave[0]
print('max abs err vs torch', np.abs(out - ref).max(), 'ref max', np.abs(ref).max())
