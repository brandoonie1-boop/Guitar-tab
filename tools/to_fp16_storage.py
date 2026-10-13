"""Store float32 initializers as float16 and insert Cast(->float32) nodes.
Compute stays in float32 (ORT folds the casts at load), download size halves.
The htdemucs_6s weights were published in fp16, so this is lossless for them."""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
min_elems = int(sys.argv[3]) if len(sys.argv) > 3 else 256
m = onnx.load(src)
g = m.graph
new_inits, cast_nodes = [], []
n_conv = 0
for t in g.initializer:
    if t.data_type == TensorProto.FLOAT and int(np.prod(t.dims)) >= min_elems:
        arr = numpy_helper.to_array(t).astype(np.float32)
        h = numpy_helper.from_array(arr.astype(np.float16), name=t.name + '__f16')
        new_inits.append(h)
        cast_nodes.append(helper.make_node('Cast', [h.name], [t.name], to=TensorProto.FLOAT, name='cast_' + t.name))
        n_conv += 1
    else:
        new_inits.append(t)
del g.initializer[:]
g.initializer.extend(new_inits)
nodes = list(g.node)
del g.node[:]
g.node.extend(cast_nodes + nodes)
onnx.checker.check_model(m)
onnx.save(m, dst)
print('converted', n_conv, 'initializers')
