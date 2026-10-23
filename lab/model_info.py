import argparse
import json
import hashlib
from pathlib import Path
import onnx

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
a = p.parse_args()
m = onnx.load(str(a.model))
onnx.checker.check_model(m)


def tensors(xs):
    return [
        {
            "name": x.name,
            "shape": [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim],
        }
        for x in xs
    ]


print(
    json.dumps(
        {
            "file": a.model.name,
            "sha256": hashlib.sha256(a.model.read_bytes()).hexdigest(),
            "inputs": tensors(m.graph.input),
            "outputs": tensors(m.graph.output),
            "operators": sorted(set(n.op_type for n in m.graph.node)),
        },
        indent=2,
    )
)
