"""Convert the original Keras 2 HDF5 weight files to PyTorch safetensors in multif0/weights/.

The HDF5 files are upstream's models/exp{1..5}multif0.pkl (despite the extension), also in this
repository's history as multif0/weights/exp{1..5}multif0.h5. Each output file is named after its
input's stem; the architecture comes from `multif0.models.PRETRAINED`.

    uv run --with h5py tools/convert_weights.py path/to/exp1multif0.h5 [...]

Convolutions keep their Keras names. Batch norms are auto-numbered in Keras, so each is mapped by
the layer feeding it, read from the model graph stored in the file.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from safetensors.torch import save_file

from multif0.models import ARCHITECTURES, PRETRAINED, WEIGHTS_DIR

KERAS_INPUTS = {"input_1": "mag", "input_2": "dphase"}


def keras_state(path: Path) -> dict[str, torch.Tensor]:
    with h5py.File(path, "r") as f:
        config = json.loads(f.attrs["model_config"])
        weights = f["model_weights"]
        state = {}
        for layer in config["config"]["layers"]:
            name, kind = layer["config"]["name"], layer["class_name"]
            group = weights[name][name] if kind in ("Conv2D", "BatchNormalization") else None
            if kind == "Conv2D":
                # (kh, kw, in, out) channels-last -> (out, in, kh, kw); both are cross-correlations
                state[f"convs.{name}.weight"] = np.transpose(group["kernel:0"][()], (3, 2, 0, 1))
                state[f"convs.{name}.bias"] = group["bias:0"][()]
            elif kind == "BatchNormalization":
                (source,) = [node[0][0] for node in layer["inbound_nodes"]]
                key = f"norms.{KERAS_INPUTS.get(source, source)}"
                state[f"{key}.weight"] = group["gamma:0"][()]
                state[f"{key}.bias"] = group["beta:0"][()]
                state[f"{key}.running_mean"] = group["moving_mean:0"][()]
                state[f"{key}.running_var"] = group["moving_variance:0"][()]
                state[f"{key}.num_batches_tracked"] = np.array(0, dtype=np.int64)
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in state.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("h5_files", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.h5_files:
        out_name = f"{path.stem}.safetensors"
        (pretrained,) = [p for p in PRETRAINED.values() if p.weights == out_name]
        state = keras_state(path)
        net = ARCHITECTURES[pretrained.architecture]()
        net.load_state_dict(state)  # strict: every parameter mapped, none left over
        save_file(state, WEIGHTS_DIR / out_name, metadata={"source": path.name})
        print(f"{path} -> {WEIGHTS_DIR / out_name} ({pretrained.name})")


if __name__ == "__main__":
    main()
