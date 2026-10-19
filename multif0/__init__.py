"""Multiple F0 estimation in vocal ensembles (Cuesta, McFee & Gómez, ISMIR 2020).

    import multif0
    model = multif0.load("model3")
    salience = multif0.salience(model, y, sr)   # (360 bins, frames), hop HOP_LENGTH at SR
    times, freqs = multif0.utils.pitch_activations_to_mf0(salience, multif0.PRETRAINED["model3"].threshold)

Importing this package imports TensorFlow.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import features, models, predict, utils  # noqa: E402
from .features import HOP_LENGTH, SR  # noqa: E402
from .models import PRETRAINED  # noqa: E402

__all__ = ["HOP_LENGTH", "PRETRAINED", "SR", "features", "load", "models", "predict", "salience", "utils"]


def load(name="model3", cpu=False):
    """Build pretrained model `name` (see `PRETRAINED`) and load its weights.

    `cpu` hides the GPU from TensorFlow, which only works before TensorFlow first uses it
    (in this process); later calls keep whatever devices TensorFlow already set up.
    """
    if cpu:
        import tensorflow as tf

        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
    return PRETRAINED[name].load()


def salience(model, y, sr):
    """The model's pitch salience for mono audio `y` at rate `sr`: (360, n_frames) float32.

    Rows are the bins of `utils.get_freq_grid()`, columns the frames of `utils.get_time_grid()`
    (hop `HOP_LENGTH` samples at `SR`).
    """
    return predict.salience_from_features(model, *features.hcqt(y, sr))
