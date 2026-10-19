"""Multiple F0 estimation in vocal ensembles (Cuesta, McFee & Gómez, ISMIR 2020), in PyTorch.

import multif0

model = multif0.MultiF0.from_pretrained("late-deep")    # device="auto": CUDA, MPS or CPU
pred = model.predict("choir.wav")                        # or model.predict(y, sr=sr)
pred.times, pred.freqs                                   # F0 estimates (Hz) per frame
pred.salience                                            # (360 bins, frames)

multif0.predict("choir.wav")                             # one-shot, keeps the model loaded
"""

from . import features, models, utils
from .core import AudioInput, MultiF0, Prediction, clear_cache, predict
from .devices import resolve_device
from .features import HOP_LENGTH, N_BINS, SR
from .models import DEFAULT_MODEL, PRETRAINED, EarlyFusion, LateFusion, Pretrained, SalienceNet

__all__ = [
    "DEFAULT_MODEL",
    "HOP_LENGTH",
    "N_BINS",
    "PRETRAINED",
    "SR",
    "AudioInput",
    "EarlyFusion",
    "LateFusion",
    "MultiF0",
    "Prediction",
    "Pretrained",
    "SalienceNet",
    "clear_cache",
    "features",
    "models",
    "predict",
    "resolve_device",
    "utils",
]
