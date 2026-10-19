"""Network architectures of the pretrained models, plus loading helpers.

Layer construction order must stay as is: the Keras 2 HDF5 weight files are
loaded by topological order, not by layer name.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import keras
from keras import layers

WEIGHTS_DIR = Path(__file__).parent / "weights"


def _squeeze_channels(x):
    return layers.Lambda(lambda t: keras.ops.squeeze(t, axis=3), name="squeeze")(x)


def _conv(filters, kernel, name, activation="relu"):
    return layers.Conv2D(filters, kernel, padding="same", activation=activation, name=name)


def _harmonic_stack(x, suffix=""):
    """conv1-conv4 (5x5) and the two 70x3 'harmonic' convs, each followed by batch norm."""
    for i, filters in enumerate([16, 32, 32, 32], start=1):
        x = layers.BatchNormalization()(_conv(filters, (5, 5), f"conv{i}{suffix}")(x))
    x = layers.BatchNormalization()(_conv(32, (70, 3), f"harm1{suffix}")(x))
    x = layers.BatchNormalization()(_conv(32, (70, 3), f"harm2{suffix}")(x))
    return x


def _head(x, deep):
    if deep:
        x = layers.BatchNormalization()(_conv(64, (3, 3), "conv7")(x))
        x = layers.BatchNormalization()(_conv(64, (3, 3), "conv8")(x))
    x = layers.BatchNormalization()(_conv(8, (360, 1), "distribution")(x))
    x = _conv(1, (1, 1), "squishy", activation="sigmoid")(x)
    return _squeeze_channels(x)


def _inputs():
    # HCQT magnitude and phase differentials: (freq bins, time frames, harmonics)
    return keras.Input(shape=(None, None, 5)), keras.Input(shape=(None, None, 5))


def _early_fusion(deep):
    mag, dphase = _inputs()
    a = layers.BatchNormalization()(mag)
    b = layers.BatchNormalization()(dphase)
    a = layers.BatchNormalization()(_conv(16, (5, 5), "conv1a")(a))
    b = layers.BatchNormalization()(_conv(16, (5, 5), "conv1b")(b))
    x = layers.Concatenate()([a, b])
    for i in (2, 3, 4):
        x = layers.BatchNormalization()(_conv(32, (5, 5), f"conv{i}")(x))
    x = layers.BatchNormalization()(_conv(32, (70, 3), "harm1")(x))
    x = layers.BatchNormalization()(_conv(32, (70, 3), "harm2")(x))
    return keras.Model(inputs=[mag, dphase], outputs=_head(x, deep))


def build_model1():
    """Early/Deep"""
    return _early_fusion(deep=True)


def build_model2():
    """Early/Shallow"""
    return _early_fusion(deep=False)


def build_model3():
    """Late/Deep"""
    mag, dphase = _inputs()
    a = _harmonic_stack(layers.BatchNormalization()(mag), "a")
    b = _harmonic_stack(layers.BatchNormalization()(dphase), "b")
    x = layers.Concatenate()([a, b])
    return keras.Model(inputs=[mag, dphase], outputs=_head(x, deep=True))


@dataclass(frozen=True)
class PretrainedModel:
    builder: Callable[[], keras.Model]
    weights: str
    threshold: float
    description: str

    def load(self):
        model = self.builder()
        model.load_weights(WEIGHTS_DIR / self.weights)
        return model


PRETRAINED = {
    "model1": PretrainedModel(build_model1, "exp1multif0.h5", 0.4, "Early/Deep"),
    "model2": PretrainedModel(build_model2, "exp2multif0.h5", 0.5, "Early/Shallow"),
    "model3": PretrainedModel(build_model3, "exp3multif0.h5", 0.5, "Late/Deep (recommended)"),
    "model4": PretrainedModel(build_model3, "exp4multif0.h5", 0.4, "Late/Deep, trained without BSQ (paper exp. 2)"),
    "model5": PretrainedModel(build_model3, "exp5multif0.h5", 0.5, "Late/Deep, trained without reverb files (paper exp. 3)"),
}
