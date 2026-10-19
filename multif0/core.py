"""The library API: load a pretrained model, then compute salience or multiple-F0 estimates."""

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import librosa
import numpy as np
import torch
from numpy.typing import NDArray

from . import features, utils
from .devices import DeviceLike, empty_cache, resolve_device
from .models import DEFAULT_MODEL, Pretrained, SalienceNet, get_pretrained

AudioInput = str | os.PathLike[str] | NDArray[np.floating[Any]] | torch.Tensor
"""An audio file path, or samples: mono (n,) or multichannel (channels, n), with `sr` given."""

# Frames per network call (~5.8 s). Each chunk is padded with CONTEXT_FRAMES of real neighbouring
# input, which covers the networks' temporal receptive field (+/-12 frames), so the result equals a
# single pass over the whole input. 500 is as fast as larger chunks on CPU and faster on MPS, and
# keeps memory near 1 GB.
CHUNK_FRAMES = 500
CONTEXT_FRAMES = 16


@dataclass(frozen=True)
class Prediction:
    """A model's output for one input.

    `salience` is (360 bins, frames): rows are `freq_grid` (20-cent bins from 32.7 Hz), columns
    `times` (hop `features.HOP_LENGTH` samples at `features.SR`, 11.6 ms). `freqs[i]` holds the F0
    estimates in Hz at `times[i]`: the salience peaks of at least `threshold`.
    """

    salience: NDArray[np.float32]
    times: NDArray[np.float64]
    freqs: list[NDArray[np.float64]]
    threshold: float

    @property
    def freq_grid(self) -> NDArray[np.float64]:
        return utils.get_freq_grid()

    @property
    def hop(self) -> float:
        """Seconds between frames."""
        return features.HOP_LENGTH / features.SR

    def to_csv(self, path: str | os.PathLike[str]) -> None:
        """Write the estimates as tab-separated rows: time, then zero or more F0s in Hz."""
        utils.save_multif0_output(self.times, self.freqs, path)


class MultiF0:
    """A pretrained network on a torch device.

        model = MultiF0.from_pretrained("late-deep", device="auto")
        pred = model.predict("choir.wav")            # or model.predict(y, sr=44100)
        pred.times, pred.freqs, pred.salience

    Delete the object, or call `release()` (or use it as a context manager), to free its device
    memory.
    """

    def __init__(self, net: SalienceNet, pretrained: Pretrained, device: DeviceLike = "auto"):
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.net: SalienceNet | None = net.eval().to(self.device)

    @classmethod
    def from_pretrained(cls, name: str = DEFAULT_MODEL, device: DeviceLike = "auto") -> Self:
        """Load a pretrained model by name (see `PRETRAINED`) or original ID ("model3").

        `device` is "auto" (CUDA, then MPS, then CPU), or any torch device.
        """
        pretrained = get_pretrained(name)
        return cls(pretrained.build(), pretrained, device)

    @property
    def threshold(self) -> float:
        """The peak threshold tuned for this model in the paper."""
        return self.pretrained.threshold

    def salience(self, audio: AudioInput, sr: int | None = None) -> NDArray[np.float32]:
        """Pitch salience of `audio`: (360 bins, frames) float32 in [0, 1]."""
        return self.salience_from_features(*_features(audio, sr))

    @torch.inference_mode()
    def salience_from_features(
        self, mag: NDArray[np.float32], dphase: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Run the network over HCQT features from `features.hcqt`, (bins, frames, harmonics)."""
        net = self._net()
        # (bins, frames, harmonics) -> (1, harmonics, bins, frames)
        mag_t, dphase_t = (
            torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(self.device)
            for x in (mag, dphase)
        )
        n_frames = mag.shape[1]
        chunks = []
        for start in range(0, n_frames, CHUNK_FRAMES):
            end = min(start + CHUNK_FRAMES, n_frames)
            lo, hi = max(0, start - CONTEXT_FRAMES), min(n_frames, end + CONTEXT_FRAMES)
            out = net(mag_t[..., lo:hi], dphase_t[..., lo:hi])[0]
            chunks.append(out[:, start - lo : end - lo].float().cpu())
        return torch.cat(chunks, dim=1).numpy()

    def predict(
        self, audio: AudioInput, sr: int | None = None, threshold: float | None = None
    ) -> Prediction:
        """Salience and multiple-F0 estimates of `audio`; `threshold` defaults to the model's."""
        salience = self.salience(audio, sr)
        threshold = self.threshold if threshold is None else threshold
        times, freqs = utils.pitch_activations_to_mf0(salience, threshold)
        return Prediction(salience, times, freqs, threshold)

    def release(self) -> None:
        """Drop the network and return its cached device memory. The model is unusable after."""
        self.net = None
        empty_cache(self.device)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"MultiF0({self.pretrained.name!r}, device={str(self.device)!r})"

    def _net(self) -> SalienceNet:
        if self.net is None:
            raise RuntimeError("this model was released")
        return self.net


def _features(audio: AudioInput, sr: int | None) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if isinstance(audio, str | os.PathLike):
        if sr is not None:
            raise ValueError("sr is only for in-memory audio; files are read at their own rate")
        return features.compute_hcqt(Path(audio))
    if sr is None:
        raise ValueError("sr is required for in-memory audio")
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        y = librosa.to_mono(y)
    if y.size == 0:
        raise ValueError("empty audio")
    return features.hcqt(y, sr)


@functools.lru_cache(maxsize=1)
def _cached(name: str, device: str) -> MultiF0:
    return MultiF0.from_pretrained(name, device)


def predict(
    audio: AudioInput,
    sr: int | None = None,
    model: str = DEFAULT_MODEL,
    device: DeviceLike = "auto",
    threshold: float | None = None,
) -> Prediction:
    """One-shot `MultiF0.predict`. Keeps the last model loaded; `clear_cache()` frees it."""
    pretrained = get_pretrained(model)
    return _cached(pretrained.name, str(resolve_device(device))).predict(audio, sr, threshold)


def clear_cache() -> None:
    """Free the model kept by `predict`."""
    _cached.cache_clear()
    empty_cache(resolve_device("auto"))
