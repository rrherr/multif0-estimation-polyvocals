"""Post-processing of model output into multiple-F0 estimates.
Some of these functions are taken/adapted from deepsalience.
"""

import csv
import os
from collections.abc import Sequence
from typing import Any

import librosa
import numpy as np
import scipy.signal
from numpy.typing import NDArray

from . import features


def get_freq_grid() -> NDArray[np.float64]:
    """Frequencies (Hz) of the HCQT bins, i.e. of the rows of the model output."""
    return np.asarray(
        librosa.cqt_frequencies(
            features.N_BINS, fmin=features.FMIN, bins_per_octave=features.BINS_PER_OCTAVE
        ),
        dtype=np.float64,
    )


def get_time_grid(n_time_frames: int) -> NDArray[np.float64]:
    """Times (s) of the HCQT frames, i.e. of the columns of the model output."""
    return np.asarray(
        librosa.frames_to_time(
            np.arange(n_time_frames), sr=features.SR, hop_length=features.HOP_LENGTH
        ),
        dtype=np.float64,
    )


def pitch_activations_to_mf0(
    pitch_activation_mat: NDArray[np.floating[Any]], thresh: float
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Convert a pitch activation map to multif0 by thresholding peak values at thresh."""
    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs: list[list[float]] = [[] for _ in range(len(times))]
    for f, t in zip(idx[0], idx[1], strict=True):
        est_freqs[t].append(freqs[f])

    return times, [np.array(lst, dtype=np.float64) for lst in est_freqs]


def save_multif0_output(
    times: Sequence[float] | NDArray[np.float64],
    freqs: Sequence[Sequence[float] | NDArray[np.float64]],
    output_path: str | os.PathLike[str],
) -> None:
    """Save multif0 output as a tab-separated file: time, then zero or more frequencies per row."""
    with open(output_path, "w", newline="") as fhandle:
        csv_writer = csv.writer(fhandle, delimiter="\t")
        for t, f in zip(times, freqs, strict=True):
            csv_writer.writerow([float(t)] + [float(x) for x in f])
