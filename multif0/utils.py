"""Post-processing of model output into multiple-F0 estimates.
Some of these functions are taken/adapted from deepsalience.
"""

import csv

import librosa
import numpy as np
import scipy.signal

from . import features


def get_freq_grid():
    """Frequencies (Hz) of the HCQT bins, i.e. of the rows of the model output."""
    return librosa.cqt_frequencies(features.N_BINS, fmin=features.FMIN, bins_per_octave=features.BINS_PER_OCTAVE)


def get_time_grid(n_time_frames):
    """Times (s) of the HCQT frames, i.e. of the columns of the model output."""
    return librosa.frames_to_time(np.arange(n_time_frames), sr=features.SR, hop_length=features.HOP_LENGTH)


def pitch_activations_to_mf0(pitch_activation_mat, thresh):
    """Convert a pitch activation map to multif0 by thresholding peak values at thresh."""
    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(times))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freqs[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    return times, est_freqs


def save_multif0_output(times, freqs, output_path):
    """Save multif0 output as a tab-separated file: time, then zero or more frequencies per row."""
    with open(output_path, "w", newline="") as fhandle:
        csv_writer = csv.writer(fhandle, delimiter="\t")
        for t, f in zip(times, freqs):
            csv_writer.writerow([float(t)] + [float(x) for x in f])
