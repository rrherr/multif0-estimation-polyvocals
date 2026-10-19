"""HCQT magnitude and phase-differential input features.

The pretrained models were trained on features computed by pumpp 0.5's
``HCQTPhaseDiff`` on top of librosa 0.7's CQT. librosa 0.9 rewrote the CQT
(filter construction and multirate downsampling), which changes the features by
up to ~20 dB and visibly degrades the models' output. To keep the pretrained
models valid, this module re-implements that exact pipeline on modern numpy and
librosa primitives.

The CQT code below is adapted from librosa 0.7.2 (ISC license,
Copyright (c) 2013--2017, librosa development team); the phase-differential
code from pumpp 0.5.0 and the resampler (with its filter tables in
resample_filters/) from resampy 0.2.2 (both ISC license, Copyright (c) 2016,
Brian McFee).
"""

import functools
from pathlib import Path

import librosa
import numba
import numpy as np
import scipy.sparse

_FILTER_DIR = Path(__file__).parent / "resample_filters"

SR = 22050
HOP_LENGTH = 256
FMIN = 32.7
N_OCTAVES = 6
OVER_SAMPLE = 5
BINS_PER_OCTAVE = 12 * OVER_SAMPLE
N_BINS = N_OCTAVES * BINS_PER_OCTAVE
HARMONICS = (1, 2, 3, 4, 5)

# librosa 0.7: pass-band of resampy's kaiser_fast filter, as a fraction of Nyquist
_BW_FASTEST = 0.85
_WINDOW = "hann"


def _sparsify_rows(x, quantile=0.01):
    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)
    mag_sort = np.sort(mags, axis=1)
    cumulative_mag = np.cumsum(mag_sort / norms, axis=1)
    threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)
    x_sparse = scipy.sparse.lil_matrix(x.shape, dtype=x.dtype)
    for i, j in enumerate(threshold_idx):
        idx = np.where(mags[i] >= mag_sort[i, j])
        x_sparse[i, idx] = x[i, idx]
    return x_sparse.tocsr()


def _constant_q_lengths(sr, fmin, n_bins, bins_per_octave):
    q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1)
    freq = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))
    if freq[-1] * (1 + 0.5 * librosa.filters.window_bandwidth(_WINDOW) / q) > sr / 2.0:
        raise ValueError("Filter pass-band lies beyond Nyquist")
    return q * sr / freq


def _cqt_filter_fft(sr, fmin, n_bins, bins_per_octave, sparsity=0.01):
    lengths = _constant_q_lengths(sr, fmin, n_bins, bins_per_octave)
    q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1)
    freqs = q * sr / lengths

    filters = []
    for ilen, freq in zip(lengths, freqs):
        sig = np.exp(np.arange(-ilen // 2, ilen // 2, dtype=float) * 1j * 2 * np.pi * freq / sr)
        sig = sig * librosa.filters.get_window(_WINDOW, len(sig))
        filters.append(sig / np.sum(np.abs(sig)))

    n_fft = int(2.0 ** np.ceil(np.log2(max(lengths))))
    basis = np.asarray([librosa.util.pad_center(f, size=n_fft) for f in filters], dtype=np.complex64)
    basis *= lengths[:, np.newaxis] / float(n_fft)
    fft_basis = np.fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]
    return _sparsify_rows(fft_basis, quantile=sparsity), n_fft


def _cqt_response(y, n_fft, hop_length, fft_basis):
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="ones", pad_mode="reflect")
    return fft_basis.dot(stft)


def _num_two_factors(x):
    num_twos = 0
    while x > 0 and x % 2 == 0:
        num_twos += 1
        x //= 2
    return num_twos


@functools.cache
def _load_filter(name):
    data = np.load(_FILTER_DIR / f"{name}.npz")
    return data["half_window"], int(data["precision"])


@numba.jit(nopython=True, nogil=True, cache=True)
def _resample_f(x, y, sample_ratio, interp_win, interp_delta, num_table):
    scale = min(1.0, sample_ratio)
    time_increment = 1.0 / sample_ratio
    index_step = int(scale * num_table)
    time_register = 0.0

    nwin = interp_win.shape[0]
    n_orig = x.shape[0]

    for t in range(y.shape[0]):
        n = int(time_register)

        # left wing of the filter response
        frac = scale * (time_register - n)
        index_frac = frac * num_table
        offset = int(index_frac)
        eta = index_frac - offset
        for i in range(min(n + 1, (nwin - offset) // index_step)):
            weight = interp_win[offset + i * index_step] + eta * interp_delta[offset + i * index_step]
            y[t] += weight * x[n - i]

        # right wing of the filter response
        frac = scale - frac
        index_frac = frac * num_table
        offset = int(index_frac)
        eta = index_frac - offset
        for k in range(min(n_orig - n - 1, (nwin - offset) // index_step)):
            weight = interp_win[offset + k * index_step] + eta * interp_delta[offset + k * index_step]
            y[t] += weight * x[n + k + 1]

        time_register += time_increment


def _resample(y, orig_sr, target_sr, res_type, scale=False):
    """Band-limited sinc resampling, matching librosa 0.7.2 + resampy 0.2.2.

    resampy >= 0.3 regenerated its filter tables, so its output differs.
    """
    if orig_sr == target_sr:
        return y
    sample_ratio = float(target_sr) / orig_sr
    interp_win, precision = _load_filter(res_type)
    if sample_ratio < 1:
        interp_win = interp_win * sample_ratio
    interp_delta = np.zeros_like(interp_win)
    interp_delta[:-1] = np.diff(interp_win)

    y_hat = np.zeros(int(len(y) * sample_ratio), dtype=y.dtype)
    _resample_f(y, y_hat, sample_ratio, interp_win, interp_delta, precision)

    y_hat = librosa.util.fix_length(y_hat, size=int(np.ceil(len(y) * sample_ratio)))
    if scale:
        y_hat /= np.sqrt(sample_ratio)
    return y_hat


def cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave):
    """Complex CQT, numerically matching ``librosa.cqt`` from librosa 0.7.2."""
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    freqs = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)[-bins_per_octave:]
    fmin_t, fmax_t = np.min(freqs), np.max(freqs)
    q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1)
    bandwidth = librosa.filters.window_bandwidth(_WINDOW)
    filter_cutoff = fmax_t * (1 + 0.5 * bandwidth / q)
    nyquist = sr / 2.0
    res_type = "kaiser_fast" if filter_cutoff < _BW_FASTEST * nyquist else "kaiser_best"

    # early downsampling
    downsample_count1 = max(0, int(np.ceil(np.log2(_BW_FASTEST * nyquist / filter_cutoff)) - 1) - 1)
    downsample_count2 = max(0, _num_two_factors(hop_length) - n_octaves + 1)
    downsample_count = min(downsample_count1, downsample_count2)
    if downsample_count > 0 and res_type == "kaiser_fast":
        factor = 2 ** downsample_count
        hop_length //= factor
        new_sr = sr / float(factor)
        y = _resample(y, sr, new_sr, res_type, scale=True)
        sr = new_sr

    cqt_resp = []
    if res_type != "kaiser_fast":
        # top octave at full rate with the best filter, the rest with kaiser_fast
        fft_basis, n_fft = _cqt_filter_fft(sr, fmin_t, n_filters, bins_per_octave)
        cqt_resp.append(_cqt_response(y, n_fft, hop_length, fft_basis))
        fmin_t /= 2
        n_octaves -= 1
        res_type = "kaiser_fast"

    if _num_two_factors(hop_length) < n_octaves - 1:
        raise ValueError(f"hop_length must be a multiple of 2^{n_octaves - 1}")

    fft_basis, n_fft = _cqt_filter_fft(sr, fmin_t, n_filters, bins_per_octave)
    my_y, my_hop = y, hop_length
    for i in range(n_octaves):
        if i > 0:
            if len(my_y) < 2:
                raise ValueError("Input signal is too short for the CQT")
            my_y = _resample(my_y, 2, 1, res_type, scale=True)
            fft_basis = fft_basis * np.sqrt(2)
            my_hop //= 2
        cqt_resp.append(_cqt_response(my_y, n_fft, my_hop, fft_basis))

    max_col = min(x.shape[1] for x in cqt_resp)
    C = np.vstack([x[:, :max_col] for x in cqt_resp][::-1])[-n_bins:]

    # sr here is the (possibly early-downsampled) rate, as in librosa 0.7.2
    lengths = _constant_q_lengths(sr, fmin, n_bins, bins_per_octave)
    return C / np.sqrt(lengths[:, np.newaxis])


def compute_hcqt(audio_path):
    """Compute HCQT log-magnitude and phase differentials for an audio file.

    Returns ``(mag, dphase)``, each float32 of shape (n_bins, n_frames, n_harmonics),
    i.e. the layout the models take as input (minus the batch axis).
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return hcqt(y, sr)


def hcqt(y, sr):
    """`compute_hcqt` for mono audio `y` already in memory, at sample rate `sr`.

    `y` is resampled to `SR` with resampy 0.2.2's ``kaiser_best`` filter, as librosa 0.7
    did when the models were trained.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected mono audio (1-D), got shape {y.shape}")
    if sr != SR:
        y = _resample(y, sr, SR, "kaiser_best")

    n_frames = int(librosa.time_to_frames(librosa.get_duration(y=y, sr=SR), sr=SR, hop_length=HOP_LENGTH))

    mags, phases = [], []
    for h in HARMONICS:
        C = cqt(y, SR, HOP_LENGTH, FMIN * h, N_BINS, BINS_PER_OCTAVE)
        C = librosa.util.fix_length(C, size=n_frames)
        C, P = librosa.magphase(C)
        mags.append(librosa.amplitude_to_db(C, ref=np.max))
        phases.append(np.angle(P))

    # (harmonics, bins, frames) -> (bins, frames, harmonics)
    mag = np.transpose(np.asarray(mags), (1, 2, 0)).astype(np.float32)
    phase = np.transpose(np.asarray(phases), (1, 2, 0))
    dphase = np.empty_like(phase)
    dphase[:, :1] = phase[:, :1]
    dphase[:, 1:] = np.diff(np.unwrap(phase, axis=1), axis=1)
    return mag, dphase.astype(np.float32)
