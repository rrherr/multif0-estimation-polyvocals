"""Compare against reference outputs of the original code (TF 1.15.5, Keras 2.3.1,
librosa 0.7.2, pumpp 0.5.0) on tests/data/synth_satb.wav.

The remaining feature differences are floating-point noise; they only matter in
bins at the -80 dB floor, where the phase is numerically arbitrary.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

import multif0
from multif0 import features, models, utils
from multif0 import predict as predict_on_audio

DATA = Path(__file__).parent / "data"
AUDIO = DATA / "synth_satb.wav"


@pytest.fixture(scope="module")
def ref():
    data = np.load(DATA / "synth_satb_ref.npz")
    # pumpp layout (frames, bins, harmonics) -> model layout (bins, frames, harmonics)
    return {k: (v.transpose(1, 0, 2) if k in ("mag", "dphase") else v) for k, v in data.items()}


@pytest.fixture(scope="module")
def hcqt():
    return features.compute_hcqt(AUDIO)


@pytest.fixture(scope="module", params=list(models.PRETRAINED))
def pretrained(request):
    return request.param, models.PRETRAINED[request.param], models.PRETRAINED[request.param].load()


def test_features_match_reference(hcqt, ref):
    mag, dphase = hcqt
    assert mag.shape == ref["mag"].shape == dphase.shape
    assert np.abs(mag - ref["mag"]).max() < 0.01  # dB

    above_floor = ref["mag"] > -79
    wrapped = np.abs(np.angle(np.exp(1j * (dphase - ref["dphase"]))))
    assert wrapped[above_floor].max() < 0.01  # radians


def test_model_matches_reference_on_reference_features(pretrained, ref, monkeypatch):
    key, _, model = pretrained
    monkeypatch.setattr(features, "compute_hcqt", lambda _: (ref["mag"], ref["dphase"]))
    salience = predict_on_audio.predict_salience(model, AUDIO)
    np.testing.assert_allclose(salience, ref[f"saliency_{key}"], atol=1e-4)


def test_end_to_end_matches_reference(pretrained, ref, hcqt, monkeypatch):
    key, pm, model = pretrained
    monkeypatch.setattr(features, "compute_hcqt", lambda _: hcqt)
    salience = predict_on_audio.predict_salience(model, AUDIO)
    expected = ref[f"saliency_{key}"]
    assert salience.shape == expected.shape
    err = np.abs(salience - expected)
    assert err.max() < 0.05
    assert err.mean() < 1e-4

    _, freqs = utils.pitch_activations_to_mf0(salience, pm.threshold)
    _, ref_freqs = utils.pitch_activations_to_mf0(expected, pm.threshold)
    differing = sum(len(a) != len(b) or not np.allclose(a, b) for a, b in zip(freqs, ref_freqs))
    assert differing <= 0.01 * len(freqs)


def test_cli_finds_the_four_voices(tmp_path):
    from make_synth_fixture import SR, DURATION, VOICES

    predict_on_audio.main([str(AUDIO), "--output-dir", str(tmp_path), "--save-salience"])

    with open(tmp_path / "synth_satb.csv", newline="") as f:
        rows = [[float(x) for x in row] for row in csv.reader(f, delimiter="\t")]
    salience = np.load(tmp_path / "synth_satb_salience.npy")
    assert len(rows) == salience.shape[1] == int(SR * DURATION / features.HOP_LENGTH)

    # in the middle of each note, every voice is detected within 50 cents in most frames
    # (a loose bar: the original model also misses the 2nd soprano note in ~15% of frames)
    for note, (start, end) in enumerate([(0.5, 1.5), (2.5, 3.5)]):
        frames = [r[1:] for r in rows if start <= r[0] <= end]
        for voice in VOICES:
            f0 = voice[note]
            hits = [any(abs(1200 * np.log2(f / f0)) < 50 for f in fr) for fr in frames]
            assert np.mean(hits) > 0.75, f"voice {f0} Hz detected in {np.mean(hits):.0%} of frames"


def test_chunking_matches_single_pass(hcqt, monkeypatch):
    model = models.PRETRAINED["model3"].load()
    monkeypatch.setattr(features, "compute_hcqt", lambda _: hcqt)
    single = predict_on_audio.predict_salience(model, AUDIO)
    monkeypatch.setattr(predict_on_audio, "CHUNK_FRAMES", 100)
    chunked = predict_on_audio.predict_salience(model, AUDIO)
    np.testing.assert_allclose(chunked, single, atol=1e-5)


def test_array_entry_point_matches_file(hcqt):
    import librosa

    model = models.PRETRAINED["model3"].load()
    y, sr = librosa.load(AUDIO, sr=None, mono=True)
    from_file = predict_on_audio.salience_from_features(model, *hcqt)
    np.testing.assert_array_equal(multif0.salience(model, y, sr), from_file)
