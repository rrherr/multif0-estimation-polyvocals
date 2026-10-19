"""Compare against reference outputs of the original code (TF 1.15.5, Keras 2.3.1,
librosa 0.7.2, pumpp 0.5.0) on tests/data/synth_satb.wav.

The remaining feature differences are floating-point noise; they only matter in
bins at the -80 dB floor, where the phase is numerically arbitrary.
"""

import csv

import numpy as np
import pytest
from conftest import AUDIO, DATA

import multif0
from multif0 import cli, core, features, models, utils


@pytest.fixture(scope="module")
def ref():
    data = np.load(DATA / "synth_satb_ref.npz")
    # pumpp layout (frames, bins, harmonics) -> model layout (bins, frames, harmonics)
    return {k: (v.transpose(1, 0, 2) if k in ("mag", "dphase") else v) for k, v in data.items()}


@pytest.fixture(scope="module")
def hcqt():
    return features.compute_hcqt(AUDIO)


@pytest.fixture(scope="module", params=[p.name for p in multif0.PRETRAINED.values()])
def model(request):
    return multif0.MultiF0.from_pretrained(request.param, device="cpu")


def test_features_match_reference(hcqt, ref):
    mag, dphase = hcqt
    assert mag.shape == ref["mag"].shape == dphase.shape
    assert np.abs(mag - ref["mag"]).max() < 0.01  # dB

    above_floor = ref["mag"] > -79
    wrapped = np.abs(np.angle(np.exp(1j * (dphase - ref["dphase"]))))
    assert wrapped[above_floor].max() < 0.01  # radians


def test_model_matches_reference_on_reference_features(model, ref):
    salience = model.salience_from_features(ref["mag"], ref["dphase"])
    np.testing.assert_allclose(salience, ref[f"saliency_{model.pretrained.alias}"], atol=1e-4)


def test_end_to_end_matches_reference(model, ref, hcqt):
    salience = model.salience_from_features(*hcqt)
    expected = ref[f"saliency_{model.pretrained.alias}"]
    assert salience.shape == expected.shape
    err = np.abs(salience - expected)
    assert err.max() < 0.05
    assert err.mean() < 1e-4

    _, freqs = utils.pitch_activations_to_mf0(salience, model.threshold)
    _, ref_freqs = utils.pitch_activations_to_mf0(expected, model.threshold)
    differing = sum(
        len(a) != len(b) or not np.allclose(a, b) for a, b in zip(freqs, ref_freqs, strict=True)
    )
    assert differing <= 0.01 * len(freqs)


def test_cli_finds_the_four_voices(tmp_path):
    from make_synth_fixture import DURATION, SR, VOICES

    cli.main([str(AUDIO), "--output-dir", str(tmp_path), "--save-salience", "--device", "cpu"])

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
    model = multif0.MultiF0.from_pretrained(device="cpu")
    monkeypatch.setattr(core, "CHUNK_FRAMES", 10_000)
    single = model.salience_from_features(*hcqt)
    monkeypatch.setattr(core, "CHUNK_FRAMES", 100)
    chunked = model.salience_from_features(*hcqt)
    np.testing.assert_allclose(chunked, single, atol=1e-4)  # conv algorithm noise


def test_sliced_cpu_convolution_matches_single_pass(hcqt, monkeypatch):
    model = multif0.MultiF0.from_pretrained(device="cpu")
    whole = model.salience_from_features(*hcqt)
    monkeypatch.setattr(models, "CPU_UNFOLD_BUDGET", 2**20)  # a few frames per slice
    sliced = model.salience_from_features(*hcqt)
    np.testing.assert_allclose(sliced, whole, atol=1e-4)
