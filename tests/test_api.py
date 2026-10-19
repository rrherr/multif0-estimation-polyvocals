"""The library API: inputs, outputs, devices and memory release."""

import librosa
import numpy as np
import pytest
import torch
from conftest import AUDIO

import multif0
from multif0 import devices, utils

ACCELERATORS = [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
    pytest.param(
        "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
    ),
]


@pytest.fixture(scope="module")
def cpu_model():
    return multif0.MultiF0.from_pretrained(device="cpu")


@pytest.fixture(scope="module")
def audio():
    return librosa.load(AUDIO, sr=None, mono=True)


@pytest.fixture(scope="module")
def cpu_salience(cpu_model):
    return cpu_model.salience(AUDIO)


def test_array_and_tensor_inputs_match_file(cpu_model, audio, cpu_salience):
    y, sr = audio
    np.testing.assert_array_equal(cpu_model.salience(y, sr=sr), cpu_salience)
    np.testing.assert_array_equal(cpu_model.salience(torch.from_numpy(y), sr=sr), cpu_salience)


def test_multichannel_input_is_downmixed(cpu_model, audio, cpu_salience):
    y, sr = audio
    stereo = np.stack([y * 0.5, y * 1.5])  # mean is y
    np.testing.assert_allclose(cpu_model.salience(stereo, sr=sr), cpu_salience, atol=1e-6)


def test_resampled_input(cpu_model, audio):
    y, sr = audio
    y44 = librosa.resample(y, orig_sr=sr, target_sr=44100)
    salience = cpu_model.salience(y44, sr=44100)
    assert salience.shape[0] == multif0.N_BINS
    assert abs(salience.shape[1] - (len(y) // multif0.HOP_LENGTH + 1)) <= 1


def test_sr_rules(cpu_model, audio):
    y, sr = audio
    with pytest.raises(ValueError, match="sr is required"):
        cpu_model.salience(y)
    with pytest.raises(ValueError, match="only for in-memory"):
        cpu_model.salience(AUDIO, sr=sr)


def test_prediction(cpu_model, cpu_salience, tmp_path):
    pred = cpu_model.predict(AUDIO)
    np.testing.assert_array_equal(pred.salience, cpu_salience)
    assert pred.threshold == 0.5
    assert len(pred.times) == len(pred.freqs) == pred.salience.shape[1]
    assert pred.hop == multif0.HOP_LENGTH / multif0.SR
    np.testing.assert_array_equal(pred.freq_grid, utils.get_freq_grid())
    assert max(len(f) for f in pred.freqs) >= 4  # four voices

    lower = cpu_model.predict(AUDIO, threshold=0.1)
    assert sum(map(len, lower.freqs)) > sum(map(len, pred.freqs))

    pred.to_csv(tmp_path / "out.csv")
    assert len((tmp_path / "out.csv").read_text().splitlines()) == len(pred.times)


def test_model_names_and_aliases():
    assert multif0.MultiF0.from_pretrained("model3", "cpu").pretrained.name == "late-deep"
    with pytest.raises(ValueError, match="unknown model"):
        multif0.MultiF0.from_pretrained("model7")


def test_empty_input(cpu_model):
    with pytest.raises(ValueError, match="empty audio"):
        cpu_model.salience(np.zeros(0, dtype=np.float32), sr=multif0.SR)


def test_release(audio):
    y, sr = audio
    with multif0.MultiF0.from_pretrained(device="cpu") as model:
        model.salience(y[:sr], sr=sr)
    assert model.net is None
    with pytest.raises(RuntimeError, match="released"):
        model.salience(y, sr=sr)


def test_one_shot_predict_caches_the_model(audio):
    y, sr = audio
    multif0.clear_cache()
    a = multif0.predict(y, sr=sr, device="cpu")
    b = multif0.predict(y, sr=sr, model="model3", device="cpu")
    assert multif0.core._cached.cache_info().hits == 1
    np.testing.assert_array_equal(a.salience, b.salience)
    multif0.clear_cache()
    assert multif0.core._cached.cache_info().currsize == 0


def test_auto_device_prefers_accelerators():
    expected = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert devices.resolve_device("auto").type == expected


@pytest.mark.parametrize("device", ACCELERATORS)
def test_accelerator_matches_cpu(device, cpu_salience):
    model = multif0.MultiF0.from_pretrained(device=device)
    assert model.device.type == device
    salience = model.salience(AUDIO)
    np.testing.assert_allclose(salience, cpu_salience, atol=1e-3)
    model.release()
