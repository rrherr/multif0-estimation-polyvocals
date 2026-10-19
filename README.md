# Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks

This repo contains the companion code for the ISMIR paper:

Helena Cuesta, Brian McFee and Emilia Gómez (2020). **Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural
Networks**. In _Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR)._ 
Montreal, Canada (virtual).

Please, cite the aforementioned paper if using this material.

**Note:** This project builds upon the **DeepSalience** project:

Rachel M. Bittner, Brian McFee, Justin Salamon, Peter Li, and Juan P. Bello "Deep Salience Representations 
for F0 Estimation in Polyphonic Music”, ISMIR 2017, Suzhou, China.

Parts of this code are taken/adapted from their scripts, publicly available in 
https://github.com/rabitt/ismir2017-deepsalience.




## About this fork

This fork keeps only **inference** with the pretrained models, ported to **PyTorch** and packaged as
an importable library, `multif0`, with a command-line tool. It runs on NVIDIA GPUs (CUDA), Apple
Silicon GPUs (MPS) or CPU, on Python 3.12 or newer. The training and experiment code (`experiments/`,
`bsh/`, `utils_train.py`) was removed; see the
[upstream repository](https://github.com/helenacuesta/multif0-estimation-polyvocals) for it.

The pretrained models were trained on HCQT features computed with pumpp 0.5 / librosa 0.7. Newer librosa
versions compute a noticeably different CQT, so [`multif0/features.py`](multif0/features.py) re-implements
that original feature pipeline (including the resampy 0.2.2 resampling filters in
`multif0/resample_filters/`). The original Keras weights were converted to safetensors
([`tools/convert_weights.py`](tools/convert_weights.py)). Outputs match the original TensorFlow 1.15 code
to within floating-point noise; see [Tests](#tests).

Audio is processed in ~5.8 s chunks with enough overlapping context that the result is identical to a
single pass over the whole input.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/rrherr/multif0-estimation-polyvocals.git
cd multif0-estimation-polyvocals
uv sync
```

or, to use it from another project, `uv add git+https://github.com/rrherr/multif0-estimation-polyvocals`.

The device is picked automatically: CUDA if available, then Apple Silicon (MPS), then CPU. PyPI's
`torch` wheels for Linux include CUDA; on Windows, install a CUDA build of torch from
[pytorch.org](https://pytorch.org/get-started/locally/). `--device` (or `device=` in the library)
overrides the choice.

## Predict

```
uv run multif0 [--model MODEL] [--output-dir DIR] [--save-salience] [--device DEVICE] INPUT [INPUT ...]
```

Each `INPUT` is an audio file (wav, flac, ogg, aiff, mp3) or a folder, in which case every audio file
in it is processed. For each file, a tab-separated `<name>.csv` is written next to the input (or to
`--output-dir`): one row per frame (11.6 ms hop) with the time in seconds followed by zero or more
F0 estimates in Hz. `--save-salience` also writes the raw model output (360 frequency bins × frames)
as `<name>_salience.npy`.

| `--model` | Original ID | Weights | Architecture | Training data | Threshold |
|---|---|---|---|---|---|
| `early-deep` | `model1` | `exp1multif0` | Early/Deep | default splits | 0.4 |
| `early-shallow` | `model2` | `exp2multif0` | Early/Shallow | default splits | 0.5 |
| `late-deep` (default, recommended) | `model3` | `exp3multif0` | Late/Deep | default splits | 0.5 |
| `late-deep-no-bsq` | `model4` | `exp4multif0` | Late/Deep | without BSQ (paper experiment 2) | 0.4 |
| `late-deep-no-reverb` | `model5` | `exp5multif0` | Late/Deep | without reverb files (paper experiment 3) | 0.5 |

Models can be named either way. `--threshold` overrides the peak-picking threshold. Examples:

```
uv run multif0 poly_sing.wav
uv run multif0 --model early-deep --device cpu ../data/vocal_music
```

`uv run predict_on_audio.py ...` is the same command.

## Use as a library

```python
import multif0

model = multif0.MultiF0.from_pretrained("late-deep")   # device="auto", "cpu", "cuda", "mps", ...
pred = model.predict("choir.wav")                       # a file, or samples: model.predict(y, sr=44100)
pred.times                                              # (frames,) seconds
pred.freqs                                              # per frame, an array of F0 estimates in Hz
pred.salience                                           # (360 bins, frames) float32 in [0, 1]
pred.to_csv("choir.csv")                                # the CLI's output format

salience = model.salience(y, sr=sr)                     # just the salience
model.release()                                         # free its GPU memory (or `with ... as model:`)

multif0.predict("choir.wav", model="early-deep")        # one-shot; keeps the last model loaded
multif0.clear_cache()                                   # until this
```

In-memory audio is a NumPy array or torch tensor, mono `(n,)` or multichannel `(channels, n)` (averaged
to mono), at any sample rate `sr`; it is resampled to 22050 Hz. Salience rows are
`pred.freq_grid` (20-cent bins from 32.7 Hz), columns `pred.times` (hop 256 samples at 22050 Hz,
`pred.hop` = 11.6 ms); `pred.threshold` is the threshold used for `freqs`.

The networks are plain `torch.nn.Module`s (`multif0.EarlyFusion`, `multif0.LateFusion`; `model.net` for
a loaded one), taking HCQT features from `multif0.features.hcqt(y, sr)`.

## Tests

```
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

`tests/test_parity.py` runs every model on a synthetic four-voice clip (`tests/data/synth_satb.wav`,
generated by `tests/make_synth_fixture.py`) and compares the features, the salience and the F0 output
with reference outputs of the original code (TensorFlow 1.15.5, Keras 2.3.1, librosa 0.7.2, pumpp 0.5.0)
stored in `tests/data/synth_satb_ref.npz`. `tests/test_api.py` covers the library API, and checks that
CUDA and MPS (where available) match the CPU.

On that clip, the PyTorch models' salience is within 2e-5 of the TensorFlow versions on CPU and MPS.
Against the original code on real recordings (a 3-minute solo vocal and 1- and 3.7-minute two- and
three-voice mixes), the TensorFlow 2 version this port was checked against gave a maximum salience
difference below 0.004 for all five models, and identical F0 output in every frame.

On an M5 Pro, `late-deep` inference on 3 minutes of audio takes ~7 s on the GPU (MPS) and ~50 s on
CPU; feature extraction, which runs on CPU, takes about 6 s (from 44.1 kHz). CUDA has not been tested.
