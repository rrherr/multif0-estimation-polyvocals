"""
Predict multiple F0 output from input audio files or folders.
This code is linked to the ISMIR paper:

Helena Cuesta, Brian McFee and Emilia Gómez (2020).
Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks.
In Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR).
Montreal, Canada (virtual).
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import features
import models
import utils

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}
# Frames per model call (~58 s), to bound memory. Each chunk is padded with
# CONTEXT_FRAMES of real neighbouring input, which covers the networks' temporal
# receptive field (+/-12 frames), so the result equals a single pass over the
# whole file (which is what the original code effectively did).
CHUNK_FRAMES = 5000
CONTEXT_FRAMES = 16


def predict_salience(model, audio_path):
    """Return the model's pitch salience, shape (360 freq bins, n_frames), float32."""
    mag, dphase = features.compute_hcqt(audio_path)
    n_frames = mag.shape[1]
    chunks = []
    for start in range(0, n_frames, CHUNK_FRAMES):
        end = min(start + CHUNK_FRAMES, n_frames)
        lo, hi = max(0, start - CONTEXT_FRAMES), min(n_frames, end + CONTEXT_FRAMES)
        out = model([mag[np.newaxis, :, lo:hi], dphase[np.newaxis, :, lo:hi]], training=False)[0]
        chunks.append(np.asarray(out)[:, start - lo : end - lo])
    return np.hstack(chunks).astype(np.float32)


def predict_file(model, threshold, audio_path, output_dir=None, save_salience=False):
    """Run multiple F0 estimation on one file and write <name>.csv; returns the CSV path."""
    audio_path = Path(audio_path)
    out_dir = Path(output_dir) if output_dir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    salience = predict_salience(model, audio_path)
    times, freqs = utils.pitch_activations_to_mf0(salience, threshold)

    csv_path = out_dir / f"{audio_path.stem}.csv"
    utils.save_multif0_output(times, freqs, csv_path)
    if save_salience:
        np.save(out_dir / f"{audio_path.stem}_salience.npy", salience)
    return csv_path


def find_audio_files(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict multiple F0 output of audio files, or of all audio files inside folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="models:\n" + "\n".join(f"  {k}  {m.description}" for k, m in models.PRETRAINED.items()),
    )
    parser.add_argument("inputs", nargs="+", help="audio files and/or folders of audio files")
    parser.add_argument("--model", default="model3", choices=models.PRETRAINED, help="pretrained model (default: model3)")
    parser.add_argument("--threshold", type=float, help="peak threshold (default: the model's tuned value)")
    parser.add_argument("--output-dir", help="where to write outputs (default: next to each input file)")
    parser.add_argument("--save-salience", action="store_true", help="also save the raw salience as <name>_salience.npy")
    parser.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU (Metal) is available")
    args = parser.parse_args(argv)

    try:
        audio_files = find_audio_files(args.inputs)
    except FileNotFoundError as e:
        parser.error(f"no such file or folder: {e}")
    if not audio_files:
        parser.error("no audio files found")

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")

    pretrained = models.PRETRAINED[args.model]
    threshold = pretrained.threshold if args.threshold is None else args.threshold
    model = pretrained.load()

    for audio_file in audio_files:
        csv_path = predict_file(model, threshold, audio_file, args.output_dir, args.save_salience)
        print(f"{audio_file} -> {csv_path}")


if __name__ == "__main__":
    main()
