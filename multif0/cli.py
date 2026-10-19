"""
Predict multiple F0 output from input audio files or folders.
This code is linked to the ISMIR paper:

Helena Cuesta, Brian McFee and Emilia Gómez (2020).
Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks.
In Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR).
Montreal, Canada (virtual).
"""

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from .core import MultiF0
from .models import DEFAULT_MODEL, PRETRAINED

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}


def find_audio_files(paths: Iterable[str | Path]) -> list[Path]:
    """Files as given, plus the audio files directly inside any folders."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def predict_file(
    model: MultiF0,
    audio_path: Path,
    output_dir: Path | None = None,
    threshold: float | None = None,
    save_salience: bool = False,
) -> Path:
    """Write <name>.csv (and optionally <name>_salience.npy) for one file; returns the CSV path."""
    out_dir = output_dir or audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = model.predict(audio_path, threshold=threshold)
    csv_path = out_dir / f"{audio_path.stem}.csv"
    pred.to_csv(csv_path)
    if save_salience:
        np.save(out_dir / f"{audio_path.stem}_salience.npy", pred.salience)
    return csv_path


def main(argv: Sequence[str] | None = None) -> None:
    models_help = "\n".join(
        f"  {p.name:<20} {p.alias}  {p.description}" for p in PRETRAINED.values()
    )
    parser = argparse.ArgumentParser(
        description="Predict multiple F0 output of audio files, or of all audio files in folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="models (by name or original ID):\n" + models_help,
    )
    parser.add_argument("inputs", nargs="+", help="audio files and/or folders of audio files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--threshold", type=float, help="peak threshold (default: the model's)")
    parser.add_argument("--output-dir", type=Path, help="default: next to each input file")
    parser.add_argument(
        "--save-salience", action="store_true", help="also save the salience as <name>_salience.npy"
    )
    parser.add_argument(
        "--device", default="auto", help="auto (CUDA, then MPS, then CPU), cpu, cuda, cuda:1, mps"
    )
    args = parser.parse_args(argv)

    try:
        audio_files = find_audio_files(args.inputs)
    except FileNotFoundError as e:
        parser.error(f"no such file or folder: {e}")
    if not audio_files:
        parser.error("no audio files found")
    try:
        model = MultiF0.from_pretrained(args.model, args.device)
    except ValueError as e:
        parser.error(str(e))

    for audio_file in audio_files:
        csv_path = predict_file(
            model, audio_file, args.output_dir, args.threshold, args.save_salience
        )
        print(f"{audio_file} -> {csv_path}")


if __name__ == "__main__":
    main()
