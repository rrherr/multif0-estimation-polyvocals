"""Generate tests/data/synth_satb.wav: a deterministic 4-voice harmonic test clip.

Each voice is a harmonic tone with light vibrato, singing two notes (one change
halfway through). Run with: uv run python tests/make_synth_fixture.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
DURATION = 4.0
# (first note, second note) in Hz for soprano, alto, tenor, bass
VOICES = [(523.25, 587.33), (392.00, 440.00), (329.63, 349.23), (130.81, 146.83)]
OUT_PATH = Path(__file__).parent / "data" / "synth_satb.wav"


def harmonic_voice(f0_track, n_harmonics=8, vibrato_hz=5.5, vibrato_cents=20):
    t = np.arange(len(f0_track)) / SR
    f0 = f0_track * 2 ** (vibrato_cents / 1200 * np.sin(2 * np.pi * vibrato_hz * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        voice += np.sin(h * phase) / h**1.5 * (h * f0 < SR / 2)
    return voice


def main():
    n = int(SR * DURATION)
    half = n // 2
    mix = np.zeros(n)
    for f_a, f_b in VOICES:
        track = np.concatenate([np.full(half, f_a), np.full(n - half, f_b)])
        mix += harmonic_voice(track)
    fade = np.minimum(1, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.02 * SR))
    mix = 0.8 * mix * fade / np.max(np.abs(mix))
    OUT_PATH.parent.mkdir(exist_ok=True)
    sf.write(OUT_PATH, mix, SR, subtype="PCM_16")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
