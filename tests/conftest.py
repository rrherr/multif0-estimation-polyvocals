import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA = Path(__file__).parent / "data"
AUDIO = DATA / "synth_satb.wav"
