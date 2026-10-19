"""Network architectures of the pretrained models, and the registry of their weights.

Inputs are HCQT magnitude and phase differentials, each (batch, 5 harmonics, 360 bins, frames);
the output is the pitch salience, (batch, 360 bins, frames), in [0, 1].

Modules are named after the original Keras layers: each convolution keeps its Keras name, and
each batch norm is named after the layer it normalizes ("mag" and "dphase" for the inputs).
`tools/convert_weights.py` relies on this to map the Keras weights.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

WEIGHTS_DIR = Path(__file__).parent / "weights"

# Keras' default batch norm epsilon, which the models were trained with
BN_EPS = 1e-3

# Most elements a CPU convolution may unfold at once (512 MB of float32). Without oneDNN (e.g. on
# Apple Silicon), torch's CPU convolution unfolds its whole input (im2col): for the 360x1 kernel
# that is 33 MB per frame, so a minute of audio would need ~170 GB.
CPU_UNFOLD_BUDGET = 2**27


class SameConv2d(nn.Conv2d):
    """Conv2d with Keras' "same" padding: for an even kernel, the extra row/column goes after.

    On CPU, wide inputs are convolved in overlapping slices along the last (time) axis to bound
    memory; the result is the same.
    """

    def forward(self, x: Tensor) -> Tensor:
        kh, kw = self.kernel_size
        x = F.pad(x, ((kw - 1) // 2, kw // 2, (kh - 1) // 2, kh // 2))
        n_out = x.shape[-1] - kw + 1
        per_frame = self.in_channels * kh * kw * (x.shape[-2] - kh + 1)
        step = max(1, CPU_UNFOLD_BUDGET // per_frame)
        if x.device.type != "cpu" or step >= n_out:
            return super().forward(x)
        slices = [super().forward(x[..., s : s + step + kw - 1]) for s in range(0, n_out, step)]
        return torch.cat(slices, dim=-1)


class SalienceNet(nn.Module):
    """Shared building blocks: named convolutions and batch norms, and the common head."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleDict()
        self.norms = nn.ModuleDict()

    def _add_conv(self, name: str, in_ch: int, out_ch: int, kernel: tuple[int, int]) -> None:
        self.convs[name] = SameConv2d(in_ch, out_ch, kernel)
        self.norms[name] = nn.BatchNorm2d(out_ch, eps=BN_EPS)

    def _add_input_norms(self) -> None:
        for name in ("mag", "dphase"):
            self.norms[name] = nn.BatchNorm2d(5, eps=BN_EPS)

    def _add_head(self, in_ch: int, deep: bool) -> None:
        if deep:
            self._add_conv("conv7", in_ch, 64, (3, 3))
            self._add_conv("conv8", 64, 64, (3, 3))
            in_ch = 64
        self._add_conv("distribution", in_ch, 8, (360, 1))
        self.convs["squishy"] = SameConv2d(8, 1, (1, 1))

    def _block(self, name: str, x: Tensor) -> Tensor:
        """Conv, ReLU, then batch norm, as in the original Keras models."""
        out: Tensor = self.norms[name](F.relu(self.convs[name](x)))
        return out

    def _head(self, x: Tensor) -> Tensor:
        for name in ("conv7", "conv8", "distribution"):
            if name in self.convs:
                x = self._block(name, x)
        return torch.sigmoid(self.convs["squishy"](x)).squeeze(1)


class EarlyFusion(SalienceNet):
    """Early/Deep (`deep=True`) and Early/Shallow: magnitude and phase merge after conv1."""

    def __init__(self, deep: bool = True) -> None:
        super().__init__()
        self._add_input_norms()
        self._add_conv("conv1a", 5, 16, (5, 5))
        self._add_conv("conv1b", 5, 16, (5, 5))
        self._add_conv("conv2", 32, 32, (5, 5))
        self._add_conv("conv3", 32, 32, (5, 5))
        self._add_conv("conv4", 32, 32, (5, 5))
        self._add_conv("harm1", 32, 32, (70, 3))
        self._add_conv("harm2", 32, 32, (70, 3))
        self._add_head(32, deep)

    def forward(self, mag: Tensor, dphase: Tensor) -> Tensor:
        a = self._block("conv1a", self.norms["mag"](mag))
        b = self._block("conv1b", self.norms["dphase"](dphase))
        x = torch.cat([a, b], dim=1)
        for name in ("conv2", "conv3", "conv4", "harm1", "harm2"):
            x = self._block(name, x)
        return self._head(x)


class LateFusion(SalienceNet):
    """Late/Deep: magnitude and phase each go through their own conv1-harm2 stack, then merge."""

    _STACK = (("conv1", 16, (5, 5)), ("conv2", 32, (5, 5)), ("conv3", 32, (5, 5)),
              ("conv4", 32, (5, 5)), ("harm1", 32, (70, 3)), ("harm2", 32, (70, 3)))  # fmt: skip

    def __init__(self) -> None:
        super().__init__()
        self._add_input_norms()
        for branch in "ab":
            in_ch = 5
            for name, out_ch, kernel in self._STACK:
                self._add_conv(name + branch, in_ch, out_ch, kernel)
                in_ch = out_ch
        self._add_head(64, deep=True)

    def _stack(self, x: Tensor, branch: str) -> Tensor:
        for name, _, _ in self._STACK:
            x = self._block(name + branch, x)
        return x

    def forward(self, mag: Tensor, dphase: Tensor) -> Tensor:
        a = self._stack(self.norms["mag"](mag), "a")
        b = self._stack(self.norms["dphase"](dphase), "b")
        return self._head(torch.cat([a, b], dim=1))


ARCHITECTURES: dict[str, Callable[[], SalienceNet]] = {
    "early-deep": lambda: EarlyFusion(deep=True),
    "early-shallow": lambda: EarlyFusion(deep=False),
    "late-deep": LateFusion,
}


@dataclass(frozen=True)
class Pretrained:
    """A pretrained model: its architecture, weights file and tuned peak threshold."""

    name: str
    alias: str
    architecture: str
    weights: str
    threshold: float
    description: str

    def build(self) -> SalienceNet:
        """The network, with its weights loaded, in eval mode, on CPU."""
        from safetensors.torch import load_file

        net = ARCHITECTURES[self.architecture]()
        net.load_state_dict(load_file(WEIGHTS_DIR / self.weights))
        return net.eval()


PRETRAINED: dict[str, Pretrained] = {
    p.name: p
    for p in [
        Pretrained("early-deep", "model1", "early-deep", "exp1multif0.safetensors", 0.4,
                   "Early/Deep"),
        Pretrained("early-shallow", "model2", "early-shallow", "exp2multif0.safetensors", 0.5,
                   "Early/Shallow"),
        Pretrained("late-deep", "model3", "late-deep", "exp3multif0.safetensors", 0.5,
                   "Late/Deep (recommended)"),
        Pretrained("late-deep-no-bsq", "model4", "late-deep", "exp4multif0.safetensors", 0.4,
                   "Late/Deep, trained without BSQ (paper experiment 2)"),
        Pretrained("late-deep-no-reverb", "model5", "late-deep", "exp5multif0.safetensors", 0.5,
                   "Late/Deep, trained without the reverb files (paper experiment 3)"),
    ]
}  # fmt: skip

DEFAULT_MODEL = "late-deep"


def get_pretrained(name: str) -> Pretrained:
    """Look up a pretrained model by name ("late-deep") or by its original ID ("model3")."""
    for p in PRETRAINED.values():
        if name in (p.name, p.alias):
            return p
    choices = ", ".join(f"{p.name} ({p.alias})" for p in PRETRAINED.values())
    raise ValueError(f"unknown model {name!r}; choose one of: {choices}")
