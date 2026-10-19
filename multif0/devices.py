"""Choosing a torch device, and returning its cached memory."""

import gc

import torch

DeviceLike = str | torch.device


def resolve_device(device: DeviceLike = "auto") -> torch.device:
    """`device` as a `torch.device`; "auto" picks CUDA, then Apple Silicon (MPS), then CPU."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def empty_cache(device: torch.device) -> None:
    """Return memory cached by torch's allocator on `device` to the system (no-op on CPU)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
