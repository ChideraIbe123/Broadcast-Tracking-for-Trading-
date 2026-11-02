"""Device selection: cuda -> mps -> cpu. Every model in hoops takes a `device` string."""
from __future__ import annotations

import os


def pick_device(preferred: str | None = None) -> str:
    """Return the best available torch device name.

    `preferred` (or env HOOPS_DEVICE) forces a device when set to cuda / mps / cpu.
    """
    import torch

    want = preferred or os.environ.get("HOOPS_DEVICE")
    if want:
        want = want.lower()
        if want.startswith("cuda") and torch.cuda.is_available():
            return want
        if want == "mps" and torch.backends.mps.is_available():
            return "mps"
        if want == "cpu":
            return "cpu"
        # fall through to auto when the forced device is unavailable
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_summary() -> str:
    import torch

    dev = pick_device()
    if dev.startswith("cuda"):
        return f"cuda: {torch.cuda.get_device_name()}"
    return f"{dev} (torch {torch.__version__})"
