"""Backend availability detection (Triton + Metal/MPS)

The Sensorium Manifold can only realistically run on accelerated hardware.
Included in this code-base are fused kernels for Apple Silicon (Metal), and
Triton (CUDA) that implement the core physics and spectral operations.
Every attempt has been made to make these kernels both accurate, and performant,
in that exact order.

That being said, this is a research project that requires the flexibility for quick
turnarounds and exploratory experimentation. As such, these are at no point to be
considered as production-ready implementations.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
from typing import TYPE_CHECKING
import torch

__all__ = [
    "triton_supported",
    "metal_supported",
]


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False

def metal_build_tools_available() -> bool:
    """Whether the host can compile Metal shaders via Xcode toolchain.

    Notes:
    - Having `xcrun` in PATH is not sufficient; the active developer directory
      must contain the `metal` and `metallib` tools.
    - This function is intentionally conservative: if we can't *prove* the tools
      exist, we return False so training can surface a clear, actionable error.
    """
    # Do not call metal_supported() here: metal_supported() is a *runtime* check,
    # while this function is a *toolchain* check. Keeping them independent avoids
    # accidental recursion and makes errors actionable.
    if TYPE_CHECKING:
        return False

    if platform.system() != "Darwin":
        return False

    try:
        if not bool(torch.backends.mps.is_available()):
            return False
    except Exception:
        return False

    if shutil.which("xcrun") is None:
        return False

    try:
        subprocess.check_output(["xcrun", "-sdk", "macosx", "--find", "metal"], stderr=subprocess.STDOUT)
        subprocess.check_output(["xcrun", "-sdk", "macosx", "--find", "metallib"], stderr=subprocess.STDOUT)
    except Exception:
        return False
    return True

def triton_supported() -> bool:
    return bool(has_module("triton") and has_module("triton.language"))

def metal_supported() -> bool:
    """Whether the current runtime *can* execute custom Metal (MPS) ops.

    This indicates platform + PyTorch MPS support. It does NOT guarantee that the
    custom extension is already built/loaded; higher-level code may JIT build it.
    """
    if TYPE_CHECKING:
        return False

    if platform.system() != "Darwin":
        return False

    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False

def get_device() -> str:
    """Get the device to use for the physics simulation.
    
    This is a wrapper around the device detection logic.
    """
    if triton_supported():
        return "cuda"
    if metal_supported():
        return "mps"
    
    return "cpu"