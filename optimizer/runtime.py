"""Backend availability detection (Triton + Metal/MPS).

Caramba has optional fused kernels that depend on accelerator-specific toolchains:
- Triton (CUDA) for fused decode / SSM kernels
- Metal (MPS) for Apple Silicon fused DBA decode (custom MSL kernel + ObjC++ bridge)

This module centralizes runtime detection in a way that is safe for import + type
checking: at type-check time we force optional backends off.
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
