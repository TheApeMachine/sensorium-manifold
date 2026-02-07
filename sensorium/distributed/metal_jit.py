from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from sensorium.kernels.runtime import metal_build_tools_available, metal_supported


_CACHED_MOD: Any | None = None
_CACHED_ERR: Exception | None = None


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


def _xcrun_find(tool: str) -> str:
    out = subprocess.check_output(
        ["xcrun", "-sdk", "macosx", "--find", str(tool)],
        stderr=subprocess.STDOUT,
    )
    p = out.decode("utf-8", errors="replace").strip()
    if not p:
        raise RuntimeError(f"xcrun returned empty path for tool {tool!r}")
    return p


def _compile_metallib(*, out_dir: Path, verbose: bool) -> Path:
    source = _this_dir() / "metal_distributed.metal"
    air = out_dir / "metal_distributed.air"
    metallib = out_dir / "distributed_ops.metallib"
    metal = _xcrun_find("metal")
    metallib_tool = _xcrun_find("metallib")

    if metallib.exists() and metallib.stat().st_mtime >= source.stat().st_mtime:
        return metallib

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [metal, "-c", str(source), "-o", str(air)]
    if verbose:
        print("[distributed] compiling Metal shader:", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"metal compile failed\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )

    cmd2 = [metallib_tool, str(air), "-o", str(metallib)]
    if verbose:
        print("[distributed] linking metallib:", " ".join(cmd2))
    p2 = subprocess.run(cmd2, capture_output=True, text=True)
    if p2.returncode != 0:
        raise RuntimeError(
            f"metallib link failed\nstdout:\n{p2.stdout}\nstderr:\n{p2.stderr}"
        )
    return metallib


def load_distributed_metal_ops(*, verbose: bool = False) -> Any:
    global _CACHED_MOD, _CACHED_ERR
    if _CACHED_MOD is not None:
        return _CACHED_MOD
    if _CACHED_ERR is not None:
        raise _CACHED_ERR
    if not metal_supported():
        err = RuntimeError("Metal/MPS is not supported on this runtime")
        _CACHED_ERR = err
        raise err
    if not metal_build_tools_available():
        err = RuntimeError("Metal build tools unavailable")
        _CACHED_ERR = err
        raise err

    import torch.utils.cpp_extension as ce

    try:
        name = "manifold_distributed_metal_ops"
        build_dir = Path(ce._get_build_directory(name, verbose=verbose))
        _compile_metallib(out_dir=build_dir, verbose=verbose)
        mod = ce.load(
            name=name,
            sources=[str(_this_dir() / "metal_ops.mm")],
            extra_cflags=["-O3", "-std=c++17", "-fobjc-arc", "-fblocks"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            with_cuda=False,
            is_python_module=True,
            build_directory=str(build_dir),
            verbose=verbose,
        )
    except Exception as e:
        _CACHED_ERR = e
        raise
    _CACHED_MOD = mod
    return mod
