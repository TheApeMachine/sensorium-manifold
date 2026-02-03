"""Unified manifold physics backend router.

This module provides a stable import path that chooses the best backend based
on the requested device:
- MPS  -> `optimizer.metal.manifold_physics`
- CUDA -> `optimizer.triton.manifold_physics`
"""

from __future__ import annotations

from typing import Literal, overload


@overload
def get_backend(device: Literal["mps"]) -> Literal["metal"]: ...


@overload
def get_backend(device: Literal["cuda"]) -> Literal["triton"]: ...


def get_backend(device: str) -> str:
    if device == "mps":
        return "metal"
    if device == "cuda":
        return "triton"
    raise RuntimeError(f"Unsupported device for manifold physics: {device!r} (expected 'mps' or 'cuda')")


def _load(device: str):
    b = get_backend(device)
    if b == "metal":
        from optimizer.metal.manifold_physics import (  # type: ignore[import-not-found]
            ManifoldPhysics,
            ManifoldPhysicsConfig,
            SpectralCarrierPhysics,
            SpectralCarrierConfig,
            ParticleGenerator,
            manifold_physics_available,
        )

        return (
            ManifoldPhysics,
            ManifoldPhysicsConfig,
            SpectralCarrierPhysics,
            SpectralCarrierConfig,
            ParticleGenerator,
            manifold_physics_available,
        )

    from optimizer.triton.manifold_physics import (  # type: ignore[import-not-found]
        ManifoldPhysics,
        ManifoldPhysicsConfig,
        SpectralCarrierPhysics,
        SpectralCarrierConfig,
    )

    def manifold_physics_available() -> bool:
        # CUDA backend availability is equivalent to torch.cuda.is_available()
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    ParticleGenerator = None  # CUDA path: generator remains torch-based upstream
    return (
        ManifoldPhysics,
        ManifoldPhysicsConfig,
        SpectralCarrierPhysics,
        SpectralCarrierConfig,
        ParticleGenerator,
        manifold_physics_available,
    )


def ManifoldPhysicsConfig(*args, device: str = "mps", **kwargs):  # type: ignore[misc]
    _, Cfg, *_ = _load(device)
    return Cfg(*args, **kwargs)


def SpectralCarrierConfig(*args, device: str = "mps", **kwargs):  # type: ignore[misc]
    *_, SCfg, _PG, _avail = _load(device)
    return SCfg(*args, **kwargs)


def manifold_physics_available(*, device: str = "mps") -> bool:
    *_, avail = _load(device)
    return bool(avail())


def ManifoldPhysics(*args, device: str = "mps", **kwargs):  # type: ignore[misc]
    MP, *_ = _load(device)
    return MP(*args, device=device, **kwargs)


def SpectralCarrierPhysics(*args, device: str = "mps", **kwargs):  # type: ignore[misc]
    _MP, _Cfg, SP, *_rest = _load(device)
    return SP(*args, device=device, **kwargs)


def ParticleGenerator(*args, device: str = "mps", **kwargs):  # type: ignore[misc]
    *_rest, PG, _avail = _load(device)
    if PG is None:
        raise RuntimeError("ParticleGenerator is only implemented for the Metal/MPS backend.")
    return PG(*args, device=device, **kwargs)

