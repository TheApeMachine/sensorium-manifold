"""Hardware abstraction layer (HAL) for high-impact kernels.

Caramba operates on a strict kernel policy:
- Dispatch deterministically to the best supported kernel path.
- Validate required kernel backends at startup (see `optimizer/kernel_registry.py`).
- If an expected fast path is unavailable, raise immediately (no silent fallbacks).

Notes:
- Many CUDA code paths currently rely on PyTorch's native CUDA kernels for norms/RoPE.
  This module keeps the public API stable while dispatching to the best validated backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import torch
from torch import Tensor
import torch.nn.functional as F

from optimizer.kernel_registry import KERNELS


def _require(cond: bool, *, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


_F = TypeVar("_F", bound=Callable[..., object])


def _typed(fn: _F) -> _F:
    """Typed no-op decorator.

    We avoid `torch.compiler.disable` here because it causes graph breaks at
    every callsite (see `TORCH_LOGS=+dynamo`), which can materially reduce
    throughput (tok/s) and increase Dynamo resume complexity.
    """

    return cast(_F, fn)


def _use_custom_kernels() -> bool:
    """Whether to use Caramba's custom Triton/Metal kernels.

    Default is **off**. Set `CARAMBA_USE_CUSTOM_KERNELS=1` to re-enable.
    """
    import os

    return str(os.environ.get("CARAMBA_USE_CUSTOM_KERNELS", "0")).lower() in ("1", "true", "yes", "on")


# ---- Hugging Face Kernel Hub (kernels) optional integrations ----
_HF_RMSNORM: Any | None = None
_HF_RMSNORM_CHECKED: bool = False
_HF_RMSNORM_ERROR: Exception | None = None


def _hf_rmsnorm(*, x: Tensor, weight: Tensor, eps: float) -> Tensor | None:
    """Best-effort Kernel Hub RMSNorm.

    Uses `kernels-community/rmsnorm` when a compatible build exists for the
    current environment; otherwise returns None.
    """
    global _HF_RMSNORM, _HF_RMSNORM_CHECKED, _HF_RMSNORM_ERROR
    if not _HF_RMSNORM_CHECKED:
        _HF_RMSNORM_CHECKED = True
        try:
            import kernels as _kernels  # type: ignore[import-not-found]
            from kernels import has_kernel  # type: ignore[import-not-found]

            # Prefer an explicit version when available.
            # kernels' typing stubs currently annotate `version` as `str|None`,
            # but the docs allow ints. Use a string to satisfy type checkers.
            if has_kernel("kernels-community/rmsnorm", version="1") or has_kernel("kernels-community/rmsnorm"):
                _HF_RMSNORM = _kernels.get_kernel("kernels-community/rmsnorm", version="1")
        except Exception as e:
            _HF_RMSNORM_ERROR = e
            _HF_RMSNORM = None

    if _HF_RMSNORM is None:
        return None

    try:
        fn = getattr(_HF_RMSNORM, "apply_rms_norm", None)
        if callable(fn):
            out = fn(x, weight, float(eps))
            return out if isinstance(out, torch.Tensor) else None
    except Exception:
        # Kernel present but failed for this input/device; fall back.
        return None

    return None


# ---- Optional backend imports (top-level; errors raised on use) ----
_METAL_IMPORT_ERROR: Exception | None = None
_TRITON_IMPORT_ERROR: Exception | None = None

_rmsnorm_fp16: Any | None = None
_rope_fp16: Any | None = None
_layernorm_fp16: Any | None = None
_dba_decode_fp16: Any | None = None
_MetalSSMSelectiveScan: Any | None = None
_AdamWMasterStep: Any | None = None
_lion_fp16: Any | None = None

try:
    from optimizer.metal import (
        AdamWMasterStep as _AdamWMasterStep,
        MetalSSMSelectiveScan as _MetalSSMSelectiveScan,
        dba_decode_fp16 as _dba_decode_fp16,
        layernorm_fp16 as _layernorm_fp16,
        lion_fp16 as _lion_fp16,
        rmsnorm_fp16 as _rmsnorm_fp16,
        rope_fp16 as _rope_fp16,
    )
except Exception as e:
    _METAL_IMPORT_ERROR = e

_rmsnorm_triton: Any | None = None
_rope_triton: Any | None = None
_layernorm_triton: Any | None = None
_fused_selective_scan: Any | None = None
_adamw_triton_master_step: Any | None = None

# Triton optional kernels are not bundled in this repository snapshot.
# We keep the HAL API stable and fall back to PyTorch implementations unless
# explicitly enabled and present.
_TRITON_IMPORT_ERROR = ImportError(
    "Optional Triton kernels (rmsnorm/rope/layernorm/adamw/ssm) are not available in this build."
)


@_typed
def rmsnorm(*, x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight."""
    if weight is not None:
        y_hf = _hf_rmsnorm(x=x, weight=weight, eps=float(eps))
        if y_hf is not None:
            return y_hf

    if _use_custom_kernels() and x.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="RMSNorm on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.float32),
            msg=f"RMSNorm on MPS requires fp16/fp32, got dtype={x.dtype}.",
        )
        if _rmsnorm_fp16 is None:
            raise RuntimeError(f"Metal RMSNorm import failed: {_METAL_IMPORT_ERROR!r}")
        return _rmsnorm_fp16(x=x, weight=weight, eps=float(eps))

    if _use_custom_kernels() and x.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="RMSNorm on CUDA requires Triton to be available and validated at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.bfloat16),
            msg=f"RMSNorm on CUDA requires fp16/bf16, got dtype={x.dtype}.",
        )
        if _rmsnorm_triton is None:
            raise RuntimeError(f"Triton RMSNorm import failed: {_TRITON_IMPORT_ERROR!r}")
        return _rmsnorm_triton(x=x, weight=weight, eps=float(eps))

    x_f = x.float()
    inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    y = (x_f * inv_rms).to(dtype=x.dtype)
    if weight is not None:
        y = y * weight
    return y


@_typed
def rope_apply(*, x: Tensor, cos: Tensor, sin: Tensor, rot_dim: int) -> Tensor:
    """Apply RoPE using cos/sin tables for the sequence window.

    Expects:
    - x: (B, H, T, D)
    - cos/sin: (T, rot_dim/2)
    """
    if _use_custom_kernels() and x.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="RoPE on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.float32),
            msg=f"RoPE on MPS requires fp16/fp32, got dtype={x.dtype}.",
        )
        if _rope_fp16 is None:
            raise RuntimeError(f"Metal RoPE import failed: {_METAL_IMPORT_ERROR!r}")
        return _rope_fp16(x=x, cos=cos, sin=sin, rot_dim=int(rot_dim))

    if _use_custom_kernels() and x.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="RoPE on CUDA requires Triton to be available and validated at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.bfloat16),
            msg=f"RoPE on CUDA requires fp16/bf16, got dtype={x.dtype}.",
        )
        if _rope_triton is None:
            raise RuntimeError(f"Triton RoPE import failed: {_TRITON_IMPORT_ERROR!r}")
        return _rope_triton(x=x, cos=cos, sin=sin, rot_dim=int(rot_dim))

    T = int(x.shape[2])
    cos2 = cos[:T].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    sin2 = sin[:T].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    rot = int(rot_dim)
    x_rot = x[..., :rot]
    x_pass = x[..., rot:]
    # HF Llama applies rotate_half on a half-split representation:
    # y1 = x1*cos - x2*sin
    # y2 = x1*sin + x2*cos
    x1 = x_rot[..., : rot // 2]
    x2 = x_rot[..., rot // 2 : rot]
    y1 = x1 * cos2 - x2 * sin2
    y2 = x1 * sin2 + x2 * cos2
    return torch.cat([y1, y2, x_pass], dim=-1)


@_typed
def layernorm(*, x: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float) -> Tensor:
    """LayerNorm over the last dimension.

    This matches PyTorch's `F.layer_norm(x, normalized_shape=(D,))` behavior.
    """
    if _use_custom_kernels() and x.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="LayerNorm on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.float32),
            msg=f"LayerNorm on MPS requires fp16/fp32, got dtype={x.dtype}.",
        )
        if _layernorm_fp16 is None:
            raise RuntimeError(f"Metal LayerNorm import failed: {_METAL_IMPORT_ERROR!r}")
        return _layernorm_fp16(x=x, weight=weight, bias=bias, eps=float(eps))

    if _use_custom_kernels() and x.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="LayerNorm on CUDA requires Triton to be available and validated at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.bfloat16),
            msg=f"LayerNorm on CUDA requires fp16/bf16, got dtype={x.dtype}.",
        )
        if _layernorm_triton is None:
            raise RuntimeError(f"Triton LayerNorm import failed: {_TRITON_IMPORT_ERROR!r}")
        return _layernorm_triton(x=x, weight=weight, bias=bias, eps=float(eps))

    D = int(x.shape[-1])
    return F.layer_norm(x, normalized_shape=(D,), weight=weight, bias=bias, eps=float(eps))


@_typed
def attention_decode(
    *,
    q_sem: Tensor,
    q_geo: Tensor,
    k_sem: Tensor,
    k_geo: Tensor,
    v: Tensor,
    k_sem_null: Tensor | None = None,
    k_geo_null: Tensor | None = None,
    v_null: Tensor | None = None,
    sem_scale: float | None = None,
    geo_scale: float | None = None,
) -> Tensor:
    """Fused decode attention (HAL).

    Current supported fast paths:
    - MPS (Metal): decoupled DBA decode (fp16)

    Signature (kwargs-only):
      q_sem, q_geo, k_sem, k_geo, v,
      k_sem_null=None, k_geo_null=None, v_null=None,
      sem_scale=None, geo_scale=None
    """
    if q_sem.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="Attention decode on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            q_sem.dtype == torch.float16,
            msg=f"Attention decode on MPS requires fp16, got dtype={q_sem.dtype}.",
        )
        if _dba_decode_fp16 is None:
            raise RuntimeError(f"Metal attention decode import failed: {_METAL_IMPORT_ERROR!r}")
        return _dba_decode_fp16(
            q_sem=q_sem,
            q_geo=q_geo,
            k_sem=k_sem,
            k_geo=k_geo,
            v=v,
            k_sem_null=k_sem_null,
            k_geo_null=k_geo_null,
            v_null=v_null,
            sem_scale=sem_scale,
            geo_scale=geo_scale,
        )

    raise RuntimeError(
        "attention_decode: no supported backend for this device/dtype.\n"
        f"device={q_sem.device.type} dtype={q_sem.dtype}\n"
        "Use the decoupled attention fused decode paths (CUDA Triton) or Metal DBA decode (MPS fp16)."
    )


@_typed
def scan(
    *,
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
) -> Tensor:
    """Fused scan/SSM kernels (HAL).

    Signature (kwargs-only):
      x, dt, A, B, C, D
    """
    if x.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="SSM scan on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            x.dtype == torch.float16,
            msg=f"SSM scan on MPS requires fp16, got dtype={x.dtype}.",
        )
        if _MetalSSMSelectiveScan is None:
            raise RuntimeError(f"Metal SSM import failed: {_METAL_IMPORT_ERROR!r}")
        return _MetalSSMSelectiveScan().run(x=x, dt=dt, A=A, B=B, C=C, D=D)

    if x.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="SSM scan on CUDA requires Triton kernels to be available and validated at startup.",
        )
        _require(
            x.dtype in (torch.float16, torch.bfloat16),
            msg=f"SSM scan on CUDA requires fp16/bf16, got dtype={x.dtype}.",
        )
        if _fused_selective_scan is None:
            raise RuntimeError(f"Triton SSM import failed: {_TRITON_IMPORT_ERROR!r}")
        return _fused_selective_scan(x, dt, A, B, C, D)

    raise RuntimeError(
        "scan: no supported backend for this device/dtype.\n"
        f"device={x.device.type} dtype={x.dtype}\n"
        "Supported backends: Metal (MPS fp16), Triton (CUDA fp16/bf16)."
    )


@_typed
def adamw_step(
    *,
    p: Tensor,
    grad: Tensor,
    master: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_size: float,
    beta1: float,
    beta2: float,
    eps: float,
    lr_wd: float,
) -> None:
    """Fused AdamW update (HAL).

    This is the low-level per-tensor update used by `AdamWMaster` when fused.
    """
    if p.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="AdamW step on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            p.dtype in (torch.float16, torch.float32),
            msg=f"AdamW step on MPS requires fp16/fp32 params, got dtype={p.dtype}.",
        )
        if _AdamWMasterStep is None:
            raise RuntimeError(f"Metal AdamW import failed: {_METAL_IMPORT_ERROR!r}")
        _AdamWMasterStep().run(
            p=p,
            grad=grad,
            master=master,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            step_size=float(step_size),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            lr_wd=float(lr_wd),
            verbose_build=False,
        )
        return

    if p.device.type == "cuda":
        _require(
            bool(KERNELS.cuda_available and KERNELS.triton_available),
            msg="AdamW step on CUDA requires Triton to be available and validated at startup.",
        )
        _require(
            p.dtype in (torch.float16, torch.bfloat16),
            msg=f"AdamW step on CUDA requires fp16/bf16 params, got dtype={p.dtype}.",
        )
        _require(
            grad.dtype == p.dtype,
            msg="AdamW step on CUDA requires grad dtype to match param dtype.",
        )
        _require(
            master.dtype == torch.float32 and exp_avg.dtype == torch.float32 and exp_avg_sq.dtype == torch.float32,
            msg="AdamW step on CUDA requires fp32 master/exp_avg/exp_avg_sq.",
        )
        _require(
            p.is_contiguous() and grad.is_contiguous() and master.is_contiguous() and exp_avg.is_contiguous() and exp_avg_sq.is_contiguous(),
            msg="AdamW step on CUDA requires all tensors to be contiguous.",
        )
        if _adamw_triton_master_step is None:
            raise RuntimeError(f"Triton AdamW import failed: {_TRITON_IMPORT_ERROR!r}")
        _adamw_triton_master_step(
            p=p.view(-1),
            grad=grad.view(-1),
            master=master.view(-1),
            exp_avg=exp_avg.view(-1),
            exp_avg_sq=exp_avg_sq.view(-1),
            step_size=float(step_size),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            lr_wd=float(lr_wd),
        )
        return

    raise RuntimeError(
        "adamw_step: no supported backend for this device/dtype.\n"
        f"device={p.device.type} dtype={p.dtype}\n"
        "Supported backends: Metal (MPS fp16) and Triton (CUDA fp16/bf16)."
    )


def lion_step(
    *,
    p: Tensor,
    grad: Tensor,
    m: Tensor,
    lr: float,
    beta1: float,
    weight_decay: float = 0.0,
) -> None:
    """Fused Lion update (HAL)."""
    if p.device.type == "mps":
        _require(
            bool(KERNELS.mps_available and KERNELS.metal_ops_loaded),
            msg="Lion step on MPS requires the Metal extension to be available and loaded at startup.",
        )
        _require(
            p.dtype in (torch.float16, torch.float32),
            msg=f"Lion step on MPS requires fp16/fp32 params, got dtype={p.dtype}.",
        )
        if _lion_fp16 is None:
            raise RuntimeError(f"Metal Lion import failed: {_METAL_IMPORT_ERROR!r}")
        _lion_fp16(
            p=p,
            grad=grad,
            m=m,
            lr=float(lr),
            beta1=float(beta1),
            weight_decay=float(weight_decay),
            verbose_build=False,
        )
        return

    raise RuntimeError(
        "lion_step: no supported backend for this device/dtype.\n"
        f"device={p.device.type} dtype={p.dtype}\n"
        "CUDA fused optimizer parity kernel is not available in this build."
    )
