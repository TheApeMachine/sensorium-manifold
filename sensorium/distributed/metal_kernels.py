from __future__ import annotations

from typing import Literal

import torch

from .metal_jit import load_distributed_metal_ops

Face = Literal["x-", "x+", "y-", "y+", "z-", "z+"]


def _ops():
    return load_distributed_metal_ops(verbose=False)


def metal_distributed_available() -> bool:
    return torch.backends.mps.is_available()


def pack_halo_face_scalar_metal(
    field: torch.Tensor, *, face: Face, halo: int
) -> torch.Tensor:
    if field.ndim != 3:
        raise ValueError(f"field must be [gx,gy,gz], got {tuple(field.shape)}")
    if field.device.type != "mps":
        raise RuntimeError("pack_halo_face_scalar_metal requires MPS tensor")

    gx, gy, gz = [int(v) for v in field.shape]
    ops = _ops()
    if face == "x-":
        out = torch.empty((halo, gy, gz), device=field.device, dtype=field.dtype)
        ops.distributed_pack_halo_x(field.contiguous(), out, int(halo), 0)
        return out
    if face == "x+":
        out = torch.empty((halo, gy, gz), device=field.device, dtype=field.dtype)
        ops.distributed_pack_halo_x(field.contiguous(), out, int(halo), gx - halo)
        return out
    if face == "y-":
        out = torch.empty((gx, halo, gz), device=field.device, dtype=field.dtype)
        ops.distributed_pack_halo_y(field.contiguous(), out, int(halo), 0)
        return out
    if face == "y+":
        out = torch.empty((gx, halo, gz), device=field.device, dtype=field.dtype)
        ops.distributed_pack_halo_y(field.contiguous(), out, int(halo), gy - halo)
        return out
    if face == "z-":
        out = torch.empty((gx, gy, halo), device=field.device, dtype=field.dtype)
        ops.distributed_pack_halo_z(field.contiguous(), out, int(halo), 0)
        return out
    out = torch.empty((gx, gy, halo), device=field.device, dtype=field.dtype)
    ops.distributed_pack_halo_z(field.contiguous(), out, int(halo), gz - halo)
    return out


def classify_migration_faces_metal(
    positions: torch.Tensor,
    *,
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
) -> torch.Tensor:
    if positions.device.type != "mps":
        raise RuntimeError("classify_migration_faces_metal requires MPS tensor")
    ops = _ops()
    out = torch.empty((positions.shape[0],), device=positions.device, dtype=torch.int32)
    ops.distributed_classify_faces(
        positions.contiguous(),
        out,
        float(lo[0]),
        float(lo[1]),
        float(lo[2]),
        float(hi[0]),
        float(hi[1]),
        float(hi[2]),
    )
    return out


def unpack_halo_face_scalar_metal(
    field: torch.Tensor,
    face_tensor: torch.Tensor,
    *,
    face: Face,
    halo: int,
) -> None:
    if field.device.type != "mps":
        raise RuntimeError("unpack_halo_face_scalar_metal requires MPS tensor")
    gx, gy, gz = [int(v) for v in field.shape]
    ops = _ops()
    if face == "x-":
        ops.distributed_unpack_halo_x(
            face_tensor.contiguous(), field.contiguous(), int(halo), 0
        )
        return
    if face == "x+":
        ops.distributed_unpack_halo_x(
            face_tensor.contiguous(), field.contiguous(), int(halo), gx - halo
        )
        return
    if face == "y-":
        ops.distributed_unpack_halo_y(
            face_tensor.contiguous(), field.contiguous(), int(halo), 0
        )
        return
    if face == "y+":
        ops.distributed_unpack_halo_y(
            face_tensor.contiguous(), field.contiguous(), int(halo), gy - halo
        )
        return
    if face == "z-":
        ops.distributed_unpack_halo_z(
            face_tensor.contiguous(), field.contiguous(), int(halo), 0
        )
        return
    ops.distributed_unpack_halo_z(
        face_tensor.contiguous(), field.contiguous(), int(halo), gz - halo
    )


def jacobi_step_halo_metal(
    phi: torch.Tensor,
    rhs: torch.Tensor,
    *,
    halo_xm: torch.Tensor,
    halo_xp: torch.Tensor,
    halo_ym: torch.Tensor,
    halo_yp: torch.Tensor,
    halo_zm: torch.Tensor,
    halo_zp: torch.Tensor,
    dx: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if phi.device.type != "mps":
        raise RuntimeError("jacobi_step_halo_metal requires MPS tensor")
    if out is None:
        out = torch.empty_like(phi)
    ops = _ops()
    ops.distributed_jacobi_halo(
        phi.contiguous(),
        rhs.contiguous(),
        halo_xm.contiguous(),
        halo_xp.contiguous(),
        halo_ym.contiguous(),
        halo_yp.contiguous(),
        halo_zm.contiguous(),
        halo_zp.contiguous(),
        out,
        float(dx),
    )
    return out


def advance_interior_halo_metal(
    *,
    rho_ext: torch.Tensor,
    mom_ext: torch.Tensor,
    e_ext: torch.Tensor,
    phi_ext: torch.Tensor,
    dt: float,
    dx: float,
    gamma: float,
    rho_min: float,
    viscosity: float,
    thermal_diffusivity: float,
    halo: int,
    out_rho: torch.Tensor,
    out_mom: torch.Tensor,
    out_e: torch.Tensor,
) -> None:
    if rho_ext.device.type != "mps":
        raise RuntimeError("advance_interior_halo_metal requires MPS tensors")
    ops = _ops()
    out_mx = torch.empty_like(out_rho)
    out_my = torch.empty_like(out_rho)
    out_mz = torch.empty_like(out_rho)
    ops.distributed_advance_interior_halo(
        rho_ext.contiguous(),
        mom_ext[..., 0].contiguous(),
        mom_ext[..., 1].contiguous(),
        mom_ext[..., 2].contiguous(),
        e_ext.contiguous(),
        phi_ext.contiguous(),
        out_rho,
        out_mx,
        out_my,
        out_mz,
        out_e,
        int(halo),
        float(dt),
        float(dx),
        float(gamma),
        float(rho_min),
        float(viscosity),
        float(thermal_diffusivity),
    )
    out_mom[..., 0].copy_(out_mx)
    out_mom[..., 1].copy_(out_my)
    out_mom[..., 2].copy_(out_mz)
