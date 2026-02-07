from __future__ import annotations

import importlib
from typing import Any, Literal

import torch

try:
    _triton = importlib.import_module("triton")
    _tl = importlib.import_module("triton.language")
except Exception:

    class _DummyTriton:
        @staticmethod
        def jit(fn=None, **_kwargs):  # type: ignore[no-untyped-def]
            if fn is None:
                return lambda f: f
            return fn

        @staticmethod
        def cdiv(x: int, y: int) -> int:
            return (x + y - 1) // y

    class _DummyTL:
        constexpr = int

    triton: Any = _DummyTriton()
    tl: Any = _DummyTL()
else:
    triton: Any = _triton
    tl: Any = _tl

Face = Literal["x-", "x+", "y-", "y+", "z-", "z+"]


def triton_distributed_available() -> bool:
    return bool(hasattr(triton, "cdiv") and hasattr(tl, "constexpr")) and hasattr(
        triton, "__name__"
    )


def _require_triton() -> None:
    if not triton_distributed_available():
        raise RuntimeError("Triton is not available")


@triton.jit
def _pack_x_face_kernel(
    field_ptr,
    out_ptr,
    gx,
    gy,
    gz,
    halo,
    start_x,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    yz = gy * gz
    hx = offs // yz
    rem = offs - hx * yz
    y = rem // gz
    z = rem - y * gz
    x = start_x + hx
    src = x * yz + y * gz + z
    v = tl.load(field_ptr + src, mask=m, other=0.0)
    tl.store(out_ptr + offs, v, mask=m)


@triton.jit
def _pack_y_face_kernel(
    field_ptr,
    out_ptr,
    gx,
    gy,
    gz,
    halo,
    start_y,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    hz = halo * gz
    x = offs // hz
    rem = offs - x * hz
    hy = rem // gz
    z = rem - hy * gz
    y = start_y + hy
    yz = gy * gz
    src = x * yz + y * gz + z
    v = tl.load(field_ptr + src, mask=m, other=0.0)
    tl.store(out_ptr + offs, v, mask=m)


@triton.jit
def _pack_z_face_kernel(
    field_ptr,
    out_ptr,
    gx,
    gy,
    gz,
    halo,
    start_z,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = offs // (gy * halo)
    rem = offs - x * (gy * halo)
    y = rem // halo
    hz = rem - y * halo
    z = start_z + hz
    yz = gy * gz
    src = x * yz + y * gz + z
    v = tl.load(field_ptr + src, mask=m, other=0.0)
    tl.store(out_ptr + offs, v, mask=m)


@triton.jit
def _unpack_x_face_kernel(
    in_ptr,
    field_ptr,
    gx,
    gy,
    gz,
    halo,
    start_x,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    yz = gy * gz
    hx = offs // yz
    rem = offs - hx * yz
    y = rem // gz
    z = rem - y * gz
    x = start_x + hx
    dst = x * yz + y * gz + z
    v = tl.load(in_ptr + offs, mask=m, other=0.0)
    tl.store(field_ptr + dst, v, mask=m)


@triton.jit
def _unpack_y_face_kernel(
    in_ptr,
    field_ptr,
    gx,
    gy,
    gz,
    halo,
    start_y,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    hz = halo * gz
    x = offs // hz
    rem = offs - x * hz
    hy = rem // gz
    z = rem - hy * gz
    y = start_y + hy
    yz = gy * gz
    dst = x * yz + y * gz + z
    v = tl.load(in_ptr + offs, mask=m, other=0.0)
    tl.store(field_ptr + dst, v, mask=m)


@triton.jit
def _unpack_z_face_kernel(
    in_ptr,
    field_ptr,
    gx,
    gy,
    gz,
    halo,
    start_z,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = offs // (gy * halo)
    rem = offs - x * (gy * halo)
    y = rem // halo
    hz = rem - y * halo
    z = start_z + hz
    yz = gy * gz
    dst = x * yz + y * gz + z
    v = tl.load(in_ptr + offs, mask=m, other=0.0)
    tl.store(field_ptr + dst, v, mask=m)


@triton.jit
def _classify_migration_faces_kernel(
    pos_ptr,
    out_codes_ptr,
    lo_x,
    lo_y,
    lo_z,
    hi_x,
    hi_y,
    hi_z,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(pos_ptr + i * 3 + 0, mask=m, other=0.0)
    y = tl.load(pos_ptr + i * 3 + 1, mask=m, other=0.0)
    z = tl.load(pos_ptr + i * 3 + 2, mask=m, other=0.0)

    code = tl.full(i.shape, 0, tl.int32)
    code = tl.where(x < lo_x, 1, code)
    code = tl.where((code == 0) & (x >= hi_x), 2, code)
    code = tl.where((code == 0) & (y < lo_y), 3, code)
    code = tl.where((code == 0) & (y >= hi_y), 4, code)
    code = tl.where((code == 0) & (z < lo_z), 5, code)
    code = tl.where((code == 0) & (z >= hi_z), 6, code)
    tl.store(out_codes_ptr + i, code, mask=m)


@triton.jit
def _accumulate_mode_shard_kernel(
    mode_idx_ptr,
    real_ptr,
    imag_ptr,
    accum_real_ptr,
    accum_imag_ptr,
    local_start,
    local_modes,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    midx = tl.load(mode_idx_ptr + i, mask=m, other=0).to(tl.int32)
    local = midx - local_start
    valid = m & (local >= 0) & (local < local_modes)
    rv = tl.load(real_ptr + i, mask=m, other=0.0)
    iv = tl.load(imag_ptr + i, mask=m, other=0.0)
    tl.atomic_add(accum_real_ptr + local, rv, mask=valid)
    tl.atomic_add(accum_imag_ptr + local, iv, mask=valid)


@triton.jit
def _jacobi_halo_kernel(
    phi_ptr,
    rhs_ptr,
    halo_xm_ptr,
    halo_xp_ptr,
    halo_ym_ptr,
    halo_yp_ptr,
    halo_zm_ptr,
    halo_zp_ptr,
    out_ptr,
    gx,
    gy,
    gz,
    dx,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n

    yz = gy * gz
    x = idx // yz
    rem = idx - x * yz
    y = rem // gz
    z = rem - y * gz

    c = tl.load(rhs_ptr + idx, mask=m, other=0.0)

    x_minus_idx = idx - yz
    x_plus_idx = idx + yz
    y_minus_idx = idx - gz
    y_plus_idx = idx + gz
    z_minus_idx = idx - 1
    z_plus_idx = idx + 1

    x_halo_idx = y * gz + z
    y_halo_idx = x * gz + z
    z_halo_idx = x * gy + y

    xm = tl.where(
        x > 0,
        tl.load(phi_ptr + x_minus_idx, mask=m, other=0.0),
        tl.load(halo_xm_ptr + x_halo_idx, mask=m, other=0.0),
    )
    xp = tl.where(
        x < (gx - 1),
        tl.load(phi_ptr + x_plus_idx, mask=m, other=0.0),
        tl.load(halo_xp_ptr + x_halo_idx, mask=m, other=0.0),
    )
    ym = tl.where(
        y > 0,
        tl.load(phi_ptr + y_minus_idx, mask=m, other=0.0),
        tl.load(halo_ym_ptr + y_halo_idx, mask=m, other=0.0),
    )
    yp = tl.where(
        y < (gy - 1),
        tl.load(phi_ptr + y_plus_idx, mask=m, other=0.0),
        tl.load(halo_yp_ptr + y_halo_idx, mask=m, other=0.0),
    )
    zm = tl.where(
        z > 0,
        tl.load(phi_ptr + z_minus_idx, mask=m, other=0.0),
        tl.load(halo_zm_ptr + z_halo_idx, mask=m, other=0.0),
    )
    zp = tl.where(
        z < (gz - 1),
        tl.load(phi_ptr + z_plus_idx, mask=m, other=0.0),
        tl.load(halo_zp_ptr + z_halo_idx, mask=m, other=0.0),
    )

    h2 = dx * dx
    out = (xm + xp + ym + yp + zm + zp - h2 * c) * (1.0 / 6.0)
    tl.store(out_ptr + idx, out, mask=m)


@triton.jit
def _advance_interior_halo_kernel(
    rho_ext_ptr,
    mx_ext_ptr,
    my_ext_ptr,
    mz_ext_ptr,
    e_ext_ptr,
    phi_ext_ptr,
    out_rho_ptr,
    out_mx_ptr,
    out_my_ptr,
    out_mz_ptr,
    out_e_ptr,
    gx,
    gy,
    gz,
    h,
    dt,
    dx,
    gamma,
    rho_min,
    viscosity,
    thermal_diff,
    n,
    BLOCK,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n

    yz = gy * gz
    x = i // yz
    rem = i - x * yz
    y = rem // gz
    z = rem - y * gz

    ey = gy + 2 * h
    ez = gz + 2 * h
    xh = x + h
    yh = y + h
    zh = z + h
    ext_idx = xh * ey * ez + yh * ez + zh
    x_stride = ey * ez
    y_stride = ez

    rho_c = tl.maximum(tl.load(rho_ext_ptr + ext_idx, mask=m, other=0.0), rho_min)
    mx_c = tl.load(mx_ext_ptr + ext_idx, mask=m, other=0.0)
    my_c = tl.load(my_ext_ptr + ext_idx, mask=m, other=0.0)
    mz_c = tl.load(mz_ext_ptr + ext_idx, mask=m, other=0.0)
    e_c = tl.load(e_ext_ptr + ext_idx, mask=m, other=0.0)

    xm = ext_idx - x_stride
    xp = ext_idx + x_stride
    ym = ext_idx - y_stride
    yp = ext_idx + y_stride
    zm = ext_idx - 1
    zp = ext_idx + 1

    e_xm = tl.load(e_ext_ptr + xm, mask=m, other=0.0)
    e_xp = tl.load(e_ext_ptr + xp, mask=m, other=0.0)
    e_ym = tl.load(e_ext_ptr + ym, mask=m, other=0.0)
    e_yp = tl.load(e_ext_ptr + yp, mask=m, other=0.0)
    e_zm = tl.load(e_ext_ptr + zm, mask=m, other=0.0)
    e_zp = tl.load(e_ext_ptr + zp, mask=m, other=0.0)

    p_xm = (gamma - 1.0) * tl.maximum(e_xm, 0.0)
    p_xp = (gamma - 1.0) * tl.maximum(e_xp, 0.0)
    p_ym = (gamma - 1.0) * tl.maximum(e_ym, 0.0)
    p_yp = (gamma - 1.0) * tl.maximum(e_yp, 0.0)
    p_zm = (gamma - 1.0) * tl.maximum(e_zm, 0.0)
    p_zp = (gamma - 1.0) * tl.maximum(e_zp, 0.0)

    inv_2dx = 0.5 / dx
    grad_px = (p_xp - p_xm) * inv_2dx
    grad_py = (p_yp - p_ym) * inv_2dx
    grad_pz = (p_zp - p_zm) * inv_2dx

    phi_xm = tl.load(phi_ext_ptr + xm, mask=m, other=0.0)
    phi_xp = tl.load(phi_ext_ptr + xp, mask=m, other=0.0)
    phi_ym = tl.load(phi_ext_ptr + ym, mask=m, other=0.0)
    phi_yp = tl.load(phi_ext_ptr + yp, mask=m, other=0.0)
    phi_zm = tl.load(phi_ext_ptr + zm, mask=m, other=0.0)
    phi_zp = tl.load(phi_ext_ptr + zp, mask=m, other=0.0)
    grad_phix = (phi_xp - phi_xm) * inv_2dx
    grad_phiy = (phi_yp - phi_ym) * inv_2dx
    grad_phiz = (phi_zp - phi_zm) * inv_2dx

    mx_xm = tl.load(mx_ext_ptr + xm, mask=m, other=0.0)
    mx_xp = tl.load(mx_ext_ptr + xp, mask=m, other=0.0)
    my_ym = tl.load(my_ext_ptr + ym, mask=m, other=0.0)
    my_yp = tl.load(my_ext_ptr + yp, mask=m, other=0.0)
    mz_zm = tl.load(mz_ext_ptr + zm, mask=m, other=0.0)
    mz_zp = tl.load(mz_ext_ptr + zp, mask=m, other=0.0)
    div_m = ((mx_xp - mx_xm) + (my_yp - my_ym) + (mz_zp - mz_zm)) * inv_2dx

    inv_dx2 = 1.0 / (dx * dx)
    lap_e = (e_xm + e_xp + e_ym + e_yp + e_zm + e_zp - 6.0 * e_c) * inv_dx2
    lap_mx = (
        mx_xm
        + mx_xp
        + tl.load(mx_ext_ptr + ym, mask=m, other=0.0)
        + tl.load(mx_ext_ptr + yp, mask=m, other=0.0)
        + tl.load(mx_ext_ptr + zm, mask=m, other=0.0)
        + tl.load(mx_ext_ptr + zp, mask=m, other=0.0)
        - 6.0 * mx_c
    ) * inv_dx2
    lap_my = (
        tl.load(my_ext_ptr + xm, mask=m, other=0.0)
        + tl.load(my_ext_ptr + xp, mask=m, other=0.0)
        + my_ym
        + my_yp
        + tl.load(my_ext_ptr + zm, mask=m, other=0.0)
        + tl.load(my_ext_ptr + zp, mask=m, other=0.0)
        - 6.0 * my_c
    ) * inv_dx2
    lap_mz = (
        tl.load(mz_ext_ptr + xm, mask=m, other=0.0)
        + tl.load(mz_ext_ptr + xp, mask=m, other=0.0)
        + tl.load(mz_ext_ptr + ym, mask=m, other=0.0)
        + tl.load(mz_ext_ptr + yp, mask=m, other=0.0)
        + mz_zm
        + mz_zp
        - 6.0 * mz_c
    ) * inv_dx2

    rho_new = tl.maximum(rho_c - dt * div_m, rho_min)
    mx_new = mx_c + dt * (-grad_px - rho_c * grad_phix + viscosity * lap_mx)
    my_new = my_c + dt * (-grad_py - rho_c * grad_phiy + viscosity * lap_my)
    mz_new = mz_c + dt * (-grad_pz - rho_c * grad_phiz + viscosity * lap_mz)
    e_new = tl.maximum(e_c + dt * (thermal_diff * lap_e), 0.0)

    tl.store(out_rho_ptr + i, rho_new, mask=m)
    tl.store(out_mx_ptr + i, mx_new, mask=m)
    tl.store(out_my_ptr + i, my_new, mask=m)
    tl.store(out_mz_ptr + i, mz_new, mask=m)
    tl.store(out_e_ptr + i, e_new, mask=m)


def pack_halo_face_scalar(
    field: torch.Tensor, *, face: Face, halo: int
) -> torch.Tensor:
    if field.ndim != 3:
        raise ValueError(f"field must be [gx,gy,gz], got {tuple(field.shape)}")
    if halo <= 0:
        raise ValueError("halo must be > 0")

    gx, gy, gz = [int(v) for v in field.shape]
    if not (field.is_cuda and triton_distributed_available()):
        if face == "x-":
            return field[:halo, :, :].contiguous()
        if face == "x+":
            return field[gx - halo :, :, :].contiguous()
        if face == "y-":
            return field[:, :halo, :].contiguous()
        if face == "y+":
            return field[:, gy - halo :, :].contiguous()
        if face == "z-":
            return field[:, :, :halo].contiguous()
        return field[:, :, gz - halo :].contiguous()

    _require_triton()
    block = 256
    if face in ("x-", "x+"):
        out = torch.empty((halo, gy, gz), device=field.device, dtype=field.dtype)
        n = int(out.numel())
        start_x = 0 if face == "x-" else gx - halo
        grid = (triton.cdiv(n, block),)
        _pack_x_face_kernel[grid](
            field,
            out,
            gx=gx,
            gy=gy,
            gz=gz,
            halo=halo,
            start_x=start_x,
            n=n,
            BLOCK=block,
        )
        return out
    if face in ("y-", "y+"):
        out = torch.empty((gx, halo, gz), device=field.device, dtype=field.dtype)
        n = int(out.numel())
        start_y = 0 if face == "y-" else gy - halo
        grid = (triton.cdiv(n, block),)
        _pack_y_face_kernel[grid](
            field,
            out,
            gx=gx,
            gy=gy,
            gz=gz,
            halo=halo,
            start_y=start_y,
            n=n,
            BLOCK=block,
        )
        return out

    out = torch.empty((gx, gy, halo), device=field.device, dtype=field.dtype)
    n = int(out.numel())
    start_z = 0 if face == "z-" else gz - halo
    grid = (triton.cdiv(n, block),)
    _pack_z_face_kernel[grid](
        field,
        out,
        gx=gx,
        gy=gy,
        gz=gz,
        halo=halo,
        start_z=start_z,
        n=n,
        BLOCK=block,
    )
    return out


def pack_halo_face(field: torch.Tensor, *, face: Face, halo: int) -> torch.Tensor:
    if field.ndim == 3:
        return pack_halo_face_scalar(field, face=face, halo=halo)
    if field.ndim == 4 and int(field.shape[-1]) == 3:
        chunks = [
            pack_halo_face_scalar(field[..., c], face=face, halo=halo) for c in range(3)
        ]
        return torch.stack(chunks, dim=-1)
    raise ValueError(
        f"field must be [gx,gy,gz] or [gx,gy,gz,3], got {tuple(field.shape)}"
    )


def classify_migration_faces(
    positions: torch.Tensor,
    *,
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
) -> torch.Tensor:
    if positions.ndim != 2 or int(positions.shape[1]) != 3:
        raise ValueError(f"positions must be [N,3], got {tuple(positions.shape)}")
    n = int(positions.shape[0])
    if n == 0:
        return torch.empty((0,), device=positions.device, dtype=torch.int32)

    if not (positions.is_cuda and triton_distributed_available()):
        out = torch.zeros((n,), device=positions.device, dtype=torch.int32)
        out = torch.where(
            positions[:, 0] < lo[0],
            torch.tensor(1, device=positions.device, dtype=torch.int32),
            out,
        )
        out = torch.where(
            (out == 0) & (positions[:, 0] >= hi[0]),
            torch.tensor(2, device=positions.device, dtype=torch.int32),
            out,
        )
        out = torch.where(
            (out == 0) & (positions[:, 1] < lo[1]),
            torch.tensor(3, device=positions.device, dtype=torch.int32),
            out,
        )
        out = torch.where(
            (out == 0) & (positions[:, 1] >= hi[1]),
            torch.tensor(4, device=positions.device, dtype=torch.int32),
            out,
        )
        out = torch.where(
            (out == 0) & (positions[:, 2] < lo[2]),
            torch.tensor(5, device=positions.device, dtype=torch.int32),
            out,
        )
        out = torch.where(
            (out == 0) & (positions[:, 2] >= hi[2]),
            torch.tensor(6, device=positions.device, dtype=torch.int32),
            out,
        )
        return out

    _require_triton()
    out = torch.empty((n,), device=positions.device, dtype=torch.int32)
    block = 256
    grid = (triton.cdiv(n, block),)
    _classify_migration_faces_kernel[grid](
        positions,
        out,
        float(lo[0]),
        float(lo[1]),
        float(lo[2]),
        float(hi[0]),
        float(hi[1]),
        float(hi[2]),
        n=n,
        BLOCK=block,
    )
    return out


def unpack_halo_face_scalar(
    field: torch.Tensor,
    face_tensor: torch.Tensor,
    *,
    face: Face,
    halo: int,
) -> None:
    if field.ndim != 3:
        raise ValueError(f"field must be [gx,gy,gz], got {tuple(field.shape)}")
    gx, gy, gz = [int(v) for v in field.shape]

    if not (field.is_cuda and triton_distributed_available()):
        if face == "x-":
            field[:halo, :, :].copy_(face_tensor)
        elif face == "x+":
            field[gx - halo :, :, :].copy_(face_tensor)
        elif face == "y-":
            field[:, :halo, :].copy_(face_tensor)
        elif face == "y+":
            field[:, gy - halo :, :].copy_(face_tensor)
        elif face == "z-":
            field[:, :, :halo].copy_(face_tensor)
        else:
            field[:, :, gz - halo :].copy_(face_tensor)
        return

    _require_triton()
    block = 256
    n = int(face_tensor.numel())
    grid = (triton.cdiv(n, block),)
    if face in ("x-", "x+"):
        start_x = 0 if face == "x-" else gx - halo
        _unpack_x_face_kernel[grid](
            face_tensor,
            field,
            gx=gx,
            gy=gy,
            gz=gz,
            halo=halo,
            start_x=start_x,
            n=n,
            BLOCK=block,
        )
    elif face in ("y-", "y+"):
        start_y = 0 if face == "y-" else gy - halo
        _unpack_y_face_kernel[grid](
            face_tensor,
            field,
            gx=gx,
            gy=gy,
            gz=gz,
            halo=halo,
            start_y=start_y,
            n=n,
            BLOCK=block,
        )
    else:
        start_z = 0 if face == "z-" else gz - halo
        _unpack_z_face_kernel[grid](
            face_tensor,
            field,
            gx=gx,
            gy=gy,
            gz=gz,
            halo=halo,
            start_z=start_z,
            n=n,
            BLOCK=block,
        )


def jacobi_step_halo(
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
    if phi.shape != rhs.shape:
        raise ValueError(
            f"phi/rhs shape mismatch: {tuple(phi.shape)} vs {tuple(rhs.shape)}"
        )
    gx, gy, gz = [int(v) for v in phi.shape]
    out_t = out if out is not None else torch.empty_like(phi)

    if not (phi.is_cuda and triton_distributed_available()):
        xm = torch.empty_like(phi)
        xp = torch.empty_like(phi)
        ym = torch.empty_like(phi)
        yp = torch.empty_like(phi)
        zm = torch.empty_like(phi)
        zp = torch.empty_like(phi)
        xm[1:, :, :] = phi[:-1, :, :]
        xm[0, :, :] = halo_xm
        xp[:-1, :, :] = phi[1:, :, :]
        xp[-1, :, :] = halo_xp
        ym[:, 1:, :] = phi[:, :-1, :]
        ym[:, 0, :] = halo_ym
        yp[:, :-1, :] = phi[:, 1:, :]
        yp[:, -1, :] = halo_yp
        zm[:, :, 1:] = phi[:, :, :-1]
        zm[:, :, 0] = halo_zm
        zp[:, :, :-1] = phi[:, :, 1:]
        zp[:, :, -1] = halo_zp
        out_t.copy_((xm + xp + ym + yp + zm + zp - (dx * dx) * rhs) / 6.0)
        return out_t

    _require_triton()
    n = int(phi.numel())
    block = 256
    grid = (triton.cdiv(n, block),)
    _jacobi_halo_kernel[grid](
        phi.view(-1),
        rhs.view(-1),
        halo_xm.view(-1),
        halo_xp.view(-1),
        halo_ym.view(-1),
        halo_yp.view(-1),
        halo_zm.view(-1),
        halo_zp.view(-1),
        out_t.view(-1),
        gx=gx,
        gy=gy,
        gz=gz,
        dx=float(dx),
        n=n,
        BLOCK=block,
    )
    return out_t


def accumulate_mode_shard(
    mode_idx: torch.Tensor,
    contrib_real: torch.Tensor,
    contrib_imag: torch.Tensor,
    *,
    local_start: int,
    accum_real: torch.Tensor,
    accum_imag: torch.Tensor,
) -> None:
    n = int(mode_idx.numel())
    if n == 0:
        return

    if not (mode_idx.is_cuda and triton_distributed_available()):
        local = mode_idx.to(torch.int64) - int(local_start)
        mask = (local >= 0) & (local < int(accum_real.numel()))
        if torch.any(mask):
            accum_real.scatter_add_(0, local[mask].to(torch.int64), contrib_real[mask])
            accum_imag.scatter_add_(0, local[mask].to(torch.int64), contrib_imag[mask])
        return

    _require_triton()
    block = 256
    grid = (triton.cdiv(n, block),)
    _accumulate_mode_shard_kernel[grid](
        mode_idx,
        contrib_real,
        contrib_imag,
        accum_real,
        accum_imag,
        int(local_start),
        int(accum_real.numel()),
        n=n,
        BLOCK=block,
    )


def advance_interior_halo(
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
    gx, gy, gz = [int(v) for v in out_rho.shape]
    n = gx * gy * gz
    if not (out_rho.is_cuda and triton_distributed_available()):
        c = (slice(halo, halo + gx), slice(halo, halo + gy), slice(halo, halo + gz))
        rho_c = torch.clamp(rho_ext[c], min=rho_min)
        mx_c = mom_ext[c + (0,)]
        my_c = mom_ext[c + (1,)]
        mz_c = mom_ext[c + (2,)]
        e_c = e_ext[c]
        inv_2dx = 0.5 / dx
        p = (gamma - 1.0) * torch.clamp(e_ext, min=0.0)
        grad_px = (
            p[halo + 1 : halo + gx + 1, halo : halo + gy, halo : halo + gz]
            - p[halo - 1 : halo + gx - 1, halo : halo + gy, halo : halo + gz]
        ) * inv_2dx
        grad_py = (
            p[halo : halo + gx, halo + 1 : halo + gy + 1, halo : halo + gz]
            - p[halo : halo + gx, halo - 1 : halo + gy - 1, halo : halo + gz]
        ) * inv_2dx
        grad_pz = (
            p[halo : halo + gx, halo : halo + gy, halo + 1 : halo + gz + 1]
            - p[halo : halo + gx, halo : halo + gy, halo - 1 : halo + gz - 1]
        ) * inv_2dx
        grad_phix = (
            phi_ext[halo + 1 : halo + gx + 1, halo : halo + gy, halo : halo + gz]
            - phi_ext[halo - 1 : halo + gx - 1, halo : halo + gy, halo : halo + gz]
        ) * inv_2dx
        grad_phiy = (
            phi_ext[halo : halo + gx, halo + 1 : halo + gy + 1, halo : halo + gz]
            - phi_ext[halo : halo + gx, halo - 1 : halo + gy - 1, halo : halo + gz]
        ) * inv_2dx
        grad_phiz = (
            phi_ext[halo : halo + gx, halo : halo + gy, halo + 1 : halo + gz + 1]
            - phi_ext[halo : halo + gx, halo : halo + gy, halo - 1 : halo + gz - 1]
        ) * inv_2dx
        div_m = (
            (
                mom_ext[halo + 1 : halo + gx + 1, halo : halo + gy, halo : halo + gz, 0]
                - mom_ext[
                    halo - 1 : halo + gx - 1, halo : halo + gy, halo : halo + gz, 0
                ]
            )
            + (
                mom_ext[halo : halo + gx, halo + 1 : halo + gy + 1, halo : halo + gz, 1]
                - mom_ext[
                    halo : halo + gx, halo - 1 : halo + gy - 1, halo : halo + gz, 1
                ]
            )
            + (
                mom_ext[halo : halo + gx, halo : halo + gy, halo + 1 : halo + gz + 1, 2]
                - mom_ext[
                    halo : halo + gx, halo : halo + gy, halo - 1 : halo + gz - 1, 2
                ]
            )
        ) * inv_2dx
        inv_dx2 = 1.0 / (dx * dx)
        lap_e = (
            e_ext[halo + 1 : halo + gx + 1, halo : halo + gy, halo : halo + gz]
            + e_ext[halo - 1 : halo + gx - 1, halo : halo + gy, halo : halo + gz]
            + e_ext[halo : halo + gx, halo + 1 : halo + gy + 1, halo : halo + gz]
            + e_ext[halo : halo + gx, halo - 1 : halo + gy - 1, halo : halo + gz]
            + e_ext[halo : halo + gx, halo : halo + gy, halo + 1 : halo + gz + 1]
            + e_ext[halo : halo + gx, halo : halo + gy, halo - 1 : halo + gz - 1]
            - 6.0 * e_c
        ) * inv_dx2

        def _lap_m(comp: int) -> torch.Tensor:
            m = mom_ext[..., comp]
            return (
                m[halo + 1 : halo + gx + 1, halo : halo + gy, halo : halo + gz]
                + m[halo - 1 : halo + gx - 1, halo : halo + gy, halo : halo + gz]
                + m[halo : halo + gx, halo + 1 : halo + gy + 1, halo : halo + gz]
                + m[halo : halo + gx, halo - 1 : halo + gy - 1, halo : halo + gz]
                + m[halo : halo + gx, halo : halo + gy, halo + 1 : halo + gz + 1]
                + m[halo : halo + gx, halo : halo + gy, halo - 1 : halo + gz - 1]
                - 6.0 * m[c]
            ) * inv_dx2

        out_rho.copy_(torch.clamp(rho_c - dt * div_m, min=rho_min))
        out_mom[..., 0].copy_(
            mx_c + dt * (-grad_px - rho_c * grad_phix + viscosity * _lap_m(0))
        )
        out_mom[..., 1].copy_(
            my_c + dt * (-grad_py - rho_c * grad_phiy + viscosity * _lap_m(1))
        )
        out_mom[..., 2].copy_(
            mz_c + dt * (-grad_pz - rho_c * grad_phiz + viscosity * _lap_m(2))
        )
        out_e.copy_(torch.clamp(e_c + dt * (thermal_diffusivity * lap_e), min=0.0))
        return

    _require_triton()
    block = 256
    grid = (triton.cdiv(n, block),)
    out_mx = torch.empty_like(out_rho)
    out_my = torch.empty_like(out_rho)
    out_mz = torch.empty_like(out_rho)
    _advance_interior_halo_kernel[grid](
        rho_ext.view(-1),
        mom_ext[..., 0].contiguous().view(-1),
        mom_ext[..., 1].contiguous().view(-1),
        mom_ext[..., 2].contiguous().view(-1),
        e_ext.view(-1),
        phi_ext.view(-1),
        out_rho.view(-1),
        out_mx.view(-1),
        out_my.view(-1),
        out_mz.view(-1),
        out_e.view(-1),
        gx=gx,
        gy=gy,
        gz=gz,
        h=int(halo),
        dt=float(dt),
        dx=float(dx),
        gamma=float(gamma),
        rho_min=float(rho_min),
        viscosity=float(viscosity),
        thermal_diff=float(thermal_diffusivity),
        n=n,
        BLOCK=block,
    )
    out_mom[..., 0].copy_(out_mx)
    out_mom[..., 1].copy_(out_my)
    out_mom[..., 2].copy_(out_mz)
