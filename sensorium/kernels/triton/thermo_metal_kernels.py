# pyright: reportInvalidTypeForm=false

"""Triton port of the optimized Metal thermodynamics kernels.

This module is intended to achieve feature parity with the fast Metal pipeline
implemented in `sensorium/kernels/metal/manifold_physics.metal`:
  - sort-based particle scatter (cell binning + reorder + CIC deposit)
  - compressible gas RK2 update with admissibility checks (NaN poisoning)
  - fused PIC gather + particle update (gravity + advection + periodic wrap)

The API is intentionally close to the Metal ops surface so the CUDA domain can
mirror the MPS domain control flow.

Triton is optional at import time; runtime entrypoints must call
`_require_triton()`.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import torch


class _DummyTriton:
    @staticmethod
    def jit(fn=None, **_kwargs):  # type: ignore[no-untyped-def]
        if fn is None:
            return lambda f: f
        return fn

    @staticmethod
    def autotune(
        configs=None,
        key=None,
        prune_configs_by=None,
        reset_to_zero=None,
        warmup=None,
        rep=None,
    ):  # type: ignore[no-untyped-def]
        def deco(f):
            return f

        return deco

    @staticmethod
    def heuristics(values=None):  # type: ignore[no-untyped-def]
        def deco(f):
            return f

        return deco


try:
    triton: Any = importlib.import_module("triton")
    tl: Any = importlib.import_module("triton.language")
except Exception as e:  # pragma: no cover
    triton = _DummyTriton()
    tl = None
    _TRITON_IMPORT_ERROR: Exception = e
else:
    _TRITON_IMPORT_ERROR = RuntimeError("unreachable")


def _require_triton() -> None:
    if triton is None or tl is None:  # pragma: no cover
        raise RuntimeError(
            f"Triton is required for CUDA thermodynamics kernels: {_TRITON_IMPORT_ERROR!r}"
        )


def _grid_num_warps(n: int) -> int:
    # Simple policy; overridden per-kernel by autotune when available.
    if n >= 1 << 20:
        return 8
    if n >= 1 << 18:
        return 4
    return 2


if hasattr(triton, "Config"):
    _SCATTER_CONFIGS = [
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
        triton.Config({"BLOCK": 512}, num_warps=8),
    ]
    _GAS_CONFIGS = [
        triton.Config({"BLOCK": 64}, num_warps=4),
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
    ]
    _GATHER_CONFIGS = [
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
    ]
else:  # pragma: no cover
    _SCATTER_CONFIGS = []
    _GAS_CONFIGS = []
    _GATHER_CONFIGS = []


@dataclass(frozen=True)
class SortScatterMeta:
    grid_x: int
    grid_y: int
    grid_z: int
    dx: float

    @property
    def inv_dx(self) -> float:
        return 1.0 / float(self.dx)

    @property
    def num_cells(self) -> int:
        return int(self.grid_x) * int(self.grid_y) * int(self.grid_z)


# =============================================================================
# Sort-based scatter (compute cell ids, count cells, reorder, CIC deposit)
# =============================================================================


@triton.autotune(configs=_SCATTER_CONFIGS, key=["N"])
@triton.jit
def scatter_compute_cell_idx_kernel(
    pos_ptr,  # fp32 [N*3]
    cell_idx_ptr,  # int32 [N]
    N,
    grid_x: tl.constexpr,
    grid_y: tl.constexpr,
    grid_z: tl.constexpr,
    inv_dx,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    px = tl.load(pos_ptr + offs * 3 + 0, mask=m, other=0.0)
    py = tl.load(pos_ptr + offs * 3 + 1, mask=m, other=0.0)
    pz = tl.load(pos_ptr + offs * 3 + 2, mask=m, other=0.0)

    gx = px * inv_dx
    gy = py * inv_dx
    gz = pz * inv_dx

    # Periodic wrap into [0, dims)
    gx = gx - float(grid_x) * tl.floor(gx / float(grid_x))
    gy = gy - float(grid_y) * tl.floor(gy / float(grid_y))
    gz = gz - float(grid_z) * tl.floor(gz / float(grid_z))

    ix = tl.floor(gx).to(tl.int32)
    iy = tl.floor(gy).to(tl.int32)
    iz = tl.floor(gz).to(tl.int32)

    cell = ix * (grid_y * grid_z) + iy * grid_z + iz
    tl.store(cell_idx_ptr + offs, cell, mask=m)


@triton.autotune(configs=_SCATTER_CONFIGS, key=["N"])
@triton.jit
def scatter_count_cells_kernel(
    cell_idx_ptr,  # int32 [N]
    cell_counts_ptr,  # int32 [num_cells]
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    cell = tl.load(cell_idx_ptr + offs, mask=m, other=0).to(tl.int32)

    # Atomic add for all active lanes.
    tl.atomic_add(cell_counts_ptr + cell, 1, mask=m)


@triton.autotune(configs=_SCATTER_CONFIGS, key=["N"])
@triton.jit
def scatter_reorder_particles_kernel(
    pos_ptr,  # fp32 [N*3]
    vel_ptr,  # fp32 [N*3]
    mass_ptr,  # fp32 [N]
    heat_ptr,  # fp32 [N]
    energy_ptr,  # fp32 [N]
    cell_idx_ptr,  # int32 [N]
    cell_starts_ptr,  # int32 [num_cells]
    cell_offsets_ptr,  # int32 [num_cells] (must start at 0)
    pos_out_ptr,  # fp32 [N*3]
    vel_out_ptr,  # fp32 [N*3]
    mass_out_ptr,  # fp32 [N]
    heat_out_ptr,  # fp32 [N]
    energy_out_ptr,  # fp32 [N]
    original_idx_out_ptr,  # int32 [N]
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    cell = tl.load(cell_idx_ptr + offs, mask=m, other=0).to(tl.int32)
    start = tl.load(cell_starts_ptr + cell, mask=m, other=0).to(tl.int32)
    local = tl.atomic_add(cell_offsets_ptr + cell, 1, mask=m).to(tl.int32)
    dest = start + local

    px = tl.load(pos_ptr + offs * 3 + 0, mask=m, other=0.0)
    py = tl.load(pos_ptr + offs * 3 + 1, mask=m, other=0.0)
    pz = tl.load(pos_ptr + offs * 3 + 2, mask=m, other=0.0)
    vx = tl.load(vel_ptr + offs * 3 + 0, mask=m, other=0.0)
    vy = tl.load(vel_ptr + offs * 3 + 1, mask=m, other=0.0)
    vz = tl.load(vel_ptr + offs * 3 + 2, mask=m, other=0.0)

    tl.store(pos_out_ptr + dest * 3 + 0, px, mask=m)
    tl.store(pos_out_ptr + dest * 3 + 1, py, mask=m)
    tl.store(pos_out_ptr + dest * 3 + 2, pz, mask=m)
    tl.store(vel_out_ptr + dest * 3 + 0, vx, mask=m)
    tl.store(vel_out_ptr + dest * 3 + 1, vy, mask=m)
    tl.store(vel_out_ptr + dest * 3 + 2, vz, mask=m)
    tl.store(mass_out_ptr + dest, tl.load(mass_ptr + offs, mask=m, other=0.0), mask=m)
    tl.store(heat_out_ptr + dest, tl.load(heat_ptr + offs, mask=m, other=0.0), mask=m)
    tl.store(
        energy_out_ptr + dest, tl.load(energy_ptr + offs, mask=m, other=0.0), mask=m
    )
    tl.store(original_idx_out_ptr + dest, offs.to(tl.int32), mask=m)


@triton.autotune(configs=_SCATTER_CONFIGS, key=["N"])
@triton.jit
def scatter_sorted_kernel(
    pos_ptr,  # fp32 [N*3]
    vel_ptr,  # fp32 [N*3]
    mass_ptr,  # fp32 [N]
    heat_ptr,  # fp32 [N]
    rho_ptr,  # fp32 [num_cells] atomic
    mom_ptr,  # fp32 [num_cells*3] atomic
    e_int_ptr,  # fp32 [num_cells] atomic
    N,
    grid_x: tl.constexpr,
    grid_y: tl.constexpr,
    grid_z: tl.constexpr,
    inv_dx,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < N

    px = tl.load(pos_ptr + i * 3 + 0, mask=m, other=0.0)
    py = tl.load(pos_ptr + i * 3 + 1, mask=m, other=0.0)
    pz = tl.load(pos_ptr + i * 3 + 2, mask=m, other=0.0)
    vx = tl.load(vel_ptr + i * 3 + 0, mask=m, other=0.0)
    vy = tl.load(vel_ptr + i * 3 + 1, mask=m, other=0.0)
    vz = tl.load(vel_ptr + i * 3 + 2, mask=m, other=0.0)
    mass = tl.load(mass_ptr + i, mask=m, other=0.0)
    e_int = tl.load(heat_ptr + i, mask=m, other=0.0)  # Metal semantics: heat only

    gx = px * inv_dx
    gy = py * inv_dx
    gz = pz * inv_dx
    gx = gx - float(grid_x) * tl.floor(gx / float(grid_x))
    gy = gy - float(grid_y) * tl.floor(gy / float(grid_y))
    gz = gz - float(grid_z) * tl.floor(gz / float(grid_z))

    bx = tl.floor(gx).to(tl.int32)
    by = tl.floor(gy).to(tl.int32)
    bz = tl.floor(gz).to(tl.int32)
    fx = gx - bx.to(tl.float32)
    fy = gy - by.to(tl.float32)
    fz = gz - bz.to(tl.float32)

    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    wx1 = fx
    wy1 = fy
    wz1 = fz

    x0 = bx
    y0 = by
    z0 = bz
    x1 = (bx + 1) % grid_x
    y1 = (by + 1) % grid_y
    z1 = (bz + 1) % grid_z

    yz = grid_y * grid_z

    def lin(ix, iy, iz):
        return ix * yz + iy * grid_z + iz

    idx000 = lin(x0, y0, z0)
    idx100 = lin(x1, y0, z0)
    idx010 = lin(x0, y1, z0)
    idx110 = lin(x1, y1, z0)
    idx001 = lin(x0, y0, z1)
    idx101 = lin(x1, y0, z1)
    idx011 = lin(x0, y1, z1)
    idx111 = lin(x1, y1, z1)

    w000 = wx0 * wy0 * wz0
    w100 = wx1 * wy0 * wz0
    w010 = wx0 * wy1 * wz0
    w110 = wx1 * wy1 * wz0
    w001 = wx0 * wy0 * wz1
    w101 = wx1 * wy0 * wz1
    w011 = wx0 * wy1 * wz1
    w111 = wx1 * wy1 * wz1

    inv_vol = inv_dx * inv_dx * inv_dx
    rho_c = mass * inv_vol
    mx = mass * vx * inv_vol
    my = mass * vy * inv_vol
    mz = mass * vz * inv_vol
    e_c = e_int * inv_vol

    def add_corner(idx, w):
        tl.atomic_add(rho_ptr + idx, rho_c * w, mask=m)
        tl.atomic_add(mom_ptr + idx * 3 + 0, mx * w, mask=m)
        tl.atomic_add(mom_ptr + idx * 3 + 1, my * w, mask=m)
        tl.atomic_add(mom_ptr + idx * 3 + 2, mz * w, mask=m)
        tl.atomic_add(e_int_ptr + idx, e_c * w, mask=m)

    add_corner(idx000, w000)
    add_corner(idx100, w100)
    add_corner(idx010, w010)
    add_corner(idx110, w110)
    add_corner(idx001, w001)
    add_corner(idx101, w101)
    add_corner(idx011, w011)
    add_corner(idx111, w111)


# =============================================================================
# Gas RK2 (Rusanov flux + pressure work + thermal conduction) with NaN poisoning
# =============================================================================


@triton.jit
def _admissible_U5(
    rho,
    mx,
    my,
    mz,
    e_int,
    gamma,
    rho_min,
):
    # Mirrors `admissible_U5` in Metal.
    f32_eps = 1.1920929e-7
    rho_eps = tl.maximum(rho_min, 0.0)
    e_eps = 4.0 * rho_eps * f32_eps
    e_int_max = 10.0 * rho_eps

    finite = (
        tl.isfinite(rho)
        & tl.isfinite(mx)
        & tl.isfinite(my)
        & tl.isfinite(mz)
        & tl.isfinite(e_int)
    )
    low = tl.abs(rho) <= rho_eps
    mom_ok_low = (
        (tl.abs(mx) <= rho_eps) & (tl.abs(my) <= rho_eps) & (tl.abs(mz) <= rho_eps)
    )
    e_ok_low = (e_int >= -e_eps) & (e_int <= e_int_max)
    ok_low = low & mom_ok_low & e_ok_low

    ok_norm = (rho > rho_eps) & (e_int >= 0.0)
    return finite & (ok_low | ok_norm) & (gamma > 1.0)


@triton.jit
def _primitives_from_U(
    rho,
    mx,
    my,
    mz,
    e_int,
    gamma,
    c_v,
    rho_min,
):
    # Mirrors `primitives_from_U` in Metal.
    f32_eps = 1.1920929e-7
    rho_eps = tl.maximum(rho_min, 0.0)
    e_eps = 4.0 * rho_eps * f32_eps
    e_int_max = 10.0 * rho_eps

    nan = float("nan")

    finite = (
        tl.isfinite(rho)
        & tl.isfinite(mx)
        & tl.isfinite(my)
        & tl.isfinite(mz)
        & tl.isfinite(e_int)
    )
    ok_params = (gamma > 1.0) & (c_v > 0.0) & tl.isfinite(gamma) & tl.isfinite(c_v)

    low = tl.abs(rho) <= rho_eps
    mom_ok_low = (
        (tl.abs(mx) <= rho_eps) & (tl.abs(my) <= rho_eps) & (tl.abs(mz) <= rho_eps)
    )
    e_ok_low = (e_int >= -e_eps) & (e_int <= e_int_max)
    ok_low = low & mom_ok_low & e_ok_low
    ok_norm = (rho > 0.0) & (e_int >= 0.0)
    ok = finite & ok_params & (ok_low | ok_norm)

    rho_safe = tl.where(low, rho_eps, rho)
    u_x = mx / rho_safe
    u_y = my / rho_safe
    u_z = mz / rho_safe

    e_used = tl.where(low, tl.maximum(e_int, 0.0), e_int)
    p = (gamma - 1.0) * e_used
    T = e_used / (rho_safe * c_v)
    c = tl.sqrt(gamma * p / rho_safe)
    speed = tl.sqrt(u_x * u_x + u_y * u_y + u_z * u_z) + c

    rho_safe = tl.where(ok, rho_safe, nan)
    u_x = tl.where(ok, u_x, nan)
    u_y = tl.where(ok, u_y, nan)
    u_z = tl.where(ok, u_z, nan)
    p = tl.where(ok, p, nan)
    T = tl.where(ok, T, nan)
    c = tl.where(ok, c, nan)
    speed = tl.where(ok, speed, nan)
    return rho_safe, u_x, u_y, u_z, p, T, c, speed


@triton.jit
def _inviscid_flux_dir(
    dir: tl.constexpr,
    rho,
    mx,
    my,
    mz,
    e_int,
    u_x,
    u_y,
    u_z,
    p,
):
    if dir == 0:
        u_d = u_x
        mom_d = mx
    elif dir == 1:
        u_d = u_y
        mom_d = my
    else:
        u_d = u_z
        mom_d = mz

    frho = mom_d
    if dir == 0:
        fmom_x = mx * u_d + p
        fmom_y = my * u_d
        fmom_z = mz * u_d
    elif dir == 1:
        fmom_x = mx * u_d
        fmom_y = my * u_d + p
        fmom_z = mz * u_d
    else:
        fmom_x = mx * u_d
        fmom_y = my * u_d
        fmom_z = mz * u_d + p
    fe = e_int * u_d
    return frho, fmom_x, fmom_y, fmom_z, fe


@triton.jit
def _rusanov_flux(
    frho_L,
    fx_L,
    fy_L,
    fz_L,
    fe_L,
    frho_R,
    fx_R,
    fy_R,
    fz_R,
    fe_R,
    rho_L,
    mx_L,
    my_L,
    mz_L,
    e_L,
    rho_R,
    mx_R,
    my_R,
    mz_R,
    e_R,
    smax,
):
    # 0.5*(F_L+F_R) - 0.5*smax*(U_R-U_L)
    half = 0.5
    frho = half * (frho_L + frho_R) - half * smax * (rho_R - rho_L)
    fx = half * (fx_L + fx_R) - half * smax * (mx_R - mx_L)
    fy = half * (fy_L + fy_R) - half * smax * (my_R - my_L)
    fz = half * (fz_L + fz_R) - half * smax * (mz_R - mz_L)
    fe = half * (fe_L + fe_R) - half * smax * (e_R - e_L)
    return frho, fx, fy, fz, fe


@triton.autotune(configs=_GAS_CONFIGS, key=["num_cells"])
@triton.jit
def gas_rk2_stage1_kernel(
    rho0_ptr,
    mom0_ptr,
    e0_ptr,
    rho1_ptr,
    mom1_ptr,
    e1_ptr,
    k1_rho_ptr,
    k1_mom_ptr,
    k1_e_ptr,
    num_cells,
    grid_x: tl.constexpr,
    grid_y: tl.constexpr,
    grid_z: tl.constexpr,
    dx,
    dt,
    gamma,
    c_v,
    rho_min,
    k_thermal,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < num_cells

    yz = grid_y * grid_z
    i = idx // yz
    rem = idx - i * yz
    j = rem // grid_z
    k = rem - j * grid_z

    ixm = (i + grid_x - 1) % grid_x
    ixp = (i + 1) % grid_x
    iym = (j + grid_y - 1) % grid_y
    iyp = (j + 1) % grid_y
    izm = (k + grid_z - 1) % grid_z
    izp = (k + 1) % grid_z

    def lin(ii, jj, kk):
        return ii * yz + jj * grid_z + kk

    c = idx
    xm = lin(ixm, j, k)
    xp = lin(ixp, j, k)
    ym = lin(i, iym, k)
    yp = lin(i, iyp, k)
    zm = lin(i, j, izm)
    zp = lin(i, j, izp)

    def load_U(idc):
        rho = tl.load(rho0_ptr + idc, mask=m, other=0.0)
        mx = tl.load(mom0_ptr + idc * 3 + 0, mask=m, other=0.0)
        my = tl.load(mom0_ptr + idc * 3 + 1, mask=m, other=0.0)
        mz = tl.load(mom0_ptr + idc * 3 + 2, mask=m, other=0.0)
        e = tl.load(e0_ptr + idc, mask=m, other=0.0)
        return rho, mx, my, mz, e

    rho_c, mx_c, my_c, mz_c, e_c = load_U(c)
    rho_xm, mx_xm, my_xm, mz_xm, e_xm = load_U(xm)
    rho_xp, mx_xp, my_xp, mz_xp, e_xp = load_U(xp)
    rho_ym, mx_ym, my_ym, mz_ym, e_ym = load_U(ym)
    rho_yp, mx_yp, my_yp, mz_yp, e_yp = load_U(yp)
    rho_zm, mx_zm, my_zm, mz_zm, e_zm = load_U(zm)
    rho_zp, mx_zp, my_zp, mz_zp, e_zp = load_U(zp)

    ok = (
        _admissible_U5(rho_c, mx_c, my_c, mz_c, e_c, gamma, rho_min)
        & _admissible_U5(rho_xm, mx_xm, my_xm, mz_xm, e_xm, gamma, rho_min)
        & _admissible_U5(rho_xp, mx_xp, my_xp, mz_xp, e_xp, gamma, rho_min)
        & _admissible_U5(rho_ym, mx_ym, my_ym, mz_ym, e_ym, gamma, rho_min)
        & _admissible_U5(rho_yp, mx_yp, my_yp, mz_yp, e_yp, gamma, rho_min)
        & _admissible_U5(rho_zm, mx_zm, my_zm, mz_zm, e_zm, gamma, rho_min)
        & _admissible_U5(rho_zp, mx_zp, my_zp, mz_zp, e_zp, gamma, rho_min)
    )

    nan = float("nan")
    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx

    # Primitive recovery
    rho_s, ux_c, uy_c, uz_c, p_c, T_c, _c_c, sp_c = _primitives_from_U(
        rho_c, mx_c, my_c, mz_c, e_c, gamma, c_v, rho_min
    )
    _, ux_xm, uy_xm, uz_xm, p_xm, T_xm, _c_xm, sp_xm = _primitives_from_U(
        rho_xm, mx_xm, my_xm, mz_xm, e_xm, gamma, c_v, rho_min
    )
    _, ux_xp, uy_xp, uz_xp, p_xp, T_xp, _c_xp, sp_xp = _primitives_from_U(
        rho_xp, mx_xp, my_xp, mz_xp, e_xp, gamma, c_v, rho_min
    )
    _, ux_ym, uy_ym, uz_ym, p_ym, T_ym, _c_ym, sp_ym = _primitives_from_U(
        rho_ym, mx_ym, my_ym, mz_ym, e_ym, gamma, c_v, rho_min
    )
    _, ux_yp, uy_yp, uz_yp, p_yp, T_yp, _c_yp, sp_yp = _primitives_from_U(
        rho_yp, mx_yp, my_yp, mz_yp, e_yp, gamma, c_v, rho_min
    )
    _, ux_zm, uy_zm, uz_zm, p_zm, T_zm, _c_zm, sp_zm = _primitives_from_U(
        rho_zm, mx_zm, my_zm, mz_zm, e_zm, gamma, c_v, rho_min
    )
    _, ux_zp, uy_zp, uz_zp, p_zp, T_zp, _c_zp, sp_zp = _primitives_from_U(
        rho_zp, mx_zp, my_zp, mz_zp, e_zp, gamma, c_v, rho_min
    )

    # Fluxes and LLF/Rusanov faces.
    frho_xm, fx_xm, fy_xm, fz_xm, fe_xm = _inviscid_flux_dir(
        0, rho_xm, mx_xm, my_xm, mz_xm, e_xm, ux_xm, uy_xm, uz_xm, p_xm
    )
    frho_cx, fx_cx, fy_cx, fz_cx, fe_cx = _inviscid_flux_dir(
        0, rho_c, mx_c, my_c, mz_c, e_c, ux_c, uy_c, uz_c, p_c
    )
    frho_xp, fx_xp, fy_xp, fz_xp, fe_xp = _inviscid_flux_dir(
        0, rho_xp, mx_xp, my_xp, mz_xp, e_xp, ux_xp, uy_xp, uz_xp, p_xp
    )

    smax_xm = tl.maximum(sp_xm, sp_c)
    smax_xp = tl.maximum(sp_c, sp_xp)
    f_rho_xm, f_mx_xm, f_my_xm, f_mz_xm, f_e_xm = _rusanov_flux(
        frho_xm,
        fx_xm,
        fy_xm,
        fz_xm,
        fe_xm,
        frho_cx,
        fx_cx,
        fy_cx,
        fz_cx,
        fe_cx,
        rho_xm,
        mx_xm,
        my_xm,
        mz_xm,
        e_xm,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        smax_xm,
    )
    f_rho_xp, f_mx_xp, f_my_xp, f_mz_xp, f_e_xp = _rusanov_flux(
        frho_cx,
        fx_cx,
        fy_cx,
        fz_cx,
        fe_cx,
        frho_xp,
        fx_xp,
        fy_xp,
        fz_xp,
        fe_xp,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        rho_xp,
        mx_xp,
        my_xp,
        mz_xp,
        e_xp,
        smax_xp,
    )

    frho_ym, fx_ym, fy_ym, fz_ym, fe_ym = _inviscid_flux_dir(
        1, rho_ym, mx_ym, my_ym, mz_ym, e_ym, ux_ym, uy_ym, uz_ym, p_ym
    )
    frho_cy, fx_cy, fy_cy, fz_cy, fe_cy = _inviscid_flux_dir(
        1, rho_c, mx_c, my_c, mz_c, e_c, ux_c, uy_c, uz_c, p_c
    )
    frho_yp, fx_yp, fy_yp, fz_yp, fe_yp = _inviscid_flux_dir(
        1, rho_yp, mx_yp, my_yp, mz_yp, e_yp, ux_yp, uy_yp, uz_yp, p_yp
    )

    smax_ym = tl.maximum(sp_ym, sp_c)
    smax_yp = tl.maximum(sp_c, sp_yp)
    f_rho_ym, f_mx_ym, f_my_ym, f_mz_ym, f_e_ym = _rusanov_flux(
        frho_ym,
        fx_ym,
        fy_ym,
        fz_ym,
        fe_ym,
        frho_cy,
        fx_cy,
        fy_cy,
        fz_cy,
        fe_cy,
        rho_ym,
        mx_ym,
        my_ym,
        mz_ym,
        e_ym,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        smax_ym,
    )
    f_rho_yp, f_mx_yp, f_my_yp, f_mz_yp, f_e_yp = _rusanov_flux(
        frho_cy,
        fx_cy,
        fy_cy,
        fz_cy,
        fe_cy,
        frho_yp,
        fx_yp,
        fy_yp,
        fz_yp,
        fe_yp,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        rho_yp,
        mx_yp,
        my_yp,
        mz_yp,
        e_yp,
        smax_yp,
    )

    frho_zm, fx_zm, fy_zm, fz_zm, fe_zm = _inviscid_flux_dir(
        2, rho_zm, mx_zm, my_zm, mz_zm, e_zm, ux_zm, uy_zm, uz_zm, p_zm
    )
    frho_cz, fx_cz, fy_cz, fz_cz, fe_cz = _inviscid_flux_dir(
        2, rho_c, mx_c, my_c, mz_c, e_c, ux_c, uy_c, uz_c, p_c
    )
    frho_zp, fx_zp, fy_zp, fz_zp, fe_zp = _inviscid_flux_dir(
        2, rho_zp, mx_zp, my_zp, mz_zp, e_zp, ux_zp, uy_zp, uz_zp, p_zp
    )

    smax_zm = tl.maximum(sp_zm, sp_c)
    smax_zp = tl.maximum(sp_c, sp_zp)
    f_rho_zm, f_mx_zm, f_my_zm, f_mz_zm, f_e_zm = _rusanov_flux(
        frho_zm,
        fx_zm,
        fy_zm,
        fz_zm,
        fe_zm,
        frho_cz,
        fx_cz,
        fy_cz,
        fz_cz,
        fe_cz,
        rho_zm,
        mx_zm,
        my_zm,
        mz_zm,
        e_zm,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        smax_zm,
    )
    f_rho_zp, f_mx_zp, f_my_zp, f_mz_zp, f_e_zp = _rusanov_flux(
        frho_cz,
        fx_cz,
        fy_cz,
        fz_cz,
        fe_cz,
        frho_zp,
        fx_zp,
        fy_zp,
        fz_zp,
        fe_zp,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        rho_zp,
        mx_zp,
        my_zp,
        mz_zp,
        e_zp,
        smax_zp,
    )

    div_frho = (
        f_rho_xp - f_rho_xm + f_rho_yp - f_rho_ym + f_rho_zp - f_rho_zm
    ) * inv_dx
    div_fmx = (f_mx_xp - f_mx_xm + f_mx_yp - f_mx_ym + f_mx_zp - f_mx_zm) * inv_dx
    div_fmy = (f_my_xp - f_my_xm + f_my_yp - f_my_ym + f_my_zp - f_my_zm) * inv_dx
    div_fmz = (f_mz_xp - f_mz_xm + f_mz_yp - f_mz_ym + f_mz_zp - f_mz_zm) * inv_dx
    div_fe = (f_e_xp - f_e_xm + f_e_yp - f_e_ym + f_e_zp - f_e_zm) * inv_dx

    div_u = ((ux_xp - ux_xm) + (uy_yp - uy_ym) + (uz_zp - uz_zm)) * (0.5 * inv_dx)
    pressure_work = -p_c * div_u
    lap_T = (T_xp + T_xm + T_yp + T_ym + T_zp + T_zm - 6.0 * T_c) * inv_dx2
    conduction = k_thermal * lap_T

    dr = -div_frho
    dmx = -div_fmx
    dmy = -div_fmy
    dmz = -div_fmz
    de = -div_fe + pressure_work + conduction

    dr = tl.where(ok, dr, nan)
    dmx = tl.where(ok, dmx, nan)
    dmy = tl.where(ok, dmy, nan)
    dmz = tl.where(ok, dmz, nan)
    de = tl.where(ok, de, nan)

    rho1 = rho_c + dt * dr
    mx1 = mx_c + dt * dmx
    my1 = my_c + dt * dmy
    mz1 = mz_c + dt * dmz
    e1 = e_c + dt * de

    ok1 = ok & _admissible_U5(rho1, mx1, my1, mz1, e1, gamma, rho_min)
    rho1 = tl.where(ok1, rho1, nan)
    mx1 = tl.where(ok1, mx1, nan)
    my1 = tl.where(ok1, my1, nan)
    mz1 = tl.where(ok1, mz1, nan)
    e1 = tl.where(ok1, e1, nan)

    tl.store(k1_rho_ptr + idx, dr, mask=m)
    tl.store(k1_mom_ptr + idx * 3 + 0, dmx, mask=m)
    tl.store(k1_mom_ptr + idx * 3 + 1, dmy, mask=m)
    tl.store(k1_mom_ptr + idx * 3 + 2, dmz, mask=m)
    tl.store(k1_e_ptr + idx, de, mask=m)

    tl.store(rho1_ptr + idx, rho1, mask=m)
    tl.store(mom1_ptr + idx * 3 + 0, mx1, mask=m)
    tl.store(mom1_ptr + idx * 3 + 1, my1, mask=m)
    tl.store(mom1_ptr + idx * 3 + 2, mz1, mask=m)
    tl.store(e1_ptr + idx, e1, mask=m)


@triton.autotune(configs=_GAS_CONFIGS, key=["num_cells"])
@triton.jit
def gas_rk2_stage2_kernel(
    rho0_ptr,
    mom0_ptr,
    e0_ptr,
    rho1_ptr,
    mom1_ptr,
    e1_ptr,
    k1_rho_ptr,
    k1_mom_ptr,
    k1_e_ptr,
    rho2_ptr,
    mom2_ptr,
    e2_ptr,
    num_cells,
    grid_x: tl.constexpr,
    grid_y: tl.constexpr,
    grid_z: tl.constexpr,
    dx,
    dt,
    gamma,
    c_v,
    rho_min,
    k_thermal,
    BLOCK: tl.constexpr,
):
    # Stage 2 mirrors stage1, but evaluates RHS at U1 and combines with k1.
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < num_cells

    yz = grid_y * grid_z
    i = idx // yz
    rem = idx - i * yz
    j = rem // grid_z
    k = rem - j * grid_z

    ixm = (i + grid_x - 1) % grid_x
    ixp = (i + 1) % grid_x
    iym = (j + grid_y - 1) % grid_y
    iyp = (j + 1) % grid_y
    izm = (k + grid_z - 1) % grid_z
    izp = (k + 1) % grid_z

    def lin(ii, jj, kk):
        return ii * yz + jj * grid_z + kk

    c = idx
    xm = lin(ixm, j, k)
    xp = lin(ixp, j, k)
    ym = lin(i, iym, k)
    yp = lin(i, iyp, k)
    zm = lin(i, j, izm)
    zp = lin(i, j, izp)

    def load_U(ptr_rho, ptr_mom, ptr_e, idc):
        rho = tl.load(ptr_rho + idc, mask=m, other=0.0)
        mx = tl.load(ptr_mom + idc * 3 + 0, mask=m, other=0.0)
        my = tl.load(ptr_mom + idc * 3 + 1, mask=m, other=0.0)
        mz = tl.load(ptr_mom + idc * 3 + 2, mask=m, other=0.0)
        e = tl.load(ptr_e + idc, mask=m, other=0.0)
        return rho, mx, my, mz, e

    rho0, mx0, my0, mz0, e0 = load_U(rho0_ptr, mom0_ptr, e0_ptr, c)
    rho_c, mx_c, my_c, mz_c, e_c = load_U(rho1_ptr, mom1_ptr, e1_ptr, c)
    rho_xm, mx_xm, my_xm, mz_xm, e_xm = load_U(rho1_ptr, mom1_ptr, e1_ptr, xm)
    rho_xp, mx_xp, my_xp, mz_xp, e_xp = load_U(rho1_ptr, mom1_ptr, e1_ptr, xp)
    rho_ym, mx_ym, my_ym, mz_ym, e_ym = load_U(rho1_ptr, mom1_ptr, e1_ptr, ym)
    rho_yp, mx_yp, my_yp, mz_yp, e_yp = load_U(rho1_ptr, mom1_ptr, e1_ptr, yp)
    rho_zm, mx_zm, my_zm, mz_zm, e_zm = load_U(rho1_ptr, mom1_ptr, e1_ptr, zm)
    rho_zp, mx_zp, my_zp, mz_zp, e_zp = load_U(rho1_ptr, mom1_ptr, e1_ptr, zp)

    ok = (
        _admissible_U5(rho_c, mx_c, my_c, mz_c, e_c, gamma, rho_min)
        & _admissible_U5(rho_xm, mx_xm, my_xm, mz_xm, e_xm, gamma, rho_min)
        & _admissible_U5(rho_xp, mx_xp, my_xp, mz_xp, e_xp, gamma, rho_min)
        & _admissible_U5(rho_ym, mx_ym, my_ym, mz_ym, e_ym, gamma, rho_min)
        & _admissible_U5(rho_yp, mx_yp, my_yp, mz_yp, e_yp, gamma, rho_min)
        & _admissible_U5(rho_zm, mx_zm, my_zm, mz_zm, e_zm, gamma, rho_min)
        & _admissible_U5(rho_zp, mx_zp, my_zp, mz_zp, e_zp, gamma, rho_min)
    )

    nan = float("nan")
    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx

    rho_s, ux_c, uy_c, uz_c, p_c, T_c, _c_c, sp_c = _primitives_from_U(
        rho_c, mx_c, my_c, mz_c, e_c, gamma, c_v, rho_min
    )
    _, ux_xm, uy_xm, uz_xm, p_xm, T_xm, _c_xm, sp_xm = _primitives_from_U(
        rho_xm, mx_xm, my_xm, mz_xm, e_xm, gamma, c_v, rho_min
    )
    _, ux_xp, uy_xp, uz_xp, p_xp, T_xp, _c_xp, sp_xp = _primitives_from_U(
        rho_xp, mx_xp, my_xp, mz_xp, e_xp, gamma, c_v, rho_min
    )
    _, ux_ym, uy_ym, uz_ym, p_ym, T_ym, _c_ym, sp_ym = _primitives_from_U(
        rho_ym, mx_ym, my_ym, mz_ym, e_ym, gamma, c_v, rho_min
    )
    _, ux_yp, uy_yp, uz_yp, p_yp, T_yp, _c_yp, sp_yp = _primitives_from_U(
        rho_yp, mx_yp, my_yp, mz_yp, e_yp, gamma, c_v, rho_min
    )
    _, ux_zm, uy_zm, uz_zm, p_zm, T_zm, _c_zm, sp_zm = _primitives_from_U(
        rho_zm, mx_zm, my_zm, mz_zm, e_zm, gamma, c_v, rho_min
    )
    _, ux_zp, uy_zp, uz_zp, p_zp, T_zp, _c_zp, sp_zp = _primitives_from_U(
        rho_zp, mx_zp, my_zp, mz_zp, e_zp, gamma, c_v, rho_min
    )

    frho_xm, fx_xm, fy_xm, fz_xm, fe_xm = _inviscid_flux_dir(
        0, rho_xm, mx_xm, my_xm, mz_xm, e_xm, ux_xm, uy_xm, uz_xm, p_xm
    )
    frho_cx, fx_cx, fy_cx, fz_cx, fe_cx = _inviscid_flux_dir(
        0, rho_c, mx_c, my_c, mz_c, e_c, ux_c, uy_c, uz_c, p_c
    )
    frho_xp, fx_xp, fy_xp, fz_xp, fe_xp = _inviscid_flux_dir(
        0, rho_xp, mx_xp, my_xp, mz_xp, e_xp, ux_xp, uy_xp, uz_xp, p_xp
    )
    smax_xm = tl.maximum(sp_xm, sp_c)
    smax_xp = tl.maximum(sp_c, sp_xp)
    f_rho_xm, f_mx_xm, f_my_xm, f_mz_xm, f_e_xm = _rusanov_flux(
        frho_xm,
        fx_xm,
        fy_xm,
        fz_xm,
        fe_xm,
        frho_cx,
        fx_cx,
        fy_cx,
        fz_cx,
        fe_cx,
        rho_xm,
        mx_xm,
        my_xm,
        mz_xm,
        e_xm,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        smax_xm,
    )
    f_rho_xp, f_mx_xp, f_my_xp, f_mz_xp, f_e_xp = _rusanov_flux(
        frho_cx,
        fx_cx,
        fy_cx,
        fz_cx,
        fe_cx,
        frho_xp,
        fx_xp,
        fy_xp,
        fz_xp,
        fe_xp,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        rho_xp,
        mx_xp,
        my_xp,
        mz_xp,
        e_xp,
        smax_xp,
    )

    frho_ym, fx_ym, fy_ym, fz_ym, fe_ym = _inviscid_flux_dir(
        1, rho_ym, mx_ym, my_ym, mz_ym, e_ym, ux_ym, uy_ym, uz_ym, p_ym
    )
    frho_cy, fx_cy, fy_cy, fz_cy, fe_cy = _inviscid_flux_dir(
        1, rho_c, mx_c, my_c, mz_c, e_c, ux_c, uy_c, uz_c, p_c
    )
    frho_yp, fx_yp, fy_yp, fz_yp, fe_yp = _inviscid_flux_dir(
        1, rho_yp, mx_yp, my_yp, mz_yp, e_yp, ux_yp, uy_yp, uz_yp, p_yp
    )
    smax_ym = tl.maximum(sp_ym, sp_c)
    smax_yp = tl.maximum(sp_c, sp_yp)
    f_rho_ym, f_mx_ym, f_my_ym, f_mz_ym, f_e_ym = _rusanov_flux(
        frho_ym,
        fx_ym,
        fy_ym,
        fz_ym,
        fe_ym,
        frho_cy,
        fx_cy,
        fy_cy,
        fz_cy,
        fe_cy,
        rho_ym,
        mx_ym,
        my_ym,
        mz_ym,
        e_ym,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        smax_ym,
    )
    f_rho_yp, f_mx_yp, f_my_yp, f_mz_yp, f_e_yp = _rusanov_flux(
        frho_cy,
        fx_cy,
        fy_cy,
        fz_cy,
        fe_cy,
        frho_yp,
        fx_yp,
        fy_yp,
        fz_yp,
        fe_yp,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        rho_yp,
        mx_yp,
        my_yp,
        mz_yp,
        e_yp,
        smax_yp,
    )

    frho_zm, fx_zm, fy_zm, fz_zm, fe_zm = _inviscid_flux_dir(
        2, rho_zm, mx_zm, my_zm, mz_zm, e_zm, ux_zm, uy_zm, uz_zm, p_zm
    )
    frho_cz, fx_cz, fy_cz, fz_cz, fe_cz = _inviscid_flux_dir(
        2, rho_c, mx_c, my_c, mz_c, e_c, ux_c, uy_c, uz_c, p_c
    )
    frho_zp, fx_zp, fy_zp, fz_zp, fe_zp = _inviscid_flux_dir(
        2, rho_zp, mx_zp, my_zp, mz_zp, e_zp, ux_zp, uy_zp, uz_zp, p_zp
    )
    smax_zm = tl.maximum(sp_zm, sp_c)
    smax_zp = tl.maximum(sp_c, sp_zp)
    f_rho_zm, f_mx_zm, f_my_zm, f_mz_zm, f_e_zm = _rusanov_flux(
        frho_zm,
        fx_zm,
        fy_zm,
        fz_zm,
        fe_zm,
        frho_cz,
        fx_cz,
        fy_cz,
        fz_cz,
        fe_cz,
        rho_zm,
        mx_zm,
        my_zm,
        mz_zm,
        e_zm,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        smax_zm,
    )
    f_rho_zp, f_mx_zp, f_my_zp, f_mz_zp, f_e_zp = _rusanov_flux(
        frho_cz,
        fx_cz,
        fy_cz,
        fz_cz,
        fe_cz,
        frho_zp,
        fx_zp,
        fy_zp,
        fz_zp,
        fe_zp,
        rho_c,
        mx_c,
        my_c,
        mz_c,
        e_c,
        rho_zp,
        mx_zp,
        my_zp,
        mz_zp,
        e_zp,
        smax_zp,
    )

    div_frho = (
        f_rho_xp - f_rho_xm + f_rho_yp - f_rho_ym + f_rho_zp - f_rho_zm
    ) * inv_dx
    div_fmx = (f_mx_xp - f_mx_xm + f_mx_yp - f_mx_ym + f_mx_zp - f_mx_zm) * inv_dx
    div_fmy = (f_my_xp - f_my_xm + f_my_yp - f_my_ym + f_my_zp - f_my_zm) * inv_dx
    div_fmz = (f_mz_xp - f_mz_xm + f_mz_yp - f_mz_ym + f_mz_zp - f_mz_zm) * inv_dx
    div_fe = (f_e_xp - f_e_xm + f_e_yp - f_e_ym + f_e_zp - f_e_zm) * inv_dx

    div_u = ((ux_xp - ux_xm) + (uy_yp - uy_ym) + (uz_zp - uz_zm)) * (0.5 * inv_dx)
    pressure_work = -p_c * div_u
    lap_T = (T_xp + T_xm + T_yp + T_ym + T_zp + T_zm - 6.0 * T_c) * inv_dx2
    conduction = k_thermal * lap_T

    dr2 = -div_frho
    dmx2 = -div_fmx
    dmy2 = -div_fmy
    dmz2 = -div_fmz
    de2 = -div_fe + pressure_work + conduction

    dr2 = tl.where(ok, dr2, nan)
    dmx2 = tl.where(ok, dmx2, nan)
    dmy2 = tl.where(ok, dmy2, nan)
    dmz2 = tl.where(ok, dmz2, nan)
    de2 = tl.where(ok, de2, nan)

    k1_r = tl.load(k1_rho_ptr + idx, mask=m, other=nan)
    k1_mx = tl.load(k1_mom_ptr + idx * 3 + 0, mask=m, other=nan)
    k1_my = tl.load(k1_mom_ptr + idx * 3 + 1, mask=m, other=nan)
    k1_mz = tl.load(k1_mom_ptr + idx * 3 + 2, mask=m, other=nan)
    k1_e = tl.load(k1_e_ptr + idx, mask=m, other=nan)

    half = 0.5
    rho2 = rho0 + half * dt * (k1_r + dr2)
    mx2 = mx0 + half * dt * (k1_mx + dmx2)
    my2 = my0 + half * dt * (k1_my + dmy2)
    mz2 = mz0 + half * dt * (k1_mz + dmz2)
    e2 = e0 + half * dt * (k1_e + de2)

    ok2 = ok & _admissible_U5(rho2, mx2, my2, mz2, e2, gamma, rho_min)
    rho2 = tl.where(ok2, rho2, nan)
    mx2 = tl.where(ok2, mx2, nan)
    my2 = tl.where(ok2, my2, nan)
    mz2 = tl.where(ok2, mz2, nan)
    e2 = tl.where(ok2, e2, nan)

    tl.store(rho2_ptr + idx, rho2, mask=m)
    tl.store(mom2_ptr + idx * 3 + 0, mx2, mask=m)
    tl.store(mom2_ptr + idx * 3 + 1, my2, mask=m)
    tl.store(mom2_ptr + idx * 3 + 2, mz2, mask=m)
    tl.store(e2_ptr + idx, e2, mask=m)


# =============================================================================
# PIC gather + particle update
# =============================================================================


@triton.autotune(configs=_GATHER_CONFIGS, key=["N"])
@triton.jit
def pic_gather_update_particles_kernel(
    pos_ptr,  # fp32 [N*3]
    mass_ptr,  # fp32 [N]
    pos_out_ptr,  # fp32 [N*3]
    vel_out_ptr,  # fp32 [N*3]
    heat_out_ptr,  # fp32 [N]
    rho_ptr,  # fp32 [num_cells]
    mom_ptr,  # fp32 [num_cells*3]
    e_int_ptr,  # fp32 [num_cells]
    phi_ptr,  # fp32 [num_cells]
    N,
    grid_x: tl.constexpr,
    grid_y: tl.constexpr,
    grid_z: tl.constexpr,
    dx,
    dt,
    domain_x,
    domain_y,
    domain_z,
    gamma,
    c_v,
    rho_min,
    gravity_enabled: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    m = p < N

    nan = float("nan")

    px = tl.load(pos_ptr + p * 3 + 0, mask=m, other=0.0)
    py = tl.load(pos_ptr + p * 3 + 1, mask=m, other=0.0)
    pz = tl.load(pos_ptr + p * 3 + 2, mask=m, other=0.0)
    mass = tl.load(mass_ptr + p, mask=m, other=0.0)

    inv_dx = 1.0 / dx

    gx = px * inv_dx
    gy = py * inv_dx
    gz = pz * inv_dx
    gx = gx - float(grid_x) * tl.floor(gx / float(grid_x))
    gy = gy - float(grid_y) * tl.floor(gy / float(grid_y))
    gz = gz - float(grid_z) * tl.floor(gz / float(grid_z))

    bx = tl.floor(gx).to(tl.int32)
    by = tl.floor(gy).to(tl.int32)
    bz = tl.floor(gz).to(tl.int32)
    fx = gx - bx.to(tl.float32)
    fy = gy - by.to(tl.float32)
    fz = gz - bz.to(tl.float32)

    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    wx1 = fx
    wy1 = fy
    wz1 = fz

    x0 = bx
    y0 = by
    z0 = bz
    x1 = (bx + 1) % grid_x
    y1 = (by + 1) % grid_y
    z1 = (bz + 1) % grid_z

    yz = grid_y * grid_z

    def lin(ix, iy, iz):
        return ix * yz + iy * grid_z + iz

    idx000 = lin(x0, y0, z0)
    idx100 = lin(x1, y0, z0)
    idx010 = lin(x0, y1, z0)
    idx110 = lin(x1, y1, z0)
    idx001 = lin(x0, y0, z1)
    idx101 = lin(x1, y0, z1)
    idx011 = lin(x0, y1, z1)
    idx111 = lin(x1, y1, z1)

    w000 = wx0 * wy0 * wz0
    w100 = wx1 * wy0 * wz0
    w010 = wx0 * wy1 * wz0
    w110 = wx1 * wy1 * wz0
    w001 = wx0 * wy0 * wz1
    w101 = wx1 * wy0 * wz1
    w011 = wx0 * wy1 * wz1
    w111 = wx1 * wy1 * wz1

    def gather_s(field_ptr):
        return (
            tl.load(field_ptr + idx000, mask=m, other=0.0) * w000
            + tl.load(field_ptr + idx100, mask=m, other=0.0) * w100
            + tl.load(field_ptr + idx010, mask=m, other=0.0) * w010
            + tl.load(field_ptr + idx110, mask=m, other=0.0) * w110
            + tl.load(field_ptr + idx001, mask=m, other=0.0) * w001
            + tl.load(field_ptr + idx101, mask=m, other=0.0) * w101
            + tl.load(field_ptr + idx011, mask=m, other=0.0) * w011
            + tl.load(field_ptr + idx111, mask=m, other=0.0) * w111
        )

    rho = gather_s(rho_ptr)
    e_int = gather_s(e_int_ptr)

    # Note: `mom_ptr` is AoS in Metal but SoA in this kernel (flat *3). Gather each component.
    # We implement explicit component gathers by shifting the pointer.
    mx = (
        tl.load(mom_ptr + idx000 * 3 + 0, mask=m, other=0.0) * w000
        + tl.load(mom_ptr + idx100 * 3 + 0, mask=m, other=0.0) * w100
        + tl.load(mom_ptr + idx010 * 3 + 0, mask=m, other=0.0) * w010
        + tl.load(mom_ptr + idx110 * 3 + 0, mask=m, other=0.0) * w110
        + tl.load(mom_ptr + idx001 * 3 + 0, mask=m, other=0.0) * w001
        + tl.load(mom_ptr + idx101 * 3 + 0, mask=m, other=0.0) * w101
        + tl.load(mom_ptr + idx011 * 3 + 0, mask=m, other=0.0) * w011
        + tl.load(mom_ptr + idx111 * 3 + 0, mask=m, other=0.0) * w111
    )
    my = (
        tl.load(mom_ptr + idx000 * 3 + 1, mask=m, other=0.0) * w000
        + tl.load(mom_ptr + idx100 * 3 + 1, mask=m, other=0.0) * w100
        + tl.load(mom_ptr + idx010 * 3 + 1, mask=m, other=0.0) * w010
        + tl.load(mom_ptr + idx110 * 3 + 1, mask=m, other=0.0) * w110
        + tl.load(mom_ptr + idx001 * 3 + 1, mask=m, other=0.0) * w001
        + tl.load(mom_ptr + idx101 * 3 + 1, mask=m, other=0.0) * w101
        + tl.load(mom_ptr + idx011 * 3 + 1, mask=m, other=0.0) * w011
        + tl.load(mom_ptr + idx111 * 3 + 1, mask=m, other=0.0) * w111
    )
    mz = (
        tl.load(mom_ptr + idx000 * 3 + 2, mask=m, other=0.0) * w000
        + tl.load(mom_ptr + idx100 * 3 + 2, mask=m, other=0.0) * w100
        + tl.load(mom_ptr + idx010 * 3 + 2, mask=m, other=0.0) * w010
        + tl.load(mom_ptr + idx110 * 3 + 2, mask=m, other=0.0) * w110
        + tl.load(mom_ptr + idx001 * 3 + 2, mask=m, other=0.0) * w001
        + tl.load(mom_ptr + idx101 * 3 + 2, mask=m, other=0.0) * w101
        + tl.load(mom_ptr + idx011 * 3 + 2, mask=m, other=0.0) * w011
        + tl.load(mom_ptr + idx111 * 3 + 2, mask=m, other=0.0) * w111
    )

    # Vacuum semantics
    inv_rho = 1.0 / rho
    u_x = tl.where(rho > 0.0, mx * inv_rho, 0.0)
    u_y = tl.where(rho > 0.0, my * inv_rho, 0.0)
    u_z = tl.where(rho > 0.0, mz * inv_rho, 0.0)

    T = tl.where(rho > 0.0, e_int / (rho * c_v), 0.0)
    heat = tl.where(rho > 0.0, mass * c_v * T, 0.0)

    # Gravity: sample gradient via face interpolation of corner phi values.
    grad_x = 0.0
    grad_y = 0.0
    grad_z = 0.0
    if gravity_enabled:
        phi000 = tl.load(phi_ptr + idx000, mask=m, other=0.0)
        phi100 = tl.load(phi_ptr + idx100, mask=m, other=0.0)
        phi010 = tl.load(phi_ptr + idx010, mask=m, other=0.0)
        phi110 = tl.load(phi_ptr + idx110, mask=m, other=0.0)
        phi001 = tl.load(phi_ptr + idx001, mask=m, other=0.0)
        phi101 = tl.load(phi_ptr + idx101, mask=m, other=0.0)
        phi011 = tl.load(phi_ptr + idx011, mask=m, other=0.0)
        phi111 = tl.load(phi_ptr + idx111, mask=m, other=0.0)

        # Face values along x
        face_x0 = (
            phi000 * wy0 * wz0
            + phi010 * wy1 * wz0
            + phi001 * wy0 * wz1
            + phi011 * wy1 * wz1
        )
        face_x1 = (
            phi100 * wy0 * wz0
            + phi110 * wy1 * wz0
            + phi101 * wy0 * wz1
            + phi111 * wy1 * wz1
        )
        grad_x = (face_x1 - face_x0) * inv_dx

        # Face values along y
        face_y0 = (
            phi000 * wx0 * wz0
            + phi100 * wx1 * wz0
            + phi001 * wx0 * wz1
            + phi101 * wx1 * wz1
        )
        face_y1 = (
            phi010 * wx0 * wz0
            + phi110 * wx1 * wz0
            + phi011 * wx0 * wz1
            + phi111 * wx1 * wz1
        )
        grad_y = (face_y1 - face_y0) * inv_dx

        # Face values along z
        face_z0 = (
            phi000 * wx0 * wy0
            + phi100 * wx1 * wy0
            + phi010 * wx0 * wy1
            + phi110 * wx1 * wy1
        )
        face_z1 = (
            phi001 * wx0 * wy0
            + phi101 * wx1 * wy0
            + phi011 * wx0 * wy1
            + phi111 * wx1 * wy1
        )
        grad_z = (face_z1 - face_z0) * inv_dx

    gax = -grad_x
    gay = -grad_y
    gaz = -grad_z

    u_x = u_x + gax * dt
    u_y = u_y + gay * dt
    u_z = u_z + gaz * dt

    pos_nx = px + u_x * dt
    pos_ny = py + u_y * dt
    pos_nz = pz + u_z * dt

    # Periodic wrap into [0, domain)
    pos_nx = pos_nx - domain_x * tl.floor(pos_nx / domain_x)
    pos_ny = pos_ny - domain_y * tl.floor(pos_ny / domain_y)
    pos_nz = pos_nz - domain_z * tl.floor(pos_nz / domain_z)

    ok = (
        m
        & tl.isfinite(mass)
        & (mass > 0.0)
        & tl.isfinite(rho)
        & tl.isfinite(e_int)
        & (c_v > 0.0)
        & tl.isfinite(T)
        & tl.isfinite(heat)
        & (heat >= 0.0)
        & tl.isfinite(u_x)
        & tl.isfinite(u_y)
        & tl.isfinite(u_z)
        & tl.isfinite(pos_nx)
        & tl.isfinite(pos_ny)
        & tl.isfinite(pos_nz)
    )

    pos_nx = tl.where(ok, pos_nx, nan)
    pos_ny = tl.where(ok, pos_ny, nan)
    pos_nz = tl.where(ok, pos_nz, nan)
    u_x = tl.where(ok, u_x, nan)
    u_y = tl.where(ok, u_y, nan)
    u_z = tl.where(ok, u_z, nan)
    heat = tl.where(ok, heat, nan)

    tl.store(pos_out_ptr + p * 3 + 0, pos_nx, mask=m)
    tl.store(pos_out_ptr + p * 3 + 1, pos_ny, mask=m)
    tl.store(pos_out_ptr + p * 3 + 2, pos_nz, mask=m)
    tl.store(vel_out_ptr + p * 3 + 0, u_x, mask=m)
    tl.store(vel_out_ptr + p * 3 + 1, u_y, mask=m)
    tl.store(vel_out_ptr + p * 3 + 2, u_z, mask=m)
    tl.store(heat_out_ptr + p, heat, mask=m)


# =============================================================================
# Python wrappers (CUDA only)
# =============================================================================


def scatter_compute_cell_idx(
    pos: torch.Tensor, out_cell_idx: torch.Tensor, meta: SortScatterMeta
) -> None:
    _require_triton()
    if pos.dtype != torch.float32:
        raise TypeError("pos must be fp32")
    if out_cell_idx.dtype != torch.int32:
        raise TypeError("out_cell_idx must be int32")
    if not pos.is_cuda:
        raise RuntimeError("scatter_compute_cell_idx requires CUDA tensors")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must be [N,3], got {tuple(pos.shape)}")
    N = int(pos.shape[0])
    grid = lambda m: (triton.cdiv(N, m["BLOCK"]),)
    scatter_compute_cell_idx_kernel[grid](
        pos,
        out_cell_idx,
        N=N,
        grid_x=int(meta.grid_x),
        grid_y=int(meta.grid_y),
        grid_z=int(meta.grid_z),
        inv_dx=float(meta.inv_dx),
    )


def scatter_count_cells(cell_idx: torch.Tensor, out_cell_counts: torch.Tensor) -> None:
    _require_triton()
    if cell_idx.dtype != torch.int32:
        raise TypeError("cell_idx must be int32")
    if out_cell_counts.dtype != torch.int32:
        raise TypeError("out_cell_counts must be int32")
    N = int(cell_idx.numel())
    grid = lambda m: (triton.cdiv(N, m["BLOCK"]),)
    scatter_count_cells_kernel[grid](
        cell_idx,
        out_cell_counts,
        N=N,
    )


def scatter_reorder_particles(
    pos: torch.Tensor,
    vel: torch.Tensor,
    mass: torch.Tensor,
    heat: torch.Tensor,
    energy: torch.Tensor,
    cell_idx: torch.Tensor,
    cell_starts: torch.Tensor,
    cell_offsets: torch.Tensor,
    pos_out: torch.Tensor,
    vel_out: torch.Tensor,
    mass_out: torch.Tensor,
    heat_out: torch.Tensor,
    energy_out: torch.Tensor,
    original_idx_out: torch.Tensor,
) -> None:
    _require_triton()
    N = int(pos.shape[0])
    grid = lambda m: (triton.cdiv(N, m["BLOCK"]),)
    scatter_reorder_particles_kernel[grid](
        pos,
        vel,
        mass,
        heat,
        energy,
        cell_idx,
        cell_starts,
        cell_offsets,
        pos_out,
        vel_out,
        mass_out,
        heat_out,
        energy_out,
        original_idx_out,
        N=N,
    )


def scatter_sorted(
    pos: torch.Tensor,
    vel: torch.Tensor,
    mass: torch.Tensor,
    heat: torch.Tensor,
    rho_field: torch.Tensor,
    mom_field: torch.Tensor,
    e_int_field: torch.Tensor,
    meta: SortScatterMeta,
) -> None:
    _require_triton()
    if mom_field.ndim != 4 or mom_field.shape[-1] != 3:
        raise ValueError(
            f"mom_field must be [gx,gy,gz,3], got {tuple(mom_field.shape)}"
        )
    N = int(pos.shape[0])
    grid = lambda m: (triton.cdiv(N, m["BLOCK"]),)
    scatter_sorted_kernel[grid](
        pos,
        vel,
        mass,
        heat,
        rho_field,
        mom_field.view(-1),
        e_int_field,
        N=N,
        grid_x=int(meta.grid_x),
        grid_y=int(meta.grid_y),
        grid_z=int(meta.grid_z),
        inv_dx=float(meta.inv_dx),
    )


def gas_rk2_stage1(
    rho0: torch.Tensor,
    mom0: torch.Tensor,
    e0: torch.Tensor,
    rho1: torch.Tensor,
    mom1: torch.Tensor,
    e1: torch.Tensor,
    k1_rho: torch.Tensor,
    k1_mom: torch.Tensor,
    k1_e: torch.Tensor,
    *,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    dx: float,
    dt: float,
    gamma: float,
    c_v: float,
    rho_min: float,
    k_thermal: float,
) -> None:
    _require_triton()
    num_cells = int(rho0.numel())
    grid = lambda m: (triton.cdiv(num_cells, m["BLOCK"]),)
    gas_rk2_stage1_kernel[grid](
        rho0.view(-1),
        mom0.view(-1),
        e0.view(-1),
        rho1.view(-1),
        mom1.view(-1),
        e1.view(-1),
        k1_rho.view(-1),
        k1_mom.view(-1),
        k1_e.view(-1),
        num_cells=num_cells,
        grid_x=int(grid_x),
        grid_y=int(grid_y),
        grid_z=int(grid_z),
        dx=float(dx),
        dt=float(dt),
        gamma=float(gamma),
        c_v=float(c_v),
        rho_min=float(rho_min),
        k_thermal=float(k_thermal),
    )


def gas_rk2_stage2(
    rho0: torch.Tensor,
    mom0: torch.Tensor,
    e0: torch.Tensor,
    rho1: torch.Tensor,
    mom1: torch.Tensor,
    e1: torch.Tensor,
    k1_rho: torch.Tensor,
    k1_mom: torch.Tensor,
    k1_e: torch.Tensor,
    rho2: torch.Tensor,
    mom2: torch.Tensor,
    e2: torch.Tensor,
    *,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    dx: float,
    dt: float,
    gamma: float,
    c_v: float,
    rho_min: float,
    k_thermal: float,
) -> None:
    _require_triton()
    num_cells = int(rho0.numel())
    grid = lambda m: (triton.cdiv(num_cells, m["BLOCK"]),)
    gas_rk2_stage2_kernel[grid](
        rho0.view(-1),
        mom0.view(-1),
        e0.view(-1),
        rho1.view(-1),
        mom1.view(-1),
        e1.view(-1),
        k1_rho.view(-1),
        k1_mom.view(-1),
        k1_e.view(-1),
        rho2.view(-1),
        mom2.view(-1),
        e2.view(-1),
        num_cells=num_cells,
        grid_x=int(grid_x),
        grid_y=int(grid_y),
        grid_z=int(grid_z),
        dx=float(dx),
        dt=float(dt),
        gamma=float(gamma),
        c_v=float(c_v),
        rho_min=float(rho_min),
        k_thermal=float(k_thermal),
    )


def pic_gather_update_particles(
    pos: torch.Tensor,
    mass: torch.Tensor,
    pos_out: torch.Tensor,
    vel_out: torch.Tensor,
    heat_out: torch.Tensor,
    rho_field: torch.Tensor,
    mom_field: torch.Tensor,
    e_int_field: torch.Tensor,
    phi_field: torch.Tensor,
    *,
    dx: float,
    dt: float,
    gamma: float,
    c_v: float,
    rho_min: float,
    gravity_enabled: bool,
) -> None:
    _require_triton()
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must be [N,3], got {tuple(pos.shape)}")
    gx, gy, gz = rho_field.shape
    domain_x = float(gx) * float(dx)
    domain_y = float(gy) * float(dx)
    domain_z = float(gz) * float(dx)
    N = int(pos.shape[0])
    grid = lambda m: (triton.cdiv(N, m["BLOCK"]),)
    pic_gather_update_particles_kernel[grid](
        pos,
        mass,
        pos_out,
        vel_out,
        heat_out,
        rho_field,
        mom_field.view(-1),
        e_int_field,
        phi_field,
        N=N,
        grid_x=int(gx),
        grid_y=int(gy),
        grid_z=int(gz),
        dx=float(dx),
        dt=float(dt),
        domain_x=float(domain_x),
        domain_y=float(domain_y),
        domain_z=float(domain_z),
        gamma=float(gamma),
        c_v=float(c_v),
        rho_min=float(rho_min),
        gravity_enabled=bool(gravity_enabled),
    )
