"""Conservative particle-in-cell (PIC) transfer utilities (periodic, CIC/trilinear).

We represent particles as Lagrangian parcels and the gas as an Eulerian grid.
This module implements conservative transfers:
- Particle → grid: mass density, momentum density, total energy density.
- Grid → particle: trilinear sampling of primitive fields (u, T, etc).

All operations are pure torch (works on CPU/CUDA/MPS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class CICStencils:
    """8-corner stencil for CIC / trilinear interpolation."""

    idx: torch.Tensor   # (N, 8) int64 flattened indices
    w: torch.Tensor     # (N, 8) float32 weights (sum=1)


def cic_stencil_periodic(
    positions: torch.Tensor,
    *,
    grid_dims: tuple[int, int, int],
    dx: float,
) -> CICStencils:
    """Compute periodic CIC stencil (8 indices + weights) for each particle.

    positions: (N,3) in simulation length units.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N,3), got {tuple(positions.shape)}")

    gx, gy, gz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    if gx <= 0 or gy <= 0 or gz <= 0:
        raise ValueError(f"grid_dims must be positive, got {grid_dims}")
    if not (dx > 0.0):
        raise ValueError(f"dx must be > 0, got {dx}")

    dev = positions.device
    pos = positions.to(torch.float32)
    domain = torch.tensor([gx * dx, gy * dx, gz * dx], device=dev, dtype=torch.float32)
    pos = torch.remainder(pos, domain)  # periodic wrap into [0, L)

    cell = pos / float(dx)
    base = torch.floor(cell).to(torch.int64)  # (N,3)
    frac = (cell - base.to(torch.float32)).clamp_(0.0, 1.0)  # (N,3)

    ix0 = torch.remainder(base[:, 0], gx)
    iy0 = torch.remainder(base[:, 1], gy)
    iz0 = torch.remainder(base[:, 2], gz)
    ix1 = torch.remainder(ix0 + 1, gx)
    iy1 = torch.remainder(iy0 + 1, gy)
    iz1 = torch.remainder(iz0 + 1, gz)

    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    wx1 = fx
    wy1 = fy
    wz1 = fz

    # 8 corners in (x,y,z) bit order: 000,100,010,110,001,101,011,111
    xs = torch.stack([ix0, ix1, ix0, ix1, ix0, ix1, ix0, ix1], dim=1)
    ys = torch.stack([iy0, iy0, iy1, iy1, iy0, iy0, iy1, iy1], dim=1)
    zs = torch.stack([iz0, iz0, iz0, iz0, iz1, iz1, iz1, iz1], dim=1)

    wx = torch.stack([wx0, wx1, wx0, wx1, wx0, wx1, wx0, wx1], dim=1)
    wy = torch.stack([wy0, wy0, wy1, wy1, wy0, wy0, wy1, wy1], dim=1)
    wz = torch.stack([wz0, wz0, wz0, wz0, wz1, wz1, wz1, wz1], dim=1)
    w = (wx * wy * wz).to(torch.float32)

    # Flatten index
    idx = (xs * (gy * gz) + ys * gz + zs).to(torch.int64)

    return CICStencils(idx=idx, w=w)


def scatter_conserved_cic(
    st: CICStencils,
    *,
    grid_dims: tuple[int, int, int],
    dx: float,
    masses: torch.Tensor,          # (N,)
    velocities: torch.Tensor,      # (N,3)
    internal_energy: torch.Tensor, # (N,) total internal energy per particle (J)
    out_rho: torch.Tensor,         # (gx,gy,gz) density
    out_mom: torch.Tensor,         # (gx,gy,gz,3) momentum density
    out_E: torch.Tensor,           # (gx,gy,gz) total energy density
) -> None:
    """Scatter particle conserved quantities to grid (per-volume densities)."""
    gx, gy, gz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    vol = float(dx) ** 3
    inv_vol = 1.0 / vol

    if masses.ndim != 1:
        raise ValueError(f"masses must have shape (N,), got {tuple(masses.shape)}")
    if velocities.shape != (masses.shape[0], 3):
        raise ValueError(f"velocities must have shape (N,3), got {tuple(velocities.shape)}")
    if internal_energy.shape != masses.shape:
        raise ValueError(f"internal_energy must have shape (N,), got {tuple(internal_energy.shape)}")

    N = int(masses.shape[0])
    if st.idx.shape != (N, 8) or st.w.shape != (N, 8):
        raise ValueError(f"stencil shape mismatch: idx={tuple(st.idx.shape)} w={tuple(st.w.shape)} N={N}")

    # Flatten output
    out_rho_f = out_rho.view(-1)
    out_E_f = out_E.view(-1)
    out_mom_f = out_mom.view(-1, 3)

    # Expand weights for vector scatter.
    w = st.w * inv_vol  # per-volume
    idx = st.idx

    m = masses.to(torch.float32)           # (N,)
    v = velocities.to(torch.float32)       # (N,3)
    ie = internal_energy.to(torch.float32) # (N,)
    ke = (0.5 * m * (v * v).sum(dim=1)).to(torch.float32)  # (N,)

    # NOTE: Be explicit about broadcast ranks:
    # - scalar stencils: (N,1) * (N,8) → (N,8)
    # - vector stencils: (N,1,1) * (N,1,3) * (N,8,1) → (N,8,3)
    rho_contrib = (m[:, None] * w).reshape(-1)  # (N*8,)
    mom_contrib = (m[:, None, None] * v[:, None, :] * w[..., None]).reshape(-1, 3)  # (N*8,3)
    E_contrib = (((ie + ke)[:, None]) * w).reshape(-1)  # (N*8,)

    idx_flat = idx.reshape(-1)
    out_rho_f.scatter_add_(0, idx_flat, rho_contrib)
    out_E_f.scatter_add_(0, idx_flat, E_contrib)
    out_mom_f.scatter_add_(0, idx_flat[:, None].expand(-1, 3), mom_contrib)


def gather_trilinear(
    st: CICStencils,
    field: torch.Tensor,
) -> torch.Tensor:
    """Gather a scalar field at particle locations using CIC weights.

    field: (gx,gy,gz)
    returns: (N,)
    """
    f = field.view(-1)
    vals = f[st.idx]  # (N,8)
    return (vals.to(torch.float32) * st.w).sum(dim=1)


def gather_trilinear_vec3(
    st: CICStencils,
    field: torch.Tensor,
) -> torch.Tensor:
    """Gather a vector field (gx,gy,gz,3) at particle locations using CIC."""
    f = field.view(-1, 3)
    vals = f[st.idx]  # (N,8,3)
    return (vals.to(torch.float32) * st.w[..., None]).sum(dim=1)

