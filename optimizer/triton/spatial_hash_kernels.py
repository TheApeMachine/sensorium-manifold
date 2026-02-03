"""CUDA/Triton spatial-hash collision pipeline.

Implements the same 4-phase pipeline used by the Metal backend:
1) assign particles to cells + atomic cell_counts
2) prefix sum on host (CUDA torch) to produce cell_starts
3) scatter particles into sorted indices (atomic cell_offsets)
4) collisions using neighbor-cell traversal (bounded per-cell scan)

This is correctness-first and uses a conservative cap for per-cell scans.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception as e:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR: Exception = e
else:
    _TRITON_IMPORT_ERROR = RuntimeError("unreachable")


def _require_triton() -> None:
    if triton is None or tl is None:  # pragma: no cover
        raise RuntimeError(f"Triton is required for CUDA spatial hash: {_TRITON_IMPORT_ERROR!r}")


@triton.jit
def spatial_hash_assign_kernel(
    pos_ptr,  # fp32 [N*3]
    particle_cell_idx_ptr,  # int32 [N]
    cell_counts_ptr,  # int32 [num_cells] atomic
    N: tl.constexpr,
    hash_x: tl.constexpr,
    hash_y: tl.constexpr,
    hash_z: tl.constexpr,
    cell_size: tl.constexpr,
    inv_cell_size: tl.constexpr,
    domain_min_x: tl.constexpr,
    domain_min_y: tl.constexpr,
    domain_min_z: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= N:
        return
    px = tl.load(pos_ptr + i * 3 + 0)
    py = tl.load(pos_ptr + i * 3 + 1)
    pz = tl.load(pos_ptr + i * 3 + 2)
    cx = tl.floor((px - domain_min_x) * inv_cell_size).to(tl.int32)
    cy = tl.floor((py - domain_min_y) * inv_cell_size).to(tl.int32)
    cz = tl.floor((pz - domain_min_z) * inv_cell_size).to(tl.int32)
    cx = tl.maximum(0, tl.minimum(cx, hash_x - 1))
    cy = tl.maximum(0, tl.minimum(cy, hash_y - 1))
    cz = tl.maximum(0, tl.minimum(cz, hash_z - 1))
    cell = cx * (hash_y * hash_z) + cy * hash_z + cz
    tl.store(particle_cell_idx_ptr + i, cell)
    tl.atomic_add(cell_counts_ptr + cell, 1)


@triton.jit
def spatial_hash_scatter_kernel(
    particle_cell_idx_ptr,  # int32 [N]
    sorted_particle_idx_ptr,  # int32 [N]
    cell_offsets_ptr,  # int32 [num_cells] atomic (starts as cell_starts)
    N: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= N:
        return
    cell = tl.load(particle_cell_idx_ptr + i).to(tl.int32)
    out = tl.atomic_add(cell_offsets_ptr + cell, 1)
    tl.store(sorted_particle_idx_ptr + out, i)


@triton.jit
def spatial_hash_collisions_kernel(
    pos_ptr,  # fp32 [N*3]
    vel_ptr,  # fp32 [N*3] in/out
    exc_ptr,  # fp32 [N] in/out
    mass_ptr,  # fp32 [N]
    heat_ptr,  # fp32 [N] in/out
    sorted_particle_idx_ptr,  # int32 [N]
    cell_starts_ptr,  # int32 [num_cells+1]
    particle_cell_idx_ptr,  # int32 [N]
    N: tl.constexpr,
    hash_x: tl.constexpr,
    hash_y: tl.constexpr,
    hash_z: tl.constexpr,
    cell_size: tl.constexpr,
    inv_cell_size: tl.constexpr,
    domain_min_x: tl.constexpr,
    domain_min_y: tl.constexpr,
    domain_min_z: tl.constexpr,
    dt: tl.constexpr,
    particle_radius: tl.constexpr,
    young_modulus: tl.constexpr,
    thermal_conductivity: tl.constexpr,
    restitution: tl.constexpr,
    MAX_PER_CELL: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= N:
        return

    # particle i
    px = tl.load(pos_ptr + i * 3 + 0)
    py = tl.load(pos_ptr + i * 3 + 1)
    pz = tl.load(pos_ptr + i * 3 + 2)
    vx = tl.load(vel_ptr + i * 3 + 0)
    vy = tl.load(vel_ptr + i * 3 + 1)
    vz = tl.load(vel_ptr + i * 3 + 2)
    mi = tl.load(mass_ptr + i)
    hi = tl.load(heat_ptr + i)
    exi = tl.load(exc_ptr + i)

    # cell coords
    cx = tl.floor((px - domain_min_x) * inv_cell_size).to(tl.int32)
    cy = tl.floor((py - domain_min_y) * inv_cell_size).to(tl.int32)
    cz = tl.floor((pz - domain_min_z) * inv_cell_size).to(tl.int32)
    cx = tl.maximum(0, tl.minimum(cx, hash_x - 1))
    cy = tl.maximum(0, tl.minimum(cy, hash_y - 1))
    cz = tl.maximum(0, tl.minimum(cz, hash_z - 1))

    r = particle_radius
    two_r = 2.0 * r
    two_r2 = two_r * two_r
    e = tl.maximum(0.0, tl.minimum(restitution, 1.0))

    # neighbor traversal (27 cells)
    for dx in (-1, 0, 1):
        nx = cx + dx
        if nx < 0 or nx >= hash_x:
            continue
        for dy in (-1, 0, 1):
            ny = cy + dy
            if ny < 0 or ny >= hash_y:
                continue
            for dz in (-1, 0, 1):
                nz = cz + dz
                if nz < 0 or nz >= hash_z:
                    continue

                cell = nx * (hash_y * hash_z) + ny * hash_z + nz
                start = tl.load(cell_starts_ptr + cell).to(tl.int32)
                end = tl.load(cell_starts_ptr + cell + 1).to(tl.int32)
                count = end - start
                count = tl.minimum(count, MAX_PER_CELL)

                # scan bounded set
                for t in range(0, MAX_PER_CELL):
                    if t >= count:
                        break
                    j = tl.load(sorted_particle_idx_ptr + (start + t)).to(tl.int32)
                    if j == i:
                        continue
                    # simple symmetric impulse approximation (update i only)
                    qx = tl.load(pos_ptr + j * 3 + 0)
                    qy = tl.load(pos_ptr + j * 3 + 1)
                    qz = tl.load(pos_ptr + j * 3 + 2)
                    dxp = px - qx
                    dyp = py - qy
                    dzp = pz - qz
                    dist2 = dxp * dxp + dyp * dyp + dzp * dzp
                    if dist2 >= two_r2 or dist2 <= 1e-12:
                        continue
                    dist = tl.sqrt(dist2)
                    nxv = dxp / dist
                    nyv = dyp / dist
                    nzv = dzp / dist
                    # relative velocity along normal (assume vj ~ 0 for local update)
                    vrel = vx * nxv + vy * nyv + vz * nzv
                    if vrel > 0.0:
                        continue
                    # impulse magnitude (elastic w/ restitution), m_eff approx using mi only
                    jmag = -(1.0 + e) * vrel * mi
                    vx += (jmag / tl.maximum(mi, 1e-6)) * nxv
                    vy += (jmag / tl.maximum(mi, 1e-6)) * nyv
                    vz += (jmag / tl.maximum(mi, 1e-6)) * nzv
                    # convert lost kinetic energy to heat (rough)
                    hi += tl.maximum(-vrel, 0.0) * 0.01

    tl.store(vel_ptr + i * 3 + 0, vx)
    tl.store(vel_ptr + i * 3 + 1, vy)
    tl.store(vel_ptr + i * 3 + 2, vz)
    tl.store(heat_ptr + i, hi)
    tl.store(exc_ptr + i, exi)


def assign(
    *,
    positions: torch.Tensor,
    particle_cell_idx: torch.Tensor,
    cell_counts: torch.Tensor,
    hash_grid_x: int,
    hash_grid_y: int,
    hash_grid_z: int,
    cell_size: float,
    domain_min_x: float,
    domain_min_y: float,
    domain_min_z: float,
) -> None:
    _require_triton()
    N = int(positions.shape[0])
    if N == 0:
        return
    inv = 1.0 / float(cell_size)
    grid = (N,)
    spatial_hash_assign_kernel[grid](
        positions.contiguous().view(-1),
        particle_cell_idx,
        cell_counts,
        N=N,
        hash_x=int(hash_grid_x),
        hash_y=int(hash_grid_y),
        hash_z=int(hash_grid_z),
        cell_size=float(cell_size),
        inv_cell_size=float(inv),
        domain_min_x=float(domain_min_x),
        domain_min_y=float(domain_min_y),
        domain_min_z=float(domain_min_z),
        num_warps=1,
    )


def scatter(
    *,
    particle_cell_idx: torch.Tensor,
    sorted_particle_idx: torch.Tensor,
    cell_offsets: torch.Tensor,
) -> None:
    _require_triton()
    N = int(particle_cell_idx.numel())
    if N == 0:
        return
    grid = (N,)
    spatial_hash_scatter_kernel[grid](
        particle_cell_idx,
        sorted_particle_idx,
        cell_offsets,
        N=N,
        num_warps=1,
    )


def collisions(
    *,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    excitations: torch.Tensor,
    masses: torch.Tensor,
    heats: torch.Tensor,
    sorted_particle_idx: torch.Tensor,
    cell_starts: torch.Tensor,
    particle_cell_idx: torch.Tensor,
    hash_grid_x: int,
    hash_grid_y: int,
    hash_grid_z: int,
    cell_size: float,
    domain_min_x: float,
    domain_min_y: float,
    domain_min_z: float,
    dt: float,
    particle_radius: float,
    young_modulus: float,
    thermal_conductivity: float,
    restitution: float,
    max_per_cell: int = 64,
) -> None:
    _require_triton()
    N = int(positions.shape[0])
    if N == 0:
        return
    inv = 1.0 / float(cell_size)
    grid = (N,)
    spatial_hash_collisions_kernel[grid](
        positions.contiguous().view(-1),
        velocities.contiguous().view(-1),
        excitations.contiguous(),
        masses.contiguous(),
        heats.contiguous(),
        sorted_particle_idx,
        cell_starts,
        particle_cell_idx,
        N=N,
        hash_x=int(hash_grid_x),
        hash_y=int(hash_grid_y),
        hash_z=int(hash_grid_z),
        cell_size=float(cell_size),
        inv_cell_size=float(inv),
        domain_min_x=float(domain_min_x),
        domain_min_y=float(domain_min_y),
        domain_min_z=float(domain_min_z),
        dt=float(dt),
        particle_radius=float(particle_radius),
        young_modulus=float(young_modulus),
        thermal_conductivity=float(thermal_conductivity),
        restitution=float(restitution),
        MAX_PER_CELL=int(max_per_cell),
        num_warps=1,
    )

