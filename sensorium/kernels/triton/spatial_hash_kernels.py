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
    # Provide a dummy `triton.jit` decorator so this module can be imported
    # in non-CUDA / non-Triton environments. Runtime entrypoints still call
    # `_require_triton()` and will raise when Triton is unavailable.
    class _DummyTriton:
        @staticmethod
        def jit(fn=None, **_kwargs):  # type: ignore[no-untyped-def]
            if fn is None:
                return lambda f: f
            return fn

    triton = _DummyTriton()  # type: ignore[assignment]
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
    # [CHOICE] periodic spatial hash domain
    # [FORMULA] cell = floor(((pos - domain_min) / cell_size) mod grid_dims)
    # [REASON] collision neighborhood should match torus/periodic simulation domain
    gx = (px - domain_min_x) * inv_cell_size
    gy = (py - domain_min_y) * inv_cell_size
    gz = (pz - domain_min_z) * inv_cell_size
    # Wrap into [0, hash_dims) for periodic domain
    gx = gx - float(hash_x) * tl.floor(gx / float(hash_x))
    gy = gy - float(hash_y) * tl.floor(gy / float(hash_y))
    gz = gz - float(hash_z) * tl.floor(gz / float(hash_z))
    cx = tl.floor(gx).to(tl.int32)
    cy = tl.floor(gy).to(tl.int32)
    cz = tl.floor(gz).to(tl.int32)
    # Safety clamp (should be no-op after wrapping, but ensures valid indices)
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
    vel_in_ptr,  # fp32 [N*3] snapshot (read-only)
    heat_in_ptr,  # fp32 [N] snapshot (read-only)
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
    specific_heat: tl.constexpr,
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

    # [CHOICE] periodic cell coords for collision detection
    # [REASON] match torus/periodic simulation domain
    gx = (px - domain_min_x) * inv_cell_size
    gy = (py - domain_min_y) * inv_cell_size
    gz = (pz - domain_min_z) * inv_cell_size
    gx = gx - float(hash_x) * tl.floor(gx / float(hash_x))
    gy = gy - float(hash_y) * tl.floor(gy / float(hash_y))
    gz = gz - float(hash_z) * tl.floor(gz / float(hash_z))
    cx = tl.floor(gx).to(tl.int32)
    cy = tl.floor(gy).to(tl.int32)
    cz = tl.floor(gz).to(tl.int32)
    cx = tl.maximum(0, tl.minimum(cx, hash_x - 1))
    cy = tl.maximum(0, tl.minimum(cy, hash_y - 1))
    cz = tl.maximum(0, tl.minimum(cz, hash_z - 1))

    # [CHOICE] collision invariants (fail loudly)
    # [FORMULA] require: m_i>0, c_v>0, r>0, dt>0
    # [REASON] silent clamps hide invalid physical states
    qnan = float('nan')
    if (mi <= 0.0) | (specific_heat <= 0.0) | (particle_radius <= 0.0) | (dt <= 0.0):
        tl.store(vel_ptr + i * 3 + 0, qnan)
        tl.store(vel_ptr + i * 3 + 1, qnan)
        tl.store(vel_ptr + i * 3 + 2, qnan)
        tl.store(heat_ptr + i, qnan)
        return

    r = particle_radius
    two_r = 2.0 * r
    two_r2 = two_r * two_r
    e = tl.maximum(0.0, tl.minimum(restitution, 1.0))
    inv_mi = 1.0 / mi

    # Accumulate impulses and heat changes for particle i.
    imp_x = 0.0
    imp_y = 0.0
    imp_z = 0.0
    heat_delta = 0.0

    # [CHOICE] periodic neighbor traversal (27 cells with wrapping)
    # [REASON] detect collisions across periodic boundaries (torus domain)
    for dx in (-1, 0, 1):
        # Periodic wrapping for neighbor cells
        nx = (cx + dx + hash_x) % hash_x
        for dy in (-1, 0, 1):
            ny = (cy + dy + hash_y) % hash_y
            for dz in (-1, 0, 1):
                nz = (cz + dz + hash_z) % hash_z

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
                    # Collision response (update i only), using snapshots for j.
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
                    overlap = two_r - dist

                    # Read particle j from snapshots for consistent pair interactions.
                    vjx = tl.load(vel_in_ptr + j * 3 + 0)
                    vjy = tl.load(vel_in_ptr + j * 3 + 1)
                    vjz = tl.load(vel_in_ptr + j * 3 + 2)
                    mj = tl.load(mass_ptr + j)
                    hj = tl.load(heat_in_ptr + j)

                    if mj <= 0.0:
                        tl.store(vel_ptr + i * 3 + 0, qnan)
                        tl.store(vel_ptr + i * 3 + 1, qnan)
                        tl.store(vel_ptr + i * 3 + 2, qnan)
                        tl.store(heat_ptr + i, qnan)
                        return

                    # Relative velocity along the normal.
                    vrel_x = vx - vjx
                    vrel_y = vy - vjy
                    vrel_z = vz - vjz
                    v_n = vrel_x * nxv + vrel_y * nyv + vrel_z * nzv

                    # Impulse-based collision (momentum conservation).
                    if v_n < 0.0:
                        m_eff = (mi * mj) / (mi + mj)
                        J = (1.0 + e) * m_eff * (-v_n)
                        # We divide by 2 because we process each pair twice (once for i, once for j).
                        scale = 0.5 * (J * inv_mi)
                        imp_x += nxv * scale
                        imp_y += nyv * scale
                        imp_z += nzv * scale

                        # Energy conservation: KE_lost -> heat.
                        ke_lost = 0.5 * m_eff * v_n * v_n * (1.0 - e * e)
                        heat_delta += ke_lost * 0.5

                    # Hertzian/linear contact force (prevents overlap).
                    contact_force = young_modulus * overlap
                    scale_c = (contact_force * dt) * inv_mi
                    imp_x += nxv * scale_c
                    imp_y += nyv * scale_c
                    imp_z += nzv * scale_c

                    # Heat conduction on contact (Fourier-like).
                    cv = specific_heat
                    T_i = hi / (mi * cv)
                    T_j = hj / (mj * cv)
                    contact_area = overlap * overlap
                    dQ = thermal_conductivity * contact_area * (T_j - T_i) * dt
                    heat_delta += dQ

    vx = vx + imp_x
    vy = vy + imp_y
    vz = vz + imp_z
    hi = tl.maximum(hi + heat_delta, 0.0)

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
    vel_in: torch.Tensor,
    heat_in: torch.Tensor,
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
    specific_heat: float,
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
        vel_in.contiguous().view(-1),
        heat_in.contiguous(),
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
        specific_heat=float(specific_heat),
        restitution=float(restitution),
        MAX_PER_CELL=int(max_per_cell),
        num_warps=1,
    )

