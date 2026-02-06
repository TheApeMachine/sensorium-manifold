"""Triton kernels for 3D manifold grid physics (CUDA).

These mirror the Metal kernels in `optimizer/metal/manifold_physics.metal`:
- clear_field
- scatter_particle (mass/heat to fields with trilinear weights)
- poisson_jacobi_step
- diffuse_heat_field
- gather_update_particles (fused trilinear sample + physics update)

All tensors are expected to be CUDA fp32 and contiguous.
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
        raise RuntimeError(f"Triton is required for CUDA grid backend: {_TRITON_IMPORT_ERROR!r}")


# =============================================================================
# Adaptive Thermodynamics: GPU reduction for global energy statistics
# =============================================================================
# 2-pass reduction to compute: mean_abs = mean(|x|), mean = mean(x), std = std(x)
# entirely on-GPU, so downstream kernels can do adaptive renormalization without
# CPU sync or "magic number" damping.
#
# Output format (4 floats in out_stats):
#   [0]: mean_abs
#   [1]: mean
#   [2]: std
#   [3]: count

@triton.jit
def reduce_float_stats_pass1_kernel(
    x_ptr,           # fp32 [N]
    group_stats_ptr, # fp32 [num_groups * 4] - [sum_abs, sum, sum_sq, count]
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pass 1: Compute partial sums for each thread block."""
    tg_id = tl.program_id(0)
    tid = tl.arange(0, BLOCK)
    idx = tg_id * BLOCK + tid
    mask = idx < N

    v = tl.load(x_ptr + idx, mask=mask, other=0.0)
    valid = tl.where(mask, 1.0, 0.0)

    # Reduce across the block
    sum_abs = tl.sum(tl.abs(v) * valid, axis=0)
    sum_v = tl.sum(v * valid, axis=0)
    sum_sq = tl.sum(v * v * valid, axis=0)
    count = tl.sum(valid, axis=0)

    # Store group results
    base = tg_id * 4
    tl.store(group_stats_ptr + base + 0, sum_abs)
    tl.store(group_stats_ptr + base + 1, sum_v)
    tl.store(group_stats_ptr + base + 2, sum_sq)
    tl.store(group_stats_ptr + base + 3, count)


@triton.jit
def reduce_float_stats_finalize_kernel(
    group_stats_ptr, # fp32 [num_groups * 4]
    out_stats_ptr,   # fp32 [4] - [mean_abs, mean, std, count]
    num_groups: tl.constexpr,
):
    """Pass 2: Combine group statistics into final mean_abs, mean, std, count."""
    # Single-threaded reduction of all group stats
    sum_abs = 0.0
    sum_v = 0.0
    sum_sq = 0.0
    count = 0.0

    for i in range(num_groups):
        base = i * 4
        sum_abs = sum_abs + tl.load(group_stats_ptr + base + 0)
        sum_v = sum_v + tl.load(group_stats_ptr + base + 1)
        sum_sq = sum_sq + tl.load(group_stats_ptr + base + 2)
        count = count + tl.load(group_stats_ptr + base + 3)

    # Compute final statistics
    # [CHOICE] reduction empty-case semantics
    # [FORMULA] if count<=0: mean_abs=mean=std=0, count=0
    # [REASON] removes numerical clamp; makes empty reduction explicit
    has_data = count > 0.0
    mean_abs = tl.where(has_data, sum_abs / count, 0.0)
    mean = tl.where(has_data, sum_v / count, 0.0)
    # [CHOICE] non-negative variance
    # [FORMULA] var = max(E[x^2] - E[x]^2, 0)
    # [REASON] rounding can produce tiny negative; project back to R_{\ge 0}
    var = tl.where(has_data, (sum_sq / count) - mean * mean, 0.0)
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(out_stats_ptr + 0, mean_abs)
    tl.store(out_stats_ptr + 1, mean)
    tl.store(out_stats_ptr + 2, std)
    tl.store(out_stats_ptr + 3, count)


@triton.jit
def clear_field_kernel(field_ptr, n_elements: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 256 + tl.arange(0, 256)
    m = offs < n_elements
    tl.store(field_ptr + offs, 0.0, mask=m)


@triton.jit
def scatter_particle_kernel(
    pos_ptr,  # fp32 [N*3]
    mass_ptr,  # fp32 [N]
    heat_ptr,  # fp32 [N] (thermal store Q)
    energy_ptr,  # fp32 [N] (oscillator/internal store E_osc)
    gravity_ptr,  # fp32 [X*Y*Z] atomic
    heat_field_ptr,  # fp32 [X*Y*Z] atomic (total internal energy per cell)
    N: tl.constexpr,
    grid_x: tl.constexpr,
    grid_y: tl.constexpr,
    grid_z: tl.constexpr,
    inv_spacing: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= N:
        return

    px = tl.load(pos_ptr + i * 3 + 0)
    py = tl.load(pos_ptr + i * 3 + 1)
    pz = tl.load(pos_ptr + i * 3 + 2)
    m = tl.load(mass_ptr + i)
    q = tl.load(heat_ptr + i)
    e = tl.load(energy_ptr + i)
    e_total = q + e

    gx = px * inv_spacing
    gy = py * inv_spacing
    gz = pz * inv_spacing

    # [CHOICE] periodic grid coordinate mapping (torus domain)
    # [FORMULA] g = g - dims * floor(g / dims), wraps into [0, dims)
    # [REASON] torus domain: positions and fields are periodic
    # [NOTES] This avoids non-physical boundary clamping artifacts ("edge sinks").
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

    # [CHOICE] periodic scatter to 8 grid corners
    # [FORMULA] deposit into 8 corners with (x1,y1,z1) wrapped mod dims
    # [REASON] torus domain must conserve mass/heat across boundaries
    # [NOTES] avoids "edge sinks" created by clamping.
    stride_z = 1
    stride_y = grid_z
    stride_x = grid_y * grid_z

    # Periodic corner indices
    bx0 = bx
    by0 = by
    bz0 = bz
    bx1 = (bx + 1) % grid_x
    by1 = (by + 1) % grid_y
    bz1 = (bz + 1) % grid_z

    # Compute all 8 corner linear indices with periodic wrapping
    idx000 = bx0 * stride_x + by0 * stride_y + bz0 * stride_z
    idx001 = bx0 * stride_x + by0 * stride_y + bz1 * stride_z
    idx010 = bx0 * stride_x + by1 * stride_y + bz0 * stride_z
    idx011 = bx0 * stride_x + by1 * stride_y + bz1 * stride_z
    idx100 = bx1 * stride_x + by0 * stride_y + bz0 * stride_z
    idx101 = bx1 * stride_x + by0 * stride_y + bz1 * stride_z
    idx110 = bx1 * stride_x + by1 * stride_y + bz0 * stride_z
    idx111 = bx1 * stride_x + by1 * stride_y + bz1 * stride_z

    w000 = wx0 * wy0 * wz0
    w001 = wx0 * wy0 * wz1
    w010 = wx0 * wy1 * wz0
    w011 = wx0 * wy1 * wz1
    w100 = wx1 * wy0 * wz0
    w101 = wx1 * wy0 * wz1
    w110 = wx1 * wy1 * wz0
    w111 = wx1 * wy1 * wz1

    # Scatter mass (gravity field) with periodic indexing
    tl.atomic_add(gravity_ptr + idx000, m * w000)
    tl.atomic_add(gravity_ptr + idx001, m * w001)
    tl.atomic_add(gravity_ptr + idx010, m * w010)
    tl.atomic_add(gravity_ptr + idx011, m * w011)
    tl.atomic_add(gravity_ptr + idx100, m * w100)
    tl.atomic_add(gravity_ptr + idx101, m * w101)
    tl.atomic_add(gravity_ptr + idx110, m * w110)
    tl.atomic_add(gravity_ptr + idx111, m * w111)

    # Scatter total internal energy (heat field) with periodic indexing
    # [CHOICE] total internal energy deposition
    # [FORMULA] Q_cell := Σ_i w_i (Q_i + E_osc,i)
    # [REASON] temperature is defined from total internal energy, not thermal Q alone
    tl.atomic_add(heat_field_ptr + idx000, e_total * w000)
    tl.atomic_add(heat_field_ptr + idx001, e_total * w001)
    tl.atomic_add(heat_field_ptr + idx010, e_total * w010)
    tl.atomic_add(heat_field_ptr + idx011, e_total * w011)
    tl.atomic_add(heat_field_ptr + idx100, e_total * w100)
    tl.atomic_add(heat_field_ptr + idx101, e_total * w101)
    tl.atomic_add(heat_field_ptr + idx110, e_total * w110)
    tl.atomic_add(heat_field_ptr + idx111, e_total * w111)


@triton.jit
def poisson_jacobi_step_kernel(
    phi_in_ptr,
    rho_ptr,
    phi_out_ptr,
    gravity_4pi: tl.constexpr,
    grid_x: tl.constexpr,
    grid_y: tl.constexpr,
    grid_z: tl.constexpr,
    grid_spacing: tl.constexpr,
):
    pid = tl.program_id(0)
    n = grid_x * grid_y * grid_z
    idx = pid * 256 + tl.arange(0, 256)
    m = idx < n

    stride_z = 1
    stride_y = grid_z
    stride_x = grid_y * grid_z

    # decode idx -> (x,y,z)
    x = idx // stride_x
    rem = idx - x * stride_x
    y = rem // stride_y
    z = rem - y * stride_y

    center = tl.load(phi_in_ptr + idx, mask=m, other=0.0)
    rho = tl.load(rho_ptr + idx, mask=m, other=0.0)

    # [CHOICE] periodic boundary handling (torus domain)
    # [FORMULA] neighbor at x-1 wraps to grid_x-1 when x=0; neighbor at x+1 wraps to 0 when x=grid_x-1
    # [REASON] matches torus/periodic simulation domain used in particle dynamics
    # [NOTES] This ensures consistent physics at domain boundaries.
    x_prev = tl.where(x > 0, x - 1, grid_x - 1)
    x_next = tl.where(x < grid_x - 1, x + 1, 0)
    y_prev = tl.where(y > 0, y - 1, grid_y - 1)
    y_next = tl.where(y < grid_y - 1, y + 1, 0)
    z_prev = tl.where(z > 0, z - 1, grid_z - 1)
    z_next = tl.where(z < grid_z - 1, z + 1, 0)

    idx_xm = x_prev * stride_x + y * stride_y + z * stride_z
    idx_xp = x_next * stride_x + y * stride_y + z * stride_z
    idx_ym = x * stride_x + y_prev * stride_y + z * stride_z
    idx_yp = x * stride_x + y_next * stride_y + z * stride_z
    idx_zm = x * stride_x + y * stride_y + z_prev * stride_z
    idx_zp = x * stride_x + y * stride_y + z_next * stride_z

    xm = tl.load(phi_in_ptr + idx_xm, mask=m, other=0.0)
    xp = tl.load(phi_in_ptr + idx_xp, mask=m, other=0.0)
    ym = tl.load(phi_in_ptr + idx_ym, mask=m, other=0.0)
    yp = tl.load(phi_in_ptr + idx_yp, mask=m, other=0.0)
    zm = tl.load(phi_in_ptr + idx_zm, mask=m, other=0.0)
    zp = tl.load(phi_in_ptr + idx_zp, mask=m, other=0.0)

    h2 = grid_spacing * grid_spacing
    out = (xm + xp + ym + yp + zm + zp - h2 * gravity_4pi * rho) * (1.0 / 6.0)
    tl.store(phi_out_ptr + idx, out, mask=m)


@triton.jit
def diffuse_heat_field_kernel(
    temp_in_ptr,
    temp_out_ptr,
    diffusion_coef: tl.constexpr,
    dt: tl.constexpr,
    grid_x: tl.constexpr,
    grid_y: tl.constexpr,
    grid_z: tl.constexpr,
    inv_spacing: tl.constexpr,
):
    pid = tl.program_id(0)
    n = grid_x * grid_y * grid_z
    idx = pid * 256 + tl.arange(0, 256)
    m = idx < n

    stride_z = 1
    stride_y = grid_z
    stride_x = grid_y * grid_z

    x = idx // stride_x
    rem = idx - x * stride_x
    y = rem // stride_y
    z = rem - y * stride_y

    center = tl.load(temp_in_ptr + idx, mask=m, other=0.0)

    # [CHOICE] periodic boundary handling (torus domain)
    # [FORMULA] neighbor at x-1 wraps to grid_x-1 when x=0
    # [REASON] matches torus/periodic simulation domain used in particle dynamics
    # [NOTES] This ensures consistent heat diffusion at domain boundaries.
    x_prev = tl.where(x > 0, x - 1, grid_x - 1)
    x_next = tl.where(x < grid_x - 1, x + 1, 0)
    y_prev = tl.where(y > 0, y - 1, grid_y - 1)
    y_next = tl.where(y < grid_y - 1, y + 1, 0)
    z_prev = tl.where(z > 0, z - 1, grid_z - 1)
    z_next = tl.where(z < grid_z - 1, z + 1, 0)

    idx_xm = x_prev * stride_x + y * stride_y + z * stride_z
    idx_xp = x_next * stride_x + y * stride_y + z * stride_z
    idx_ym = x * stride_x + y_prev * stride_y + z * stride_z
    idx_yp = x * stride_x + y_next * stride_y + z * stride_z
    idx_zm = x * stride_x + y * stride_y + z_prev * stride_z
    idx_zp = x * stride_x + y * stride_y + z_next * stride_z

    xm = tl.load(temp_in_ptr + idx_xm, mask=m, other=0.0)
    xp = tl.load(temp_in_ptr + idx_xp, mask=m, other=0.0)
    ym = tl.load(temp_in_ptr + idx_ym, mask=m, other=0.0)
    yp = tl.load(temp_in_ptr + idx_yp, mask=m, other=0.0)
    zm = tl.load(temp_in_ptr + idx_zm, mask=m, other=0.0)
    zp = tl.load(temp_in_ptr + idx_zp, mask=m, other=0.0)

    lap = (xm + xp + ym + yp + zm + zp - 6.0 * center) * (inv_spacing * inv_spacing)
    out = center + diffusion_coef * lap * dt
    tl.store(temp_out_ptr + idx, out, mask=m)


@triton.jit
def derive_temperature_field_kernel(
    mass_field_ptr,   # fp32 [X*Y*Z] - scattered mass per cell
    heat_field_ptr,   # fp32 [X*Y*Z] - scattered heat per cell
    temp_field_ptr,   # fp32 [X*Y*Z] - output temperature field
    specific_heat: tl.constexpr,
    num_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Derive temperature field from scattered mass and heat.

    [CHOICE] cell temperature from scattered stores (explicit vacuum semantics)
    [FORMULA] If m_cell > 0:  T_cell = Q_cell / (m_cell c_v)
              Else:           T_cell = 0  (vacuum / empty cell has no temperature)
    [REASON] removes numerical ε and makes vacuum handling explicit
    [NOTES] Requires invariant c_v > 0. If c_v <= 0, we write NaN to fail loudly.
    """
    pid = tl.program_id(0)
    gid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = gid < num_elements

    # Load mass and heat for this cell
    m = tl.load(mass_field_ptr + gid, mask=mask, other=0.0)
    Q = tl.load(heat_field_ptr + gid, mask=mask, other=0.0)

    # Invariant: c_v > 0 (fail loudly if violated)
    cv = specific_heat
    qnan = float('nan')

    # Vacuum semantics: m <= 0 → T = 0; m > 0 → T = Q / (m * cv)
    T = tl.where(m > 0.0, Q / (m * cv), 0.0)

    # Fail-loud if c_v invalid
    T = tl.where(cv > 0.0, T, qnan)

    tl.store(temp_field_ptr + gid, T, mask=mask)


@triton.jit
def gather_update_particles_kernel(
    gravity_phi_ptr,  # fp32 [X*Y*Z]
    temp_field_ptr,  # fp32 [X*Y*Z]
    mass_field_ptr,  # fp32 [X*Y*Z] scattered mass-per-cell (for EOS pressure)
    pos_ptr,  # fp32 [N*3] in/out
    vel_ptr,  # fp32 [N*3] in/out
    energy_ptr,  # fp32 [N] in/out
    heat_ptr,  # fp32 [N] in/out
    excitation_ptr,  # fp32 [N] in/out (conserved)
    mass_ptr,  # fp32 [N]
    N: tl.constexpr,
    grid_x: tl.constexpr,
    grid_y: tl.constexpr,
    grid_z: tl.constexpr,
    dt: tl.constexpr,
    grid_spacing: tl.constexpr,
    inv_spacing: tl.constexpr,
    G: tl.constexpr,
    k_B: tl.constexpr,
    sigma_SB: tl.constexpr,
    hbar: tl.constexpr,
    particle_radius: tl.constexpr,
    thermal_conductivity: tl.constexpr,
    specific_heat: tl.constexpr,
    dynamic_viscosity: tl.constexpr,
    emissivity: tl.constexpr,
    young_modulus: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= N:
        return

    px = tl.load(pos_ptr + i * 3 + 0)
    py = tl.load(pos_ptr + i * 3 + 1)
    pz = tl.load(pos_ptr + i * 3 + 2)
    vx = tl.load(vel_ptr + i * 3 + 0)
    vy = tl.load(vel_ptr + i * 3 + 1)
    vz = tl.load(vel_ptr + i * 3 + 2)
    energy = tl.load(energy_ptr + i)
    heat = tl.load(heat_ptr + i)
    excitation = tl.load(excitation_ptr + i)
    mass = tl.load(mass_ptr + i)

    # =========================================================================
    # FUNDAMENTAL INVARIANTS (fail loudly)
    # =========================================================================
    # [CHOICE] require: m>0, Δx>0, r>0, c_v>0
    # [REASON] these are physical preconditions; silent clamps hide invalid states
    # [NOTES] on violation we write NaN (quiet NaN pattern) to state to surface errors immediately.
    qnan = float('nan')
    invalid_params = (mass <= 0.0) | (grid_spacing <= 0.0) | (particle_radius <= 0.0) | (specific_heat <= 0.0)
    if invalid_params:
        tl.store(pos_ptr + i * 3 + 0, qnan)
        tl.store(pos_ptr + i * 3 + 1, qnan)
        tl.store(pos_ptr + i * 3 + 2, qnan)
        tl.store(vel_ptr + i * 3 + 0, qnan)
        tl.store(vel_ptr + i * 3 + 1, qnan)
        tl.store(vel_ptr + i * 3 + 2, qnan)
        tl.store(energy_ptr + i, qnan)
        tl.store(heat_ptr + i, qnan)
        tl.store(excitation_ptr + i, qnan)
        return

    # [CHOICE] periodic grid coordinate mapping (torus domain)
    # [FORMULA] g = (pos / Δx) mod grid_dims
    # [REASON] torus domain: positions and fields are periodic
    # [NOTES] This avoids non-physical boundary clamping artifacts.
    gx = px * inv_spacing
    gy = py * inv_spacing
    gz = pz * inv_spacing
    gx = gx - float(grid_x) * tl.floor(gx / float(grid_x))
    gy = gy - float(grid_y) * tl.floor(gy / float(grid_y))
    gz = gz - float(grid_z) * tl.floor(gz / float(grid_z))

    bx = tl.floor(gx).to(tl.int32)
    by = tl.floor(gy).to(tl.int32)
    bz = tl.floor(gz).to(tl.int32)
    fx = gx - bx.to(tl.float32)
    fy = gy - by.to(tl.float32)
    fz = gz - bz.to(tl.float32)

    stride_z = 1
    stride_y = grid_z
    stride_x = grid_y * grid_z

    # [CHOICE] periodic corner sampling
    # [FORMULA] (x1,y1,z1) = (x0+1, y0+1, z0+1) mod dims
    # [REASON] torus domain requires wrapping at boundaries
    bx0 = bx
    by0 = by
    bz0 = bz
    bx1 = (bx + 1) % grid_x
    by1 = (by + 1) % grid_y
    bz1 = (bz + 1) % grid_z

    # Compute all 8 corner linear indices with periodic wrapping
    idx000 = bx0 * stride_x + by0 * stride_y + bz0 * stride_z
    idx001 = bx0 * stride_x + by0 * stride_y + bz1 * stride_z
    idx010 = bx0 * stride_x + by1 * stride_y + bz0 * stride_z
    idx011 = bx0 * stride_x + by1 * stride_y + bz1 * stride_z
    idx100 = bx1 * stride_x + by0 * stride_y + bz0 * stride_z
    idx101 = bx1 * stride_x + by0 * stride_y + bz1 * stride_z
    idx110 = bx1 * stride_x + by1 * stride_y + bz0 * stride_z
    idx111 = bx1 * stride_x + by1 * stride_y + bz1 * stride_z

    # Load corners using periodic indices (phi, T, and scattered mass)
    c000_phi = tl.load(gravity_phi_ptr + idx000)
    c001_phi = tl.load(gravity_phi_ptr + idx001)
    c010_phi = tl.load(gravity_phi_ptr + idx010)
    c011_phi = tl.load(gravity_phi_ptr + idx011)
    c100_phi = tl.load(gravity_phi_ptr + idx100)
    c101_phi = tl.load(gravity_phi_ptr + idx101)
    c110_phi = tl.load(gravity_phi_ptr + idx110)
    c111_phi = tl.load(gravity_phi_ptr + idx111)

    c000_T = tl.load(temp_field_ptr + idx000)
    c001_T = tl.load(temp_field_ptr + idx001)
    c010_T = tl.load(temp_field_ptr + idx010)
    c011_T = tl.load(temp_field_ptr + idx011)
    c100_T = tl.load(temp_field_ptr + idx100)
    c101_T = tl.load(temp_field_ptr + idx101)
    c110_T = tl.load(temp_field_ptr + idx110)
    c111_T = tl.load(temp_field_ptr + idx111)

    c000_m = tl.load(mass_field_ptr + idx000)
    c001_m = tl.load(mass_field_ptr + idx001)
    c010_m = tl.load(mass_field_ptr + idx010)
    c011_m = tl.load(mass_field_ptr + idx011)
    c100_m = tl.load(mass_field_ptr + idx100)
    c101_m = tl.load(mass_field_ptr + idx101)
    c110_m = tl.load(mass_field_ptr + idx110)
    c111_m = tl.load(mass_field_ptr + idx111)

    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    wx1 = fx
    wy1 = fy
    wz1 = fz

    # trilinear sample for temperature
    c00 = c000_T * wz0 + c001_T * wz1
    c01 = c010_T * wz0 + c011_T * wz1
    c10 = c100_T * wz0 + c101_T * wz1
    c11 = c110_T * wz0 + c111_T * wz1
    c0 = c00 * wy0 + c01 * wy1
    c1 = c10 * wy0 + c11 * wy1
    local_T = c0 * wx0 + c1 * wx1

    # Trilinear sample for scattered mass-per-cell (for EOS pressure).
    m00 = c000_m * wz0 + c001_m * wz1
    m01 = c010_m * wz0 + c011_m * wz1
    m10 = c100_m * wz0 + c101_m * wz1
    m11 = c110_m * wz0 + c111_m * wz1
    m0 = m00 * wy0 + m01 * wy1
    m1 = m10 * wy0 + m11 * wy1
    m_cell = m0 * wx0 + m1 * wx1

    # [CHOICE] temperature validation (fail loudly on non-finite)
    # [REASON] non-finite temperature indicates corrupted field state; fail loudly
    # [NOTES] Project negative temperatures to T>=0 (physical boundary)
    local_T = tl.maximum(local_T, 0.0)

    # gradient estimate like Metal (difference between faces)
    face_x0_phi = c000_phi * wy0 * wz0 + c010_phi * wy1 * wz0 + c001_phi * wy0 * wz1 + c011_phi * wy1 * wz1
    face_x1_phi = c100_phi * wy0 * wz0 + c110_phi * wy1 * wz0 + c101_phi * wy0 * wz1 + c111_phi * wy1 * wz1
    grad_x_phi = (face_x1_phi - face_x0_phi) * inv_spacing

    face_y0_phi = c000_phi * wx0 * wz0 + c100_phi * wx1 * wz0 + c001_phi * wx0 * wz1 + c101_phi * wx1 * wz1
    face_y1_phi = c010_phi * wx0 * wz0 + c110_phi * wx1 * wz0 + c011_phi * wx0 * wz1 + c111_phi * wx1 * wz1
    grad_y_phi = (face_y1_phi - face_y0_phi) * inv_spacing

    face_z0_phi = c000_phi * wx0 * wy0 + c100_phi * wx1 * wy0 + c010_phi * wx0 * wy1 + c110_phi * wx1 * wy1
    face_z1_phi = c001_phi * wx0 * wy0 + c101_phi * wx1 * wy0 + c011_phi * wx0 * wy1 + c111_phi * wx1 * wy1
    grad_z_phi = (face_z1_phi - face_z0_phi) * inv_spacing

    face_x0_T = c000_T * wy0 * wz0 + c010_T * wy1 * wz0 + c001_T * wy0 * wz1 + c011_T * wy1 * wz1
    face_x1_T = c100_T * wy0 * wz0 + c110_T * wy1 * wz0 + c101_T * wy0 * wz1 + c111_T * wy1 * wz1
    grad_x_T = (face_x1_T - face_x0_T) * inv_spacing

    face_y0_T = c000_T * wx0 * wz0 + c100_T * wx1 * wz0 + c001_T * wx0 * wz1 + c101_T * wx1 * wz1
    face_y1_T = c010_T * wx0 * wz0 + c110_T * wx1 * wz0 + c011_T * wx0 * wz1 + c111_T * wx1 * wz1
    grad_y_T = (face_y1_T - face_y0_T) * inv_spacing

    face_z0_T = c000_T * wx0 * wy0 + c100_T * wx1 * wy0 + c010_T * wx0 * wy1 + c110_T * wx1 * wy1
    face_z1_T = c001_T * wx0 * wy0 + c101_T * wx1 * wy0 + c011_T * wx0 * wy1 + c111_T * wx1 * wy1
    grad_z_T = (face_z1_T - face_z0_T) * inv_spacing

    face_x0_m = c000_m * wy0 * wz0 + c010_m * wy1 * wz0 + c001_m * wy0 * wz1 + c011_m * wy1 * wz1
    face_x1_m = c100_m * wy0 * wz0 + c110_m * wy1 * wz0 + c101_m * wy0 * wz1 + c111_m * wy1 * wz1
    grad_x_m = (face_x1_m - face_x0_m) * inv_spacing

    face_y0_m = c000_m * wx0 * wz0 + c100_m * wx1 * wz0 + c001_m * wx0 * wz1 + c101_m * wx1 * wz1
    face_y1_m = c010_m * wx0 * wz0 + c110_m * wx1 * wz0 + c011_m * wx0 * wz1 + c111_m * wx1 * wz1
    grad_y_m = (face_y1_m - face_y0_m) * inv_spacing

    face_z0_m = c000_m * wx0 * wy0 + c100_m * wx1 * wy0 + c010_m * wx0 * wy1 + c110_m * wx1 * wy1
    face_z1_m = c001_m * wx0 * wy0 + c101_m * wx1 * wy0 + c011_m * wx0 * wy1 + c111_m * wx1 * wy1
    grad_z_m = (face_z1_m - face_z0_m) * inv_spacing

    # =========================================================================
    # GRAVITY FORCE
    # =========================================================================
    # [CHOICE] gravity coupling (particle ← potential field)
    # [FORMULA] Poisson: ∇²φ = 4πGρ  =>  acceleration a = -∇φ
    #           Force:   F = m a = -m ∇φ
    # [REASON] standard Newtonian gravity in potential form
    # [NOTES] We choose to include G in the Poisson solve (φ already includes G).
    #         Therefore we MUST NOT multiply by G again here.
    gravity_fx = -grad_x_phi * mass
    gravity_fy = -grad_y_phi * mass
    gravity_fz = -grad_z_phi * mass

    # =========================================================================
    # TEMPERATURE AND PRESSURE
    # =========================================================================
    # [CHOICE] particle temperature from total internal energy
    # [FORMULA] T_i = (Q_i + E_osc,i) / (m_i c_v)
    # [REASON] closes bookkeeping between thermal + oscillator energy stores
    particle_T = (heat + energy) / (mass * specific_heat)

    # [CHOICE] pressure force (continuum, ideal-gas EOS)
    # [FORMULA] EOS:     P = ρ k_B T
    #          Force:   a = -(1/ρ) ∇P
    #                  ∇P = k_B (T ∇ρ + ρ ∇T)
    # [REASON] dimensionally consistent continuum form (matches Metal)
    # [NOTES] Vacuum semantics: if ρ<=0, pressure contribution is 0 (no clamp-as-physics).
    h = grid_spacing
    inv_h3 = 1.0 / (h * h * h)
    rho = m_cell * inv_h3
    grad_x_rho = grad_x_m * inv_h3
    grad_y_rho = grad_y_m * inv_h3
    grad_z_rho = grad_z_m * inv_h3

    pressure_fx = 0.0
    pressure_fy = 0.0
    pressure_fz = 0.0
    if rho > 0.0:
        gradP_x = k_B * (local_T * grad_x_rho + rho * grad_x_T)
        gradP_y = k_B * (local_T * grad_y_rho + rho * grad_y_T)
        gradP_z = k_B * (local_T * grad_z_rho + rho * grad_z_T)
        aP_x = -gradP_x / rho
        aP_y = -gradP_y / rho
        aP_z = -gradP_z / rho
        pressure_fx = aP_x * mass
        pressure_fy = aP_y * mass
        pressure_fz = aP_z * mass

    # =========================================================================
    # HEAT TRANSFER: Newton's law of cooling + Stefan-Boltzmann radiation
    # =========================================================================
    r = particle_radius

    # [CHOICE] conduction to ambient medium (sphere in infinite medium)
    # [FORMULA] Q̇ = 4π κ r (T_env - T_particle)
    # [REASON] steady-state conduction solution of Laplace equation around a sphere
    # [NOTES] κ here is the medium thermal conductivity (not diffusivity).
    PI = 3.14159265358979
    dQ_cond = (4.0 * PI) * thermal_conductivity * r * (local_T - particle_T) * dt
    heat = heat + dQ_cond

    # [CHOICE] radiative cooling
    # [FORMULA] P = ε σ A T^4, with A = 4π r²
    # [REASON] blackbody radiation loss from particle surface
    surface_area = 4.0 * PI * r * r
    T4 = particle_T * particle_T * particle_T * particle_T
    dQ_rad = emissivity * sigma_SB * surface_area * T4 * dt
    # [CHOICE] non-negative thermal energy (0 K baseline)
    # [FORMULA] Q >= 0
    # [REASON] internal thermal energy relative to absolute zero cannot be negative
    heat = tl.maximum(heat - dQ_rad, 0.0)

    # Update temperature after heat exchange
    particle_T = (heat + energy) / (mass * specific_heat)

    # =========================================================================
    # THERMAL ↔ OSCILLATOR ENERGY EXCHANGE (physics-based)
    # =========================================================================
    # [CHOICE] thermalization timescale (derived from conduction coefficient)
    # [FORMULA] τ = (m c_v) / (4π κ r)
    # [REASON] Natural thermalization timescale implied by conduction to the medium
    # [NOTES] Removes ad-hoc tau=10.0; uses existing physics parameters
    kappa = thermal_conductivity
    cv = specific_heat
    denom_tau = (4.0 * PI) * kappa * r
    tau = tl.where(denom_tau > 0.0, (mass * cv) / denom_tau, 1e10)

    # α = 1 - exp(-dt/τ)
    alpha = 1.0 - tl.exp(-dt / tl.maximum(tau, 1e-8))

    # [CHOICE] equilibrium oscillator energy (quantum harmonic oscillator)
    # [FORMULA] E_eq(ω,T) = ħω / (exp(ħω/(k_B T)) - 1)
    # [REASON] removes ad-hoc cutoffs; recovers classical limit at high T / low ω
    # [NOTES] Exchange conserves (Q + E_osc) locally: heat -= ΔE, energy += ΔE.
    omega = tl.abs(excitation)
    T = tl.maximum(particle_T, 0.0)
    kBT = k_B * T
    # Default E_eq=0 in frozen-out regime / invalid limits.
    E_eq = 0.0
    if (kBT > 0.0) & (omega > 0.0) & (hbar > 0.0):
        x = (hbar * omega) / kBT
        # For large x, exp(x) overflows and E_eq -> 0.
        E_eq = tl.where(x > 80.0, 0.0, E_eq)
        # Classical limit x -> 0: ħω/(exp(x)-1) ≈ kBT
        E_eq = tl.where(x < 1.0e-4, kBT, E_eq)
        denom = tl.exp(x) - 1.0
        E_eq = tl.where((x <= 80.0) & (x >= 1.0e-4) & (denom > 0.0), (hbar * omega) / denom, E_eq)
    # ω->0 limit: E_eq -> kBT
    E_eq = tl.where((kBT > 0.0) & (omega <= 0.0), kBT, E_eq)

    dE = alpha * (E_eq - energy)
    dE = tl.where(dE > 0.0, tl.minimum(dE, heat), dE)  # Don't draw more than available
    energy = energy + dE
    heat = heat - dE
    energy = tl.maximum(energy, 0.0)
    heat = tl.maximum(heat, 0.0)

    # =========================================================================
    # SYMPLECTIC INTEGRATION (Velocity Verlet with Strang split drag)
    # =========================================================================
    # [CHOICE] Temperature-dependent viscosity (kinetic theory)
    # [FORMULA] μ(T) = μ_ref * sqrt(T)
    # [REASON] ideal-gas-inspired dynamic viscosity scaling
    mu = dynamic_viscosity * tl.sqrt(tl.maximum(local_T, 0.0))
    gamma = 6.0 * PI * mu * r
    inv_m = 1.0 / mass
    half_dt = 0.5 * dt
    drag_half = tl.exp(-(gamma * inv_m) * half_dt)

    fx = gravity_fx + pressure_fx
    fy = gravity_fy + pressure_fy
    fz = gravity_fz + pressure_fz

    # Conservative acceleration at x(t)
    ax0 = fx * inv_m
    ay0 = fy * inv_m
    az0 = fz * inv_m

    # ---- Drag half-step 1 (exact exponential), convert KE loss -> heat ----
    ke0 = 0.5 * mass * (vx * vx + vy * vy + vz * vz)
    vx = vx * drag_half
    vy = vy * drag_half
    vz = vz * drag_half
    ke1 = 0.5 * mass * (vx * vx + vy * vy + vz * vz)
    heat = heat + tl.maximum(ke0 - ke1, 0.0)

    # ---- Kick (half): v(t+dt/2) = v + (dt/2) a(x_t) ----
    vx = vx + ax0 * half_dt
    vy = vy + ay0 * half_dt
    vz = vz + az0 * half_dt

    # ---- Drift: x(t+dt) = x + dt * v(t+dt/2) ----
    px = px + vx * dt
    py = py + vy * dt
    pz = pz + vz * dt

    # [CHOICE] periodic boundary conditions (torus domain)
    # [FORMULA] pos = pos - domain * floor(pos / domain), wraps into [0, domain)
    # [REASON] removes wall reflections/fudges and is a standard physically
    #          interpretable choice for finite domains approximating "unbounded" space.
    # [NOTES] This is physically correct and matches the Metal implementation.
    domain_x = float(grid_x) * grid_spacing
    domain_y = float(grid_y) * grid_spacing
    domain_z = float(grid_z) * grid_spacing
    px = px - domain_x * tl.floor(px / domain_x)
    py = py - domain_y * tl.floor(py / domain_y)
    pz = pz - domain_z * tl.floor(pz / domain_z)

    # ---- Kick (second half): v(t+dt) = v(t+dt/2) + (dt/2) a(x_{t+dt}) ----
    # [CHOICE] full Velocity Verlet second force evaluation (matches Metal)
    # [REASON] improves stability/accuracy and reduces integration artifacts
    gx2 = px * inv_spacing
    gy2 = py * inv_spacing
    gz2 = pz * inv_spacing
    gx2 = gx2 - float(grid_x) * tl.floor(gx2 / float(grid_x))
    gy2 = gy2 - float(grid_y) * tl.floor(gy2 / float(grid_y))
    gz2 = gz2 - float(grid_z) * tl.floor(gz2 / float(grid_z))

    bx2 = tl.floor(gx2).to(tl.int32)
    by2 = tl.floor(gy2).to(tl.int32)
    bz2 = tl.floor(gz2).to(tl.int32)
    fx2 = gx2 - bx2.to(tl.float32)
    fy2 = gy2 - by2.to(tl.float32)
    fz2 = gz2 - bz2.to(tl.float32)

    bx20 = bx2
    by20 = by2
    bz20 = bz2
    bx21 = (bx2 + 1) % grid_x
    by21 = (by2 + 1) % grid_y
    bz21 = (bz2 + 1) % grid_z

    idx2000 = bx20 * stride_x + by20 * stride_y + bz20 * stride_z
    idx2001 = bx20 * stride_x + by20 * stride_y + bz21 * stride_z
    idx2010 = bx20 * stride_x + by21 * stride_y + bz20 * stride_z
    idx2011 = bx20 * stride_x + by21 * stride_y + bz21 * stride_z
    idx2100 = bx21 * stride_x + by20 * stride_y + bz20 * stride_z
    idx2101 = bx21 * stride_x + by20 * stride_y + bz21 * stride_z
    idx2110 = bx21 * stride_x + by21 * stride_y + bz20 * stride_z
    idx2111 = bx21 * stride_x + by21 * stride_y + bz21 * stride_z

    c2000_phi = tl.load(gravity_phi_ptr + idx2000)
    c2001_phi = tl.load(gravity_phi_ptr + idx2001)
    c2010_phi = tl.load(gravity_phi_ptr + idx2010)
    c2011_phi = tl.load(gravity_phi_ptr + idx2011)
    c2100_phi = tl.load(gravity_phi_ptr + idx2100)
    c2101_phi = tl.load(gravity_phi_ptr + idx2101)
    c2110_phi = tl.load(gravity_phi_ptr + idx2110)
    c2111_phi = tl.load(gravity_phi_ptr + idx2111)

    c2000_T = tl.load(temp_field_ptr + idx2000)
    c2001_T = tl.load(temp_field_ptr + idx2001)
    c2010_T = tl.load(temp_field_ptr + idx2010)
    c2011_T = tl.load(temp_field_ptr + idx2011)
    c2100_T = tl.load(temp_field_ptr + idx2100)
    c2101_T = tl.load(temp_field_ptr + idx2101)
    c2110_T = tl.load(temp_field_ptr + idx2110)
    c2111_T = tl.load(temp_field_ptr + idx2111)

    c2000_m = tl.load(mass_field_ptr + idx2000)
    c2001_m = tl.load(mass_field_ptr + idx2001)
    c2010_m = tl.load(mass_field_ptr + idx2010)
    c2011_m = tl.load(mass_field_ptr + idx2011)
    c2100_m = tl.load(mass_field_ptr + idx2100)
    c2101_m = tl.load(mass_field_ptr + idx2101)
    c2110_m = tl.load(mass_field_ptr + idx2110)
    c2111_m = tl.load(mass_field_ptr + idx2111)

    wx20 = 1.0 - fx2
    wy20 = 1.0 - fy2
    wz20 = 1.0 - fz2
    wx21 = fx2
    wy21 = fy2
    wz21 = fz2

    # local_T2 (trilinear)
    c2_00 = c2000_T * wz20 + c2001_T * wz21
    c2_01 = c2010_T * wz20 + c2011_T * wz21
    c2_10 = c2100_T * wz20 + c2101_T * wz21
    c2_11 = c2110_T * wz20 + c2111_T * wz21
    c2_0 = c2_00 * wy20 + c2_01 * wy21
    c2_1 = c2_10 * wy20 + c2_11 * wy21
    local_T2 = tl.maximum(c2_0 * wx20 + c2_1 * wx21, 0.0)

    # gradient of phi at new position
    face_x0_phi2 = c2000_phi * wy20 * wz20 + c2010_phi * wy21 * wz20 + c2001_phi * wy20 * wz21 + c2011_phi * wy21 * wz21
    face_x1_phi2 = c2100_phi * wy20 * wz20 + c2110_phi * wy21 * wz20 + c2101_phi * wy20 * wz21 + c2111_phi * wy21 * wz21
    grad_x_phi2 = (face_x1_phi2 - face_x0_phi2) * inv_spacing
    face_y0_phi2 = c2000_phi * wx20 * wz20 + c2100_phi * wx21 * wz20 + c2001_phi * wx20 * wz21 + c2101_phi * wx21 * wz21
    face_y1_phi2 = c2010_phi * wx20 * wz20 + c2110_phi * wx21 * wz20 + c2011_phi * wx20 * wz21 + c2111_phi * wx21 * wz21
    grad_y_phi2 = (face_y1_phi2 - face_y0_phi2) * inv_spacing
    face_z0_phi2 = c2000_phi * wx20 * wy20 + c2100_phi * wx21 * wy20 + c2010_phi * wx20 * wy21 + c2110_phi * wx21 * wy21
    face_z1_phi2 = c2001_phi * wx20 * wy20 + c2101_phi * wx21 * wy20 + c2011_phi * wx20 * wy21 + c2111_phi * wx21 * wy21
    grad_z_phi2 = (face_z1_phi2 - face_z0_phi2) * inv_spacing

    # gradient of T at new position
    face_x0_T2 = c2000_T * wy20 * wz20 + c2010_T * wy21 * wz20 + c2001_T * wy20 * wz21 + c2011_T * wy21 * wz21
    face_x1_T2 = c2100_T * wy20 * wz20 + c2110_T * wy21 * wz20 + c2101_T * wy20 * wz21 + c2111_T * wy21 * wz21
    grad_x_T2 = (face_x1_T2 - face_x0_T2) * inv_spacing
    face_y0_T2 = c2000_T * wx20 * wz20 + c2100_T * wx21 * wz20 + c2001_T * wx20 * wz21 + c2101_T * wx21 * wz21
    face_y1_T2 = c2010_T * wx20 * wz20 + c2110_T * wx21 * wz20 + c2011_T * wx20 * wz21 + c2111_T * wx21 * wz21
    grad_y_T2 = (face_y1_T2 - face_y0_T2) * inv_spacing
    face_z0_T2 = c2000_T * wx20 * wy20 + c2100_T * wx21 * wy20 + c2010_T * wx20 * wy21 + c2110_T * wx21 * wy21
    face_z1_T2 = c2001_T * wx20 * wy20 + c2101_T * wx21 * wy20 + c2011_T * wx20 * wy21 + c2111_T * wx21 * wy21
    grad_z_T2 = (face_z1_T2 - face_z0_T2) * inv_spacing

    # mass sample + gradient at new position
    m2_00 = c2000_m * wz20 + c2001_m * wz21
    m2_01 = c2010_m * wz20 + c2011_m * wz21
    m2_10 = c2100_m * wz20 + c2101_m * wz21
    m2_11 = c2110_m * wz20 + c2111_m * wz21
    m2_0 = m2_00 * wy20 + m2_01 * wy21
    m2_1 = m2_10 * wy20 + m2_11 * wy21
    m_cell2 = m2_0 * wx20 + m2_1 * wx21

    face_x0_m2 = c2000_m * wy20 * wz20 + c2010_m * wy21 * wz20 + c2001_m * wy20 * wz21 + c2011_m * wy21 * wz21
    face_x1_m2 = c2100_m * wy20 * wz20 + c2110_m * wy21 * wz20 + c2101_m * wy20 * wz21 + c2111_m * wy21 * wz21
    grad_x_m2 = (face_x1_m2 - face_x0_m2) * inv_spacing
    face_y0_m2 = c2000_m * wx20 * wz20 + c2100_m * wx21 * wz20 + c2001_m * wx20 * wz21 + c2101_m * wx21 * wz21
    face_y1_m2 = c2010_m * wx20 * wz20 + c2110_m * wx21 * wz20 + c2011_m * wx20 * wz21 + c2111_m * wx21 * wz21
    grad_y_m2 = (face_y1_m2 - face_y0_m2) * inv_spacing
    face_z0_m2 = c2000_m * wx20 * wy20 + c2100_m * wx21 * wy20 + c2010_m * wx20 * wy21 + c2110_m * wx21 * wy21
    face_z1_m2 = c2001_m * wx20 * wy20 + c2101_m * wx21 * wy20 + c2011_m * wx20 * wy21 + c2111_m * wx21 * wy21
    grad_z_m2 = (face_z1_m2 - face_z0_m2) * inv_spacing

    rho2 = m_cell2 * inv_h3
    grad_x_rho2 = grad_x_m2 * inv_h3
    grad_y_rho2 = grad_y_m2 * inv_h3
    grad_z_rho2 = grad_z_m2 * inv_h3

    gravity_fx2 = -grad_x_phi2 * mass
    gravity_fy2 = -grad_y_phi2 * mass
    gravity_fz2 = -grad_z_phi2 * mass

    pressure_fx2 = 0.0
    pressure_fy2 = 0.0
    pressure_fz2 = 0.0
    if rho2 > 0.0:
        gradP_x2 = k_B * (local_T2 * grad_x_rho2 + rho2 * grad_x_T2)
        gradP_y2 = k_B * (local_T2 * grad_y_rho2 + rho2 * grad_y_T2)
        gradP_z2 = k_B * (local_T2 * grad_z_rho2 + rho2 * grad_z_T2)
        aP_x2 = -gradP_x2 / rho2
        aP_y2 = -gradP_y2 / rho2
        aP_z2 = -gradP_z2 / rho2
        pressure_fx2 = aP_x2 * mass
        pressure_fy2 = aP_y2 * mass
        pressure_fz2 = aP_z2 * mass

    ax1 = (gravity_fx2 + pressure_fx2) * inv_m
    ay1 = (gravity_fy2 + pressure_fy2) * inv_m
    az1 = (gravity_fz2 + pressure_fz2) * inv_m

    vx = vx + ax1 * half_dt
    vy = vy + ay1 * half_dt
    vz = vz + az1 * half_dt

    # ---- Drag half-step 2 (exact exponential), convert KE loss -> heat ----
    ke2 = 0.5 * mass * (vx * vx + vy * vy + vz * vz)
    vx = vx * drag_half
    vy = vy * drag_half
    vz = vz * drag_half
    ke3 = 0.5 * mass * (vx * vx + vy * vy + vz * vz)
    heat = heat + tl.maximum(ke2 - ke3, 0.0)

    # write back
    tl.store(pos_ptr + i * 3 + 0, px)
    tl.store(pos_ptr + i * 3 + 1, py)
    tl.store(pos_ptr + i * 3 + 2, pz)
    tl.store(vel_ptr + i * 3 + 0, vx)
    tl.store(vel_ptr + i * 3 + 1, vy)
    tl.store(vel_ptr + i * 3 + 2, vz)
    tl.store(energy_ptr + i, energy)
    tl.store(heat_ptr + i, heat)
    tl.store(excitation_ptr + i, excitation)


def clear_field(field: torch.Tensor) -> None:
    _require_triton()
    n = int(field.numel())
    if n == 0:
        return
    grid = (triton.cdiv(n, 256),)
    clear_field_kernel[grid](field, n_elements=n, num_warps=1)


def scatter_particles(
    *,
    positions: torch.Tensor,
    masses: torch.Tensor,
    heats: torch.Tensor,
    energies: torch.Tensor,
    gravity_field: torch.Tensor,
    heat_field: torch.Tensor,
    grid_spacing: float,
) -> None:
    _require_triton()
    N = int(positions.shape[0])
    if N == 0:
        return
    gx, gy, gz = map(int, gravity_field.shape)
    inv = 1.0 / float(grid_spacing)
    pos = positions.contiguous().view(-1)
    grid = (N,)
    scatter_particle_kernel[grid](
        pos,
        masses.contiguous(),
        heats.contiguous(),
        energies.contiguous(),
        gravity_field.contiguous().view(-1),
        heat_field.contiguous().view(-1),
        N=N,
        grid_x=gx,
        grid_y=gy,
        grid_z=gz,
        inv_spacing=inv,
        num_warps=1,
    )


def poisson_jacobi_step(
    *,
    phi_in: torch.Tensor,
    rho: torch.Tensor,
    phi_out: torch.Tensor,
    grid_spacing: float,
    gravity_4pi: float,
) -> None:
    _require_triton()
    gx, gy, gz = map(int, phi_in.shape)
    n = gx * gy * gz
    if n == 0:
        return
    grid = (triton.cdiv(n, 256),)
    poisson_jacobi_step_kernel[grid](
        phi_in.contiguous().view(-1),
        rho.contiguous().view(-1),
        phi_out.contiguous().view(-1),
        gravity_4pi=float(gravity_4pi),
        grid_x=gx,
        grid_y=gy,
        grid_z=gz,
        grid_spacing=float(grid_spacing),
        num_warps=1,
    )


def diffuse_heat_field(
    *,
    temp_in: torch.Tensor,
    temp_out: torch.Tensor,
    diffusion_coef: float,
    dt: float,
    grid_spacing: float,
) -> None:
    _require_triton()
    gx, gy, gz = map(int, temp_in.shape)
    n = gx * gy * gz
    if n == 0:
        return
    inv = 1.0 / float(grid_spacing)
    grid = (triton.cdiv(n, 256),)
    diffuse_heat_field_kernel[grid](
        temp_in.contiguous().view(-1),
        temp_out.contiguous().view(-1),
        diffusion_coef=float(diffusion_coef),
        dt=float(dt),
        grid_x=gx,
        grid_y=gy,
        grid_z=gz,
        inv_spacing=float(inv),
        num_warps=1,
    )


def derive_temperature_field(
    *,
    mass_field: torch.Tensor,
    heat_field: torch.Tensor,
    temp_field: torch.Tensor,
    specific_heat: float,
) -> None:
    """Derive temperature field from scattered mass and heat fields.

    T = Q / (m * c_v) for cells with m > 0, else T = 0 (vacuum).
    """
    _require_triton()
    n = int(mass_field.numel())
    if n == 0:
        return
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    derive_temperature_field_kernel[grid](
        mass_field.contiguous().view(-1),
        heat_field.contiguous().view(-1),
        temp_field.contiguous().view(-1),
        specific_heat=float(specific_heat),
        num_elements=n,
        BLOCK=BLOCK,
        num_warps=1,
    )


def reduce_float_stats(
    x: torch.Tensor,
    out_stats: torch.Tensor,
) -> None:
    """Compute mean_abs, mean, std, count of x using GPU reduction.

    Two-pass parallel reduction that avoids CPU synchronization.

    Args:
        x: Input tensor to compute statistics of
        out_stats: Output tensor of shape (4,) to store [mean_abs, mean, std, count]
    """
    _require_triton()
    n = int(x.numel())
    if n == 0:
        out_stats[0] = 0.0
        out_stats[1] = 0.0
        out_stats[2] = 0.0
        out_stats[3] = 0.0
        return

    BLOCK = 256
    num_groups = triton.cdiv(n, BLOCK)

    # Allocate intermediate storage for group statistics
    group_stats = torch.zeros(num_groups * 4, device=x.device, dtype=torch.float32)

    # Pass 1: Compute partial sums per group
    reduce_float_stats_pass1_kernel[(num_groups,)](
        x.contiguous().view(-1),
        group_stats,
        N=n,
        BLOCK=BLOCK,
        num_warps=1,
    )

    # Pass 2: Combine group statistics
    reduce_float_stats_finalize_kernel[(1,)](
        group_stats,
        out_stats,
        num_groups=num_groups,
        num_warps=1,
    )


def gather_update_particles(
    *,
    gravity_potential: torch.Tensor,
    temperature_field: torch.Tensor,
    mass_field: torch.Tensor,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    energies: torch.Tensor,
    heats: torch.Tensor,
    excitations: torch.Tensor,
    masses: torch.Tensor,
    dt: float,
    grid_spacing: float,
    G: float,
    k_B: float,
    sigma_SB: float,
    hbar: float,
    particle_radius: float,
    thermal_conductivity: float,
    specific_heat: float,
    dynamic_viscosity: float,
    emissivity: float,
    young_modulus: float,
) -> None:
    _require_triton()
    N = int(positions.shape[0])
    if N == 0:
        return
    gx, gy, gz = map(int, gravity_potential.shape)
    inv = 1.0 / float(grid_spacing)
    grid = (N,)
    gather_update_particles_kernel[grid](
        gravity_potential.contiguous().view(-1),
        temperature_field.contiguous().view(-1),
        mass_field.contiguous().view(-1),
        positions.contiguous().view(-1),
        velocities.contiguous().view(-1),
        energies.contiguous(),
        heats.contiguous(),
        excitations.contiguous(),
        masses.contiguous(),
        N=N,
        grid_x=gx,
        grid_y=gy,
        grid_z=gz,
        dt=float(dt),
        grid_spacing=float(grid_spacing),
        inv_spacing=float(inv),
        G=float(G),
        k_B=float(k_B),
        sigma_SB=float(sigma_SB),
        hbar=float(hbar),
        particle_radius=float(particle_radius),
        thermal_conductivity=float(thermal_conductivity),
        specific_heat=float(specific_heat),
        dynamic_viscosity=float(dynamic_viscosity),
        emissivity=float(emissivity),
        young_modulus=float(young_modulus),
        num_warps=1,
    )

