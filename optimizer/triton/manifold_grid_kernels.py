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
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR: Exception = e
else:
    _TRITON_IMPORT_ERROR = RuntimeError("unreachable")


def _require_triton() -> None:
    if triton is None or tl is None:  # pragma: no cover
        raise RuntimeError(f"Triton is required for CUDA grid backend: {_TRITON_IMPORT_ERROR!r}")


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
    heat_ptr,  # fp32 [N]
    gravity_ptr,  # fp32 [X*Y*Z] atomic
    temp_ptr,  # fp32 [X*Y*Z] atomic
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
    h = tl.load(heat_ptr + i)

    gx = px * inv_spacing
    gy = py * inv_spacing
    gz = pz * inv_spacing

    # clamp to [0, dim-1.001] so base+1 is valid
    gx = tl.maximum(0.0, tl.minimum(gx, float(grid_x) - 1.001))
    gy = tl.maximum(0.0, tl.minimum(gy, float(grid_y) - 1.001))
    gz = tl.maximum(0.0, tl.minimum(gz, float(grid_z) - 1.001))

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

    # 8 corners
    stride_z = 1
    stride_y = grid_z
    stride_x = grid_y * grid_z
    base = bx * stride_x + by * stride_y + bz * stride_z

    w000 = wx0 * wy0 * wz0
    w001 = wx0 * wy0 * wz1
    w010 = wx0 * wy1 * wz0
    w011 = wx0 * wy1 * wz1
    w100 = wx1 * wy0 * wz0
    w101 = wx1 * wy0 * wz1
    w110 = wx1 * wy1 * wz0
    w111 = wx1 * wy1 * wz1

    tl.atomic_add(gravity_ptr + base + 0, m * w000)
    tl.atomic_add(gravity_ptr + base + stride_z, m * w001)
    tl.atomic_add(gravity_ptr + base + stride_y, m * w010)
    tl.atomic_add(gravity_ptr + base + stride_y + stride_z, m * w011)
    tl.atomic_add(gravity_ptr + base + stride_x, m * w100)
    tl.atomic_add(gravity_ptr + base + stride_x + stride_z, m * w101)
    tl.atomic_add(gravity_ptr + base + stride_x + stride_y, m * w110)
    tl.atomic_add(gravity_ptr + base + stride_x + stride_y + stride_z, m * w111)

    tl.atomic_add(temp_ptr + base + 0, h * w000)
    tl.atomic_add(temp_ptr + base + stride_z, h * w001)
    tl.atomic_add(temp_ptr + base + stride_y, h * w010)
    tl.atomic_add(temp_ptr + base + stride_y + stride_z, h * w011)
    tl.atomic_add(temp_ptr + base + stride_x, h * w100)
    tl.atomic_add(temp_ptr + base + stride_x + stride_z, h * w101)
    tl.atomic_add(temp_ptr + base + stride_x + stride_y, h * w110)
    tl.atomic_add(temp_ptr + base + stride_x + stride_y + stride_z, h * w111)


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

    xm = tl.where(x > 0, tl.load(phi_in_ptr + idx - stride_x, mask=m, other=0.0), 0.0)
    xp = tl.where(x < (grid_x - 1), tl.load(phi_in_ptr + idx + stride_x, mask=m, other=0.0), 0.0)
    ym = tl.where(y > 0, tl.load(phi_in_ptr + idx - stride_y, mask=m, other=0.0), 0.0)
    yp = tl.where(y < (grid_y - 1), tl.load(phi_in_ptr + idx + stride_y, mask=m, other=0.0), 0.0)
    zm = tl.where(z > 0, tl.load(phi_in_ptr + idx - stride_z, mask=m, other=0.0), 0.0)
    zp = tl.where(z < (grid_z - 1), tl.load(phi_in_ptr + idx + stride_z, mask=m, other=0.0), 0.0)

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
    # boundary handling: use center value on boundary (Neumann-ish)
    xm = tl.where(x > 0, tl.load(temp_in_ptr + idx - stride_x, mask=m, other=center), center)
    xp = tl.where(x < (grid_x - 1), tl.load(temp_in_ptr + idx + stride_x, mask=m, other=center), center)
    ym = tl.where(y > 0, tl.load(temp_in_ptr + idx - stride_y, mask=m, other=center), center)
    yp = tl.where(y < (grid_y - 1), tl.load(temp_in_ptr + idx + stride_y, mask=m, other=center), center)
    zm = tl.where(z > 0, tl.load(temp_in_ptr + idx - stride_z, mask=m, other=center), center)
    zp = tl.where(z < (grid_z - 1), tl.load(temp_in_ptr + idx + stride_z, mask=m, other=center), center)

    lap = (xm + xp + ym + yp + zm + zp - 6.0 * center) * (inv_spacing * inv_spacing)
    out = center + diffusion_coef * lap * dt
    tl.store(temp_out_ptr + idx, out, mask=m)


@triton.jit
def gather_update_particles_kernel(
    gravity_phi_ptr,  # fp32 [X*Y*Z]
    temp_field_ptr,  # fp32 [X*Y*Z]
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

    # trilinear coords
    gx = tl.maximum(0.0, tl.minimum(px * inv_spacing, float(grid_x) - 1.001))
    gy = tl.maximum(0.0, tl.minimum(py * inv_spacing, float(grid_y) - 1.001))
    gz = tl.maximum(0.0, tl.minimum(pz * inv_spacing, float(grid_z) - 1.001))

    bx = tl.floor(gx).to(tl.int32)
    by = tl.floor(gy).to(tl.int32)
    bz = tl.floor(gz).to(tl.int32)
    fx = gx - bx.to(tl.float32)
    fy = gy - by.to(tl.float32)
    fz = gz - bz.to(tl.float32)

    stride_z = 1
    stride_y = grid_z
    stride_x = grid_y * grid_z
    base = bx * stride_x + by * stride_y + bz * stride_z

    # corners
    c000_phi = tl.load(gravity_phi_ptr + base + 0)
    c001_phi = tl.load(gravity_phi_ptr + base + stride_z)
    c010_phi = tl.load(gravity_phi_ptr + base + stride_y)
    c011_phi = tl.load(gravity_phi_ptr + base + stride_y + stride_z)
    c100_phi = tl.load(gravity_phi_ptr + base + stride_x)
    c101_phi = tl.load(gravity_phi_ptr + base + stride_x + stride_z)
    c110_phi = tl.load(gravity_phi_ptr + base + stride_x + stride_y)
    c111_phi = tl.load(gravity_phi_ptr + base + stride_x + stride_y + stride_z)

    c000_T = tl.load(temp_field_ptr + base + 0)
    c001_T = tl.load(temp_field_ptr + base + stride_z)
    c010_T = tl.load(temp_field_ptr + base + stride_y)
    c011_T = tl.load(temp_field_ptr + base + stride_y + stride_z)
    c100_T = tl.load(temp_field_ptr + base + stride_x)
    c101_T = tl.load(temp_field_ptr + base + stride_x + stride_z)
    c110_T = tl.load(temp_field_ptr + base + stride_x + stride_y)
    c111_T = tl.load(temp_field_ptr + base + stride_x + stride_y + stride_z)

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

    # forces
    gravity_fx = -grad_x_phi * mass * G
    gravity_fy = -grad_y_phi * mass * G
    gravity_fz = -grad_z_phi * mass * G

    particle_T = heat / (tl.maximum(mass, 1e-6) * specific_heat)
    pressure_fx = -grad_x_T * k_B * mass
    pressure_fy = -grad_y_T * k_B * mass
    pressure_fz = -grad_z_T * k_B * mass

    # heat transfer
    r = particle_radius
    heat_transfer_coef = thermal_conductivity * r
    dQ_cond = heat_transfer_coef * (local_T - particle_T) * dt
    heat = heat + dQ_cond
    surface_area = r * r
    T4 = particle_T * particle_T * particle_T * particle_T
    dQ_rad = emissivity * sigma_SB * surface_area * T4 * dt
    heat = tl.maximum(heat - dQ_rad, 0.0)
    particle_T = heat / (tl.maximum(mass, 1e-6) * specific_heat)

    # thermalization
    tau = 10.0
    dQ_th = (energy / tau) * dt
    dQ_th = tl.minimum(dQ_th, energy)
    energy = energy - dQ_th
    heat = heat + dQ_th
    energy = tl.maximum(energy, 0.0)
    heat = tl.maximum(heat, 0.0)

    gamma = 6.0 * 3.14159 * dynamic_viscosity * r
    fx = gravity_fx + pressure_fx
    fy = gravity_fy + pressure_fy
    fz = gravity_fz + pressure_fz

    ax = fx / tl.maximum(mass, 1e-6)
    ay = fy / tl.maximum(mass, 1e-6)
    az = fz / tl.maximum(mass, 1e-6)

    # clamp acceleration
    acc_mag = tl.sqrt(ax * ax + ay * ay + az * az)
    max_acc = 10.0
    scale = tl.where(acc_mag > max_acc, max_acc / acc_mag, 1.0)
    ax = ax * scale
    ay = ay * scale
    az = az * scale

    ke_before = 0.5 * mass * (vx * vx + vy * vy + vz * vz)
    damp = tl.exp(-gamma * dt)
    vx = vx * damp + ax * dt
    vy = vy * damp + ay * dt
    vz = vz * damp + az * dt
    ke_after = 0.5 * mass * (vx * vx + vy * vy + vz * vz)
    ke_lost = tl.maximum(ke_before - ke_after, 0.0)
    heat = heat + ke_lost

    vel_mag = tl.sqrt(vx * vx + vy * vy + vz * vz)
    max_vel = 2.0
    vscale = tl.where(vel_mag > max_vel, max_vel / vel_mag, 1.0)
    vx = vx * vscale
    vy = vy * vscale
    vz = vz * vscale

    # position update
    px = px + vx * dt
    py = py + vy * dt
    pz = pz + vz * dt

    grid_max_x = float(grid_x) * grid_spacing * 0.95
    grid_max_y = float(grid_y) * grid_spacing * 0.95
    grid_max_z = float(grid_z) * grid_spacing * 0.95
    grid_min = 0.5

    # reflect boundaries
    if px < grid_min:
        px = grid_min
        vx = tl.abs(vx) * 0.5
    if py < grid_min:
        py = grid_min
        vy = tl.abs(vy) * 0.5
    if pz < grid_min:
        pz = grid_min
        vz = tl.abs(vz) * 0.5
    if px > grid_max_x:
        px = grid_max_x
        vx = -tl.abs(vx) * 0.5
    if py > grid_max_y:
        py = grid_max_y
        vy = -tl.abs(vy) * 0.5
    if pz > grid_max_z:
        pz = grid_max_z
        vz = -tl.abs(vz) * 0.5

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
    gravity_field: torch.Tensor,
    temperature_field: torch.Tensor,
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
        gravity_field.contiguous().view(-1),
        temperature_field.contiguous().view(-1),
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


def gather_update_particles(
    *,
    gravity_potential: torch.Tensor,
    temperature_field: torch.Tensor,
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
        particle_radius=float(particle_radius),
        thermal_conductivity=float(thermal_conductivity),
        specific_heat=float(specific_heat),
        dynamic_viscosity=float(dynamic_viscosity),
        emissivity=float(emissivity),
        young_modulus=float(young_modulus),
        num_warps=1,
    )

