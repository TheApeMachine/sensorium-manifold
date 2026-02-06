"""Compressible ideal-gas dynamics (periodic grid, torch implementation).

This module provides a correctness-first implementation of the compressible
Navier–Stokes equations for an ideal gas on a periodic Cartesian grid.

The intent is:
- One auditable implementation used by both MPS (Metal) and CUDA (Triton) runners.
- Small deterministic CPU tests for conservation/regressions.

Numerics:
- Inviscid fluxes: Rusanov/LLF (robust, diffusive).
- Time stepping: RK2 (Heun).
- Spatial derivatives: second-order central differences via periodic rolls.

This is not tuned for performance yet; it is designed to be *internally consistent*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class GasNumerics:
    cfl: float = 0.4
    cfl_diffusion: float = 0.15
    # [CHOICE] vacuum floors (simulation units)
    # [REASON] true vacuum is numerically hostile for explicit compressible solvers;
    #          these floors keep wave speeds and diffusion coefficients bounded.
    rho_min: float = 1e-3
    p_min: float = 1e-3


def _roll(t: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    return torch.roll(t, shifts=int(shift), dims=int(dim))


def central_diff_periodic(f: torch.Tensor, dx: float, dim: int) -> torch.Tensor:
    """Second-order central difference with periodic BC."""
    return (_roll(f, -1, dim) - _roll(f, +1, dim)) * (0.5 / float(dx))


def laplacian_periodic(f: torch.Tensor, dx: float) -> torch.Tensor:
    """3D 7-point Laplacian with periodic BC."""
    inv_dx2 = 1.0 / (float(dx) * float(dx))
    out = (-6.0) * f
    out = out + _roll(f, +1, 0) + _roll(f, -1, 0)
    out = out + _roll(f, +1, 1) + _roll(f, -1, 1)
    out = out + _roll(f, +1, 2) + _roll(f, -1, 2)
    return out * inv_dx2


def conserved_to_primitives(
    rho: torch.Tensor,
    mom: torch.Tensor,
    E: torch.Tensor,
    *,
    gamma: float,
    R_specific: float,
    numerics: GasNumerics,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert conserved (rho, mom, E) → (u, p, T, e_int_density)."""
    rho_safe = torch.clamp(rho, min=float(numerics.rho_min))
    u = mom / rho_safe[..., None]
    ke_density = 0.5 * (mom * mom).sum(dim=-1) / rho_safe
    e_int_density = E - ke_density
    p = (float(gamma) - 1.0) * e_int_density
    p = torch.clamp(p, min=float(numerics.p_min))
    T = p / (rho_safe * float(R_specific))
    return u, p, T, e_int_density


def primitives_to_conserved(
    rho: torch.Tensor,
    u: torch.Tensor,
    p: torch.Tensor,
    *,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mom = rho[..., None] * u
    e_int_density = p / (float(gamma) - 1.0)
    ke_density = 0.5 * rho * (u * u).sum(dim=-1)
    E = e_int_density + ke_density
    return rho, mom, E


def inviscid_flux_x(rho: torch.Tensor, mom: torch.Tensor, E: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return flux vector F in x-direction for conserved variables.

    Output shape: (..., 5) ordered as [rho, mom_x, mom_y, mom_z, E].
    """
    u = mom / rho[..., None]
    ux = u[..., 0]
    flux_rho = mom[..., 0]
    flux_mx = mom[..., 0] * ux + p
    flux_my = mom[..., 1] * ux
    flux_mz = mom[..., 2] * ux
    flux_E = (E + p) * ux
    return torch.stack([flux_rho, flux_mx, flux_my, flux_mz, flux_E], dim=-1)


def inviscid_flux_y(rho: torch.Tensor, mom: torch.Tensor, E: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    u = mom / rho[..., None]
    uy = u[..., 1]
    flux_rho = mom[..., 1]
    flux_mx = mom[..., 0] * uy
    flux_my = mom[..., 1] * uy + p
    flux_mz = mom[..., 2] * uy
    flux_E = (E + p) * uy
    return torch.stack([flux_rho, flux_mx, flux_my, flux_mz, flux_E], dim=-1)


def inviscid_flux_z(rho: torch.Tensor, mom: torch.Tensor, E: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    u = mom / rho[..., None]
    uz = u[..., 2]
    flux_rho = mom[..., 2]
    flux_mx = mom[..., 0] * uz
    flux_my = mom[..., 1] * uz
    flux_mz = mom[..., 2] * uz + p
    flux_E = (E + p) * uz
    return torch.stack([flux_rho, flux_mx, flux_my, flux_mz, flux_E], dim=-1)


def sound_speed(gamma: float, p: torch.Tensor, rho: torch.Tensor, numerics: GasNumerics) -> torch.Tensor:
    rho_safe = torch.clamp(rho, min=float(numerics.rho_min))
    return torch.sqrt(float(gamma) * torch.clamp(p, min=float(numerics.p_min)) / rho_safe)


def _rusanov_flux(
    U_L: torch.Tensor,
    U_R: torch.Tensor,
    F_L: torch.Tensor,
    F_R: torch.Tensor,
    s_max: torch.Tensor,
) -> torch.Tensor:
    # F = 0.5*(F_L+F_R) - 0.5*s_max*(U_R-U_L)
    return 0.5 * (F_L + F_R) - 0.5 * s_max[..., None] * (U_R - U_L)


def _pack_U(rho: torch.Tensor, mom: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    return torch.cat([rho[..., None], mom, E[..., None]], dim=-1)


def _unpack_U(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = U[..., 0]
    mom = U[..., 1:4]
    E = U[..., 4]
    return rho, mom, E


def project_positive_state(
    rho: torch.Tensor,
    mom: torch.Tensor,
    E: torch.Tensor,
    *,
    gamma: float,
    numerics: GasNumerics,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a conserved state into the physically admissible set.

    Enforces:
    - ρ >= rho_min
    - internal energy density e_int = E - 0.5|mom|^2/ρ >= p_min/(γ-1)
    """
    rho = torch.clamp(rho, min=float(numerics.rho_min))
    u = mom / rho[..., None]
    ke_density = 0.5 * rho * (u * u).sum(dim=-1)
    e_int = E - ke_density
    e_int = torch.clamp(e_int, min=float(numerics.p_min) / (float(gamma) - 1.0))
    E = e_int + ke_density
    return rho, mom, E


def _div_flux_periodic(flux: torch.Tensor, dx: float) -> torch.Tensor:
    # flux shape (..., 5), with spatial dims first.
    dFx = (flux - _roll(flux, +1, 0)) / float(dx)
    return dFx


def compute_dt_cfl(
    rho: torch.Tensor,
    mom: torch.Tensor,
    E: torch.Tensor,
    *,
    gamma: float,
    R_specific: float,
    dx: float,
    numerics: GasNumerics,
) -> float:
    """Compute a stable timestep based on advective CFL (device reduction)."""
    u, p, _T, _ = conserved_to_primitives(rho, mom, E, gamma=gamma, R_specific=R_specific, numerics=numerics)
    c = sound_speed(float(gamma), p, rho, numerics)

    # [CHOICE] multi-D CFL for unsplit flux update
    # [FORMULA] dt <= CFL / Σ_d ( (|u_d| + c) / dx )
    # [REASON] the RHS is a sum of directional flux divergences; stability depends on the sum.
    # [NOTES] For dx=dy=dz, this is dt <= CFL * dx / (|u_x|+|u_y|+|u_z| + 3c).
    speed_sum = (u.abs().sum(dim=-1) + 3.0 * c) / float(dx)
    max_rate = float(speed_sum.max().detach().item()) if speed_sum.numel() else 0.0
    if not (max_rate > 0.0):
        return float("inf")
    return float(numerics.cfl) / max_rate


def compute_dt_stable(
    rho: torch.Tensor,
    mom: torch.Tensor,
    E: torch.Tensor,
    *,
    gamma: float,
    R_specific: float,
    dx: float,
    mu: float,
    k_thermal: float,
    c_p: float,
    numerics: GasNumerics,
) -> float:
    """Compute a conservative stable dt including advective + diffusive constraints.

    This is an explicit scheme, so viscosity and thermal conduction require dt ~ O(dx^2).
    """
    dt_adv = compute_dt_cfl(rho, mom, E, gamma=gamma, R_specific=R_specific, dx=dx, numerics=numerics)

    rho_min = float(numerics.rho_min)
    dx2 = float(dx) * float(dx)

    dt = float(dt_adv)

    # Viscous stability (very conservative): dt <= cfl_diff * dx^2 / nu_max, nu=mu/rho.
    if float(mu) > 0.0:
        nu_max = float(mu) / rho_min
        dt_visc = float(numerics.cfl_diffusion) * dx2 / nu_max
        dt = min(dt, dt_visc)

    # Thermal diffusion stability: alpha = k / (rho c_p). Use rho_min bound.
    if float(k_thermal) > 0.0 and float(c_p) > 0.0:
        alpha_max = float(k_thermal) / (rho_min * float(c_p))
        dt_cond = float(numerics.cfl_diffusion) * dx2 / alpha_max
        dt = min(dt, dt_cond)

    return dt


def navier_stokes_rhs(
    rho: torch.Tensor,
    mom: torch.Tensor,
    E: torch.Tensor,
    *,
    gamma: float,
    R_specific: float,
    dx: float,
    mu: float,
    k_thermal: float,
    g_accel: torch.Tensor | None,
    numerics: GasNumerics,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute time derivatives d(rho)/dt, d(mom)/dt, d(E)/dt.

    Viscosity and conductivity are treated as spatially constant in this first pass.
    """
    # Keep the RHS evaluation in the physically admissible set. Without this,
    # intermediate RK stages can generate momentum in near-vacuum cells and
    # produce NaNs (especially on accelerators).
    rho, mom, E = project_positive_state(rho, mom, E, gamma=gamma, numerics=numerics)

    u, p, T, _e_int = conserved_to_primitives(rho, mom, E, gamma=gamma, R_specific=R_specific, numerics=numerics)

    # --------------------------- Inviscid (LLF) ---------------------------
    U = _pack_U(rho, mom, E)

    # x-direction interfaces: i-1/2 uses (i-1, i)
    rho_L = _roll(rho, +1, 0)
    mom_L = _roll(mom, +1, 0)
    E_L = _roll(E, +1, 0)
    uL, pL, _TL, _ = conserved_to_primitives(rho_L, mom_L, E_L, gamma=gamma, R_specific=R_specific, numerics=numerics)

    # Current cell as right state
    uR, pR = u, p

    UL = _pack_U(rho_L, mom_L, E_L)
    UR = U

    FL = inviscid_flux_x(torch.clamp(rho_L, min=numerics.rho_min), mom_L, E_L, pL)
    FR = inviscid_flux_x(torch.clamp(rho, min=numerics.rho_min), mom, E, pR)
    sL = torch.linalg.vector_norm(uL, dim=-1) + sound_speed(gamma, pL, rho_L, numerics)
    sR = torch.linalg.vector_norm(uR, dim=-1) + sound_speed(gamma, pR, rho, numerics)
    smax_x = torch.maximum(sL, sR)
    Fx = _rusanov_flux(UL, UR, FL, FR, smax_x)

    # y-direction
    rho_Ly = _roll(rho, +1, 1)
    mom_Ly = _roll(mom, +1, 1)
    E_Ly = _roll(E, +1, 1)
    uLy, pLy, _TLy, _ = conserved_to_primitives(rho_Ly, mom_Ly, E_Ly, gamma=gamma, R_specific=R_specific, numerics=numerics)
    ULy = _pack_U(rho_Ly, mom_Ly, E_Ly)
    FLy = inviscid_flux_y(torch.clamp(rho_Ly, min=numerics.rho_min), mom_Ly, E_Ly, pLy)
    FRy = inviscid_flux_y(torch.clamp(rho, min=numerics.rho_min), mom, E, p)
    sLy = torch.linalg.vector_norm(uLy, dim=-1) + sound_speed(gamma, pLy, rho_Ly, numerics)
    sRy = torch.linalg.vector_norm(u, dim=-1) + sound_speed(gamma, p, rho, numerics)
    smax_y = torch.maximum(sLy, sRy)
    Fy = _rusanov_flux(ULy, U, FLy, FRy, smax_y)

    # z-direction
    rho_Lz = _roll(rho, +1, 2)
    mom_Lz = _roll(mom, +1, 2)
    E_Lz = _roll(E, +1, 2)
    uLz, pLz, _TLz, _ = conserved_to_primitives(rho_Lz, mom_Lz, E_Lz, gamma=gamma, R_specific=R_specific, numerics=numerics)
    ULz = _pack_U(rho_Lz, mom_Lz, E_Lz)
    FLz = inviscid_flux_z(torch.clamp(rho_Lz, min=numerics.rho_min), mom_Lz, E_Lz, pLz)
    FRz = inviscid_flux_z(torch.clamp(rho, min=numerics.rho_min), mom, E, p)
    sLz = torch.linalg.vector_norm(uLz, dim=-1) + sound_speed(gamma, pLz, rho_Lz, numerics)
    sRz = torch.linalg.vector_norm(u, dim=-1) + sound_speed(gamma, p, rho, numerics)
    smax_z = torch.maximum(sLz, sRz)
    Fz = _rusanov_flux(ULz, U, FLz, FRz, smax_z)

    # Divergence of interface fluxes: ∇·F ≈ (F_{i+1/2} - F_{i-1/2}) / dx.
    # Here `Fx[i]` is the i-1/2 interface flux (between i-1 and i), so the
    # forward-shifted flux `_roll(Fx,-1)` corresponds to i+1/2.
    dUdt = -(
        (_roll(Fx, -1, 0) - Fx) / float(dx)
        + (_roll(Fy, -1, 1) - Fy) / float(dx)
        + (_roll(Fz, -1, 2) - Fz) / float(dx)
    )

    drho_dt, dmom_dt, dE_dt = _unpack_U(dUdt)

    # --------------------------- Viscosity + conduction ---------------------------
    # NOTE: first-pass constant-coefficient model (Stokes hypothesis).
    # Compute ∇u and ∇·u.
    ux, uy, uz = u.unbind(dim=-1)
    dux_dx = central_diff_periodic(ux, dx, 0)
    dux_dy = central_diff_periodic(ux, dx, 1)
    dux_dz = central_diff_periodic(ux, dx, 2)
    duy_dx = central_diff_periodic(uy, dx, 0)
    duy_dy = central_diff_periodic(uy, dx, 1)
    duy_dz = central_diff_periodic(uy, dx, 2)
    duz_dx = central_diff_periodic(uz, dx, 0)
    duz_dy = central_diff_periodic(uz, dx, 1)
    duz_dz = central_diff_periodic(uz, dx, 2)

    div_u = dux_dx + duy_dy + duz_dz
    grad_div_u = torch.stack(
        [
            central_diff_periodic(div_u, dx, 0),
            central_diff_periodic(div_u, dx, 1),
            central_diff_periodic(div_u, dx, 2),
        ],
        dim=-1,
    )

    lap_u = torch.stack([laplacian_periodic(ux, dx), laplacian_periodic(uy, dx), laplacian_periodic(uz, dx)], dim=-1)

    mu_f = float(mu)
    lam = -(2.0 / 3.0) * mu_f
    div_tau = mu_f * lap_u + (mu_f + lam) * grad_div_u

    dmom_dt = dmom_dt + div_tau

    # Viscous work term in total energy: ∇·(τ · u)
    # Construct τ_ij from symmetric gradient + bulk term.
    # S_ij = ∂u_i/∂x_j + ∂u_j/∂x_i
    # τ_ij = μ S_ij + λ (∇·u) δ_ij
    # Then v_i = Σ_j τ_ij u_j and add ∇·v to energy.
    # Diagonal terms:
    tau_xx = mu_f * (2.0 * dux_dx) + lam * div_u
    tau_yy = mu_f * (2.0 * duy_dy) + lam * div_u
    tau_zz = mu_f * (2.0 * duz_dz) + lam * div_u
    # Off-diagonals (symmetric):
    tau_xy = mu_f * (dux_dy + duy_dx)
    tau_xz = mu_f * (dux_dz + duz_dx)
    tau_yz = mu_f * (duy_dz + duz_dy)

    v_x = tau_xx * ux + tau_xy * uy + tau_xz * uz
    v_y = tau_xy * ux + tau_yy * uy + tau_yz * uz
    v_z = tau_xz * ux + tau_yz * uy + tau_zz * uz
    div_tau_u = (
        central_diff_periodic(v_x, dx, 0)
        + central_diff_periodic(v_y, dx, 1)
        + central_diff_periodic(v_z, dx, 2)
    )
    dE_dt = dE_dt + div_tau_u

    # Heat conduction: ∇·(k ∇T) = k ∇²T (constant k)
    dE_dt = dE_dt + float(k_thermal) * laplacian_periodic(T, dx)

    # --------------------------- Gravity source ---------------------------
    if g_accel is not None:
        # g_accel is (gx,gy,gz,3) acceleration field.
        dmom_dt = dmom_dt + rho[..., None] * g_accel
        dE_dt = dE_dt + (rho[..., None] * g_accel * u).sum(dim=-1)

    return drho_dt, dmom_dt, dE_dt


def advance_navier_stokes_rk2(
    rho: torch.Tensor,
    mom: torch.Tensor,
    E: torch.Tensor,
    *,
    gamma: float,
    R_specific: float,
    dx: float,
    dt: float,
    mu: float,
    k_thermal: float,
    g_accel: torch.Tensor | None,
    numerics: GasNumerics,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance (rho,mom,E) one timestep with RK2 (Heun)."""

    def _project_positive(
        rho_t: torch.Tensor, mom_t: torch.Tensor, E_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return project_positive_state(rho_t, mom_t, E_t, gamma=gamma, numerics=numerics)

    k1_rho, k1_mom, k1_E = navier_stokes_rhs(
        rho, mom, E,
        gamma=gamma, R_specific=R_specific, dx=dx,
        mu=mu, k_thermal=k_thermal, g_accel=g_accel,
        numerics=numerics,
    )
    rho1 = rho + float(dt) * k1_rho
    mom1 = mom + float(dt) * k1_mom
    E1 = E + float(dt) * k1_E
    rho1, mom1, E1 = _project_positive(rho1, mom1, E1)

    k2_rho, k2_mom, k2_E = navier_stokes_rhs(
        rho1, mom1, E1,
        gamma=gamma, R_specific=R_specific, dx=dx,
        mu=mu, k_thermal=k_thermal, g_accel=g_accel,
        numerics=numerics,
    )

    rho_next = rho + 0.5 * float(dt) * (k1_rho + k2_rho)
    mom_next = mom + 0.5 * float(dt) * (k1_mom + k2_mom)
    E_next = E + 0.5 * float(dt) * (k1_E + k2_E)

    rho_next, mom_next, E_next = _project_positive(rho_next, mom_next, E_next)

    return rho_next, mom_next, E_next

