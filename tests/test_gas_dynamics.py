"""Regression tests for compressible ideal-gas dynamics (periodic grid).

These tests run on CPU and validate:
- Global conservation in a periodic domain (mass, momentum, energy).
- Basic shock-tube sanity (positivity + finite values).
- Viscous decay of a shear wave (qualitative rate check).
- Thermal diffusion reduces temperature extrema.
"""

from __future__ import annotations

import math

import pytest
import torch

from sensorium.kernels.gas_dynamics import (
    GasNumerics,
    advance_navier_stokes_rk2,
    compute_dt_cfl,
    compute_dt_stable,
    conserved_to_primitives,
    primitives_to_conserved,
)


def _make_grid(nx: int, ny: int = 1, nz: int = 1, *, device: str = "cpu") -> tuple[int, int, int, torch.device]:
    dev = torch.device(device)
    return int(nx), int(ny), int(nz), dev


def _sum3(t: torch.Tensor) -> float:
    return float(t.to(torch.float64).sum().detach().cpu().item())


@pytest.mark.parametrize("shape", [(32, 32, 32), (64, 16, 8)])
def test_conservation_periodic_inviscid(shape):
    gx, gy, gz = shape
    dx = 1.0
    dt_max = 1e-3
    gamma = 1.4
    R = 287.0  # J/(kg K) (SI-like)
    numerics = GasNumerics(rho_min=1e-9, p_min=1e-9)

    # Smooth, positive initial state.
    xs = torch.linspace(0.0, 2.0 * math.pi, gx, dtype=torch.float32)
    ys = torch.linspace(0.0, 2.0 * math.pi, gy, dtype=torch.float32)
    zs = torch.linspace(0.0, 2.0 * math.pi, gz, dtype=torch.float32)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")

    rho = 1.0 + 0.1 * torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    u = torch.stack(
        [
            0.05 * torch.sin(Y),
            0.05 * torch.sin(Z),
            0.05 * torch.sin(X),
        ],
        dim=-1,
    )
    T = 300.0 + 5.0 * torch.cos(X)
    p = rho * R * T
    rho, mom, E = primitives_to_conserved(rho, u, p, gamma=gamma)

    mass0 = _sum3(rho)
    mom0 = tuple(_sum3(mom[..., i]) for i in range(3))
    E0 = _sum3(E)

    for _ in range(10):
        dt_cfl = compute_dt_cfl(rho, mom, E, gamma=gamma, R_specific=R, dx=dx, numerics=numerics)
        dt = min(dt_max, float(dt_cfl)) if (math.isfinite(float(dt_cfl)) and float(dt_cfl) > 0.0) else dt_max
        rho, mom, E = advance_navier_stokes_rk2(
            rho, mom, E,
            gamma=gamma, R_specific=R,
            dx=dx, dt=dt,
            mu=0.0, k_thermal=0.0,
            g_accel=None,
            numerics=numerics,
        )

    mass1 = _sum3(rho)
    mom1 = tuple(_sum3(mom[..., i]) for i in range(3))
    E1 = _sum3(E)

    assert abs(mass1 - mass0) / max(1.0, abs(mass0)) < 5e-6
    for i in range(3):
        assert abs(mom1[i] - mom0[i]) / max(1.0, abs(mom0[i])) < 5e-4
    assert abs(E1 - E0) / max(1.0, abs(E0)) < 5e-6


def test_sod_shocktube_positivity_and_finite():
    gx, gy, gz, dev = _make_grid(256, 1, 1)
    dx = 1.0
    gamma = 1.4
    R = 287.0
    numerics = GasNumerics(rho_min=1e-8, p_min=1e-8)

    rho = torch.ones(gx, gy, gz, device=dev, dtype=torch.float32)
    u = torch.zeros(gx, gy, gz, 3, device=dev, dtype=torch.float32)

    rho[: gx // 2] = 1.0
    rho[gx // 2 :] = 0.125
    p = torch.ones(gx, gy, gz, device=dev, dtype=torch.float32)
    p[: gx // 2] = 1.0
    p[gx // 2 :] = 0.1

    rho, mom, E = primitives_to_conserved(rho, u, p, gamma=gamma)

    dt_max = 2e-4
    for _ in range(200):
        dt_cfl = compute_dt_cfl(rho, mom, E, gamma=gamma, R_specific=R, dx=dx, numerics=numerics)
        dt = min(dt_max, float(dt_cfl)) if (math.isfinite(float(dt_cfl)) and float(dt_cfl) > 0.0) else dt_max
        rho, mom, E = advance_navier_stokes_rk2(
            rho, mom, E,
            gamma=gamma, R_specific=R,
            dx=dx, dt=dt,
            mu=0.0, k_thermal=0.0,
            g_accel=None,
            numerics=numerics,
        )

    u_out, p_out, T_out, _ = conserved_to_primitives(rho, mom, E, gamma=gamma, R_specific=R, numerics=numerics)
    assert torch.isfinite(rho).all()
    assert torch.isfinite(u_out).all()
    assert torch.isfinite(p_out).all()
    assert torch.isfinite(T_out).all()
    assert float(rho.min().detach().item()) > 0.0
    assert float(p_out.min().detach().item()) > 0.0


def test_viscous_shear_wave_decays():
    gx, gy, gz, dev = _make_grid(128, 1, 1)
    dx = 1.0
    gamma = 1.4
    R = 287.0
    numerics = GasNumerics(rho_min=1e-8, p_min=1e-8)

    x = torch.arange(gx, device=dev, dtype=torch.float32) * dx
    L = gx * dx
    k = 2.0 * math.pi / L

    rho = torch.ones(gx, gy, gz, device=dev, dtype=torch.float32)
    u = torch.zeros(gx, gy, gz, 3, device=dev, dtype=torch.float32)
    A0 = 0.1
    u[..., 1] = A0 * torch.sin(k * x)[:, None, None]  # u_y(x)
    p = torch.ones(gx, gy, gz, device=dev, dtype=torch.float32) * (rho * R * 300.0)

    rho, mom, E = primitives_to_conserved(rho, u, p, gamma=gamma)

    mu = 0.05
    dt_max = 5e-3
    steps = 50
    t = 0.0
    cp = gamma * R / (gamma - 1.0)

    for _ in range(steps):
        dt_s = compute_dt_stable(
            rho, mom, E,
            gamma=gamma, R_specific=R, dx=dx,
            mu=mu, k_thermal=0.0, c_p=cp,
            numerics=numerics,
        )
        dt = min(dt_max, float(dt_s)) if (math.isfinite(float(dt_s)) and float(dt_s) > 0.0) else dt_max
        t += float(dt)
        rho, mom, E = advance_navier_stokes_rk2(
            rho, mom, E,
            gamma=gamma, R_specific=R,
            dx=dx, dt=dt,
            mu=mu, k_thermal=0.0,
            g_accel=None,
            numerics=numerics,
        )

    u_out, _p_out, _T_out, _ = conserved_to_primitives(rho, mom, E, gamma=gamma, R_specific=R, numerics=numerics)
    uy = u_out[..., 1].squeeze()
    # Estimate amplitude via projection onto sin(kx)
    s = torch.sin(k * x)
    A_est = float((uy * s).mean().detach().cpu().item()) * 2.0

    # For ρ≈1, diffusion rate ~ exp(-μ k^2 t) (qualitative).
    A_pred = A0 * math.exp(-mu * (k * k) * t)
    assert A_est <= A0 * 1.001
    assert A_est == pytest.approx(A_pred, rel=0.35, abs=0.02)


def test_thermal_diffusion_reduces_temperature_extrema():
    gx, gy, gz, dev = _make_grid(64, 64, 1)
    dx = 1.0
    gamma = 1.4
    R = 287.0
    numerics = GasNumerics(rho_min=1e-8, p_min=1e-8)

    xs = torch.linspace(-1.0, 1.0, gx, device=dev, dtype=torch.float32)
    ys = torch.linspace(-1.0, 1.0, gy, device=dev, dtype=torch.float32)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    rho = torch.ones(gx, gy, gz, device=dev, dtype=torch.float32)
    u = torch.zeros(gx, gy, gz, 3, device=dev, dtype=torch.float32)
    T = 300.0 + 50.0 * torch.exp(-20.0 * (X * X + Y * Y))[:, :, None]
    p = rho * R * T
    rho, mom, E = primitives_to_conserved(rho, u, p, gamma=gamma)

    mu = 0.0
    k_th = 0.5
    dt_max = 1e-3
    cp = gamma * R / (gamma - 1.0)

    u0, p0, T0, _ = conserved_to_primitives(rho, mom, E, gamma=gamma, R_specific=R, numerics=numerics)
    Tmax0 = float(T0.max().detach().cpu().item())
    Tmin0 = float(T0.min().detach().cpu().item())
    Etot0 = _sum3(E)

    for _ in range(100):
        dt_s = compute_dt_stable(
            rho, mom, E,
            gamma=gamma, R_specific=R, dx=dx,
            mu=mu, k_thermal=k_th, c_p=cp,
            numerics=numerics,
        )
        dt = min(dt_max, float(dt_s)) if (math.isfinite(float(dt_s)) and float(dt_s) > 0.0) else dt_max
        rho, mom, E = advance_navier_stokes_rk2(
            rho, mom, E,
            gamma=gamma, R_specific=R,
            dx=dx, dt=dt,
            mu=mu, k_thermal=k_th,
            g_accel=None,
            numerics=numerics,
        )

    _u1, _p1, T1, _ = conserved_to_primitives(rho, mom, E, gamma=gamma, R_specific=R, numerics=numerics)
    Tmax1 = float(T1.max().detach().cpu().item())
    Tmin1 = float(T1.min().detach().cpu().item())
    Etot1 = _sum3(E)

    # Conduction should not significantly increase extrema (allow tiny numerical wiggle).
    assert Tmax1 <= Tmax0 * 1.001
    assert Tmin1 >= Tmin0 - 1e-3
    # Energy should remain globally conserved in periodic domain (diffusion is divergence term).
    assert abs(Etot1 - Etot0) / max(1.0, abs(Etot0)) < 1e-5

