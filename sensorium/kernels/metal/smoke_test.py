"""Metal/MPS smoke test for manifold backend.

Run on Apple Silicon with MPS:

```bash
python -m sensorium.kernels.metal.smoke_test
```
"""

from __future__ import annotations

import torch

from sensorium.kernels.metal.manifold_physics import (
    ThermodynamicsDomain,
    ThermodynamicsDomainConfig,
    HydrodynamicFieldConfig,
    HydrodynamicDomain,
)


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable; run this on an Apple Silicon machine with MPS enabled.")

    device = "mps"

    # ---- spectral smoke ----
    N = 1024
    osc_phase = torch.rand(N, device=device, dtype=torch.float32) * (2.0 * torch.pi)
    exc = torch.randn(N, device=device, dtype=torch.float32)
    energy = torch.rand(N, device=device, dtype=torch.float32)
    pos = torch.rand(N, 3, device=device, dtype=torch.float32) * 8.0
    scfg = HydrodynamicFieldConfig(omega_bins=256)
    sp = HydrodynamicDomain(scfg, grid_size=(8, 8, 8), dt=0.01, device=device)
    out = sp.step(osc_phase, exc, energy, particle_positions=pos)
    assert "osc_energy" in out
    assert torch.isfinite(out["osc_phase"]).all().item()
    assert torch.isfinite(out["osc_energy"]).all().item()

    # ---- grid physics smoke ----
    pcfg = ThermodynamicsDomainConfig(grid_size=(16, 16, 16), dt_max=0.01)
    mp = ThermodynamicsDomain(pcfg, device=device)

    P = 2048
    positions = torch.rand(P, 3, device=device, dtype=torch.float32) * 12.0 + 1.0
    velocities = torch.zeros(P, 3, device=device, dtype=torch.float32)
    energies = torch.rand(P, device=device, dtype=torch.float32)
    heats = torch.zeros(P, device=device, dtype=torch.float32)
    excitations = torch.rand(P, device=device, dtype=torch.float32)
    masses = torch.ones(P, device=device, dtype=torch.float32)

    mp.step(positions, velocities, energies, heats, excitations, masses)

    # ---- integrated tick smoke (thermo + hydrodynamic) ----
    # Ensure both domains can be advanced on the same particle set.
    osc_phase2 = torch.rand(P, device=device, dtype=torch.float32) * (2.0 * torch.pi)
    osc_energy2 = energies.clone()
    for _ in range(3):
        positions, velocities, energies, heats, excitations = mp.step(
            positions, velocities, energies, heats, excitations, masses
        )
        out_h = sp.step(osc_phase2, excitations, osc_energy2, particle_positions=positions)
        osc_phase2 = out_h["osc_phase"]
        assert torch.isfinite(out_h["amplitudes"]).all().item()

    # ---- collision API smoke ----
    # Collision routines are disabled in the single-model codebase.
    try:
        mp.compute_interactions_spatial_hash(
            positions[:64],
            torch.randn(64, 3, device=device, dtype=torch.float32) * 0.1,
            torch.rand(64, device=device, dtype=torch.float32),
            torch.ones(64, device=device, dtype=torch.float32),
            torch.rand(64, device=device, dtype=torch.float32),
            dt=0.01,
        )
        raise AssertionError("Expected compute_interactions_spatial_hash() to be disabled")
    except RuntimeError:
        pass

    print("Metal smoke test passed.")


if __name__ == "__main__":
    main()

