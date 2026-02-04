"""Metal/MPS smoke test for manifold backend.

Run on Apple Silicon with MPS:

```bash
python -m optimizer.metal.smoke_test
```
"""

from __future__ import annotations

import torch

from optimizer.metal.manifold_physics import (
    ManifoldPhysics,
    ManifoldPhysicsConfig,
    SpectralCarrierConfig,
    SpectralCarrierPhysics,
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

    scfg = SpectralCarrierConfig(max_carriers=256)
    sp = SpectralCarrierPhysics(scfg, grid_size=(8, 8, 8), dt=0.01, device=device)
    out = sp.step(osc_phase, exc, energy)
    assert "carrier_state" in out and "osc_energy" in out
    assert torch.isfinite(out["osc_phase"]).all().item()
    assert torch.isfinite(out["osc_energy"]).all().item()

    out2 = sp.idle_compute(out["osc_phase"], exc, out["osc_energy"], steps=2, mode="explore")
    assert out2["frequencies"].numel() >= 0

    # ---- grid physics smoke ----
    pcfg = ManifoldPhysicsConfig(grid_size=(16, 16, 16), dt=0.01)
    mp = ManifoldPhysics(pcfg, device=device)

    P = 2048
    positions = torch.rand(P, 3, device=device, dtype=torch.float32) * 12.0 + 1.0
    velocities = torch.zeros(P, 3, device=device, dtype=torch.float32)
    energies = torch.rand(P, device=device, dtype=torch.float32)
    heats = torch.zeros(P, device=device, dtype=torch.float32)
    excitations = torch.rand(P, device=device, dtype=torch.float32)
    masses = torch.ones(P, device=device, dtype=torch.float32)

    mp.step(positions, velocities, energies, heats, excitations, masses)

    # ---- spatial hash collisions smoke ----
    # Exercise the full pipeline: assign → scan → scatter → collide
    P2 = 8192
    positions2 = torch.rand(P2, 3, device=device, dtype=torch.float32) * 12.0 + 1.0
    velocities2 = torch.randn(P2, 3, device=device, dtype=torch.float32) * 0.1
    excitations2 = torch.rand(P2, device=device, dtype=torch.float32)
    masses2 = torch.rand(P2, device=device, dtype=torch.float32) + 0.5
    heats2 = torch.rand(P2, device=device, dtype=torch.float32)

    v_out, e_out, h_out = mp.compute_interactions_spatial_hash(
        positions2, velocities2, excitations2, masses2, heats2, dt=0.01
    )
    assert torch.isfinite(v_out).all().item()
    assert torch.isfinite(e_out).all().item()
    assert torch.isfinite(h_out).all().item()

    print("Metal smoke test passed.")


if __name__ == "__main__":
    main()

