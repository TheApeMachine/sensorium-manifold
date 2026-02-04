"""CUDA/Triton smoke test for manifold backends.

Run on a CUDA machine:

```bash
python -m optimizer.triton.smoke_test
```
"""

from __future__ import annotations

import torch

from optimizer.triton.manifold_physics import (
    ManifoldPhysics,
    ManifoldPhysicsConfig,
    SpectralCarrierConfig,
    SpectralCarrierPhysics,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; run this on an NVIDIA machine.")

    device = "cuda"

    # ---- spectral smoke ----
    N = 512
    osc_phase = torch.rand(N, device=device, dtype=torch.float32) * (2.0 * torch.pi)
    exc = torch.randn(N, device=device, dtype=torch.float32)
    energy = torch.rand(N, device=device, dtype=torch.float32)

    scfg = SpectralCarrierConfig(max_carriers=32)
    sp = SpectralCarrierPhysics(scfg, grid_size=(8, 8, 8), dt=0.01, device=device)
    out = sp.step(osc_phase, exc, energy)
    assert "carrier_state" in out and "osc_energy" in out
    out2 = sp.idle_compute(out["osc_phase"], exc, out["osc_energy"], steps=2, mode="explore")
    assert out2["frequencies"].numel() >= 0

    # ---- grid physics smoke ----
    pcfg = ManifoldPhysicsConfig(grid_size=(16, 16, 16), dt=0.01)
    mp = ManifoldPhysics(pcfg, device=device)

    P = 256
    positions = torch.rand(P, 3, device=device, dtype=torch.float32) * 12.0 + 1.0
    velocities = torch.zeros(P, 3, device=device, dtype=torch.float32)
    energies = torch.rand(P, device=device, dtype=torch.float32)
    heats = torch.zeros(P, device=device, dtype=torch.float32)
    excitations = torch.rand(P, device=device, dtype=torch.float32)
    masses = torch.ones(P, device=device, dtype=torch.float32)

    mp.step(positions, velocities, energies, heats, excitations, masses)

    print("Triton smoke test passed.")


if __name__ == "__main__":
    main()

