from __future__ import annotations

import torch

from thermo_manifold.bridge.manifold import BridgeManifold


def test_bridge_heat_increases_on_mismatch() -> None:
    device = torch.device("cpu")
    sem_dim = 4
    bins = torch.linspace(0.0, 1.0, steps=8, device=device)
    bridge = BridgeManifold(sem_dim=sem_dim, spec_bins=bins, dt=1e-2, device=device, num_carriers=4)

    # Contradictory evidence: fixed semantic inputs with alternating spectral positions.
    sem_pos = torch.randn(6, sem_dim, device=device)
    spec_pos = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], device=device)

    heat_means = []
    for _ in range(6):
        out = bridge.observe(sem_pos=sem_pos, spec_pos=spec_pos)
        heat_means.append(out.heat_mean)

    assert heat_means[-1] > heat_means[0]
