from __future__ import annotations

import torch

from sensorium.distributed import (
    CartesianTopology,
    DistributedManifoldPipeline,
    DistributedStepConfig,
    DistributedThermoConfig,
    LoopbackTransport,
    ShardedWaveConfig,
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_distributed_pipeline_single_rank_smoke() -> None:
    dev = _device()
    topology = CartesianTopology(
        global_grid_size=(16, 16, 16),
        tile_grid_shape=(1, 1, 1),
        halo_thickness=1,
        periodic=True,
    )
    rank_cfg = topology.build_rank_config(0)

    pipeline = DistributedManifoldPipeline(
        rank_config=rank_cfg,
        transport=LoopbackTransport(),
        config=DistributedStepConfig(
            thermo=DistributedThermoConfig(
                grid_spacing=1.0 / 16.0, gravity_jacobi_iters=6
            ),
            wave=ShardedWaveConfig(total_modes=64, omega_min=-4.0, omega_max=4.0),
            dt=0.003,
        ),
        device=dev,
    )

    n = 128
    state = {
        "positions": torch.rand((n, 3), device=dev, dtype=torch.float32),
        "velocities": torch.randn((n, 3), device=dev, dtype=torch.float32) * 0.1,
        "masses": torch.ones((n,), device=dev, dtype=torch.float32) * 0.5,
        "heats": torch.ones((n,), device=dev, dtype=torch.float32),
        "energies": torch.ones((n,), device=dev, dtype=torch.float32) * 0.2,
        "excitations": torch.linspace(-3.0, 3.0, n, device=dev, dtype=torch.float32),
        "phase": torch.zeros((n,), device=dev, dtype=torch.float32),
    }

    out = pipeline.step(state)

    assert out["positions"].shape == (n, 3)
    assert out["velocities"].shape == (n, 3)
    assert out["rho_field"].shape == (16, 16, 16)
    assert out["mom_field"].shape == (16, 16, 16, 3)
    assert out["e_int_field"].shape == (16, 16, 16)
    assert out["psi_real_local"].shape[0] == 64
    assert torch.all(torch.isfinite(out["rho_field"]))
    assert torch.all(out["rho_field"] >= 0.0)
