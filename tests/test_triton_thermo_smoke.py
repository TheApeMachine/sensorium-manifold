import pytest
import torch


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not _has_triton(), reason="Triton not available")
def test_triton_thermo_step_is_finite() -> None:
    from sensorium.kernels.triton.manifold_physics import (
        ThermodynamicsDomain,
        ThermodynamicsDomainConfig,
    )

    cfg = ThermodynamicsDomainConfig(grid_size=(16, 16, 16), dt_max=0.01)
    dom = ThermodynamicsDomain(cfg, device="cuda")

    gx, gy, gz = cfg.grid_size
    dx = 1.0 / float(max(gx, gy, gz))
    domain = torch.tensor(
        [gx * dx, gy * dx, gz * dx], device="cuda", dtype=torch.float32
    )

    N = 2048
    positions = torch.rand(N, 3, device="cuda", dtype=torch.float32) * domain
    velocities = torch.randn(N, 3, device="cuda", dtype=torch.float32) * 0.05
    energies = torch.rand(N, device="cuda", dtype=torch.float32) * 0.01
    heats = torch.rand(N, device="cuda", dtype=torch.float32) * 0.01
    excitations = torch.randn(N, device="cuda", dtype=torch.float32) * 0.01
    masses = torch.ones(N, device="cuda", dtype=torch.float32)

    positions, velocities, energies, heats, excitations = dom.step(
        positions, velocities, energies, heats, excitations, masses
    )

    assert positions.shape == (N, 3)
    assert velocities.shape == (N, 3)
    assert energies.shape == (N,)
    assert heats.shape == (N,)
    assert excitations.shape == (N,)

    assert torch.isfinite(positions).all().item()
    assert torch.isfinite(velocities).all().item()
    assert torch.isfinite(energies).all().item()
    assert torch.isfinite(heats).all().item()
    assert torch.isfinite(excitations).all().item()
    assert (heats >= 0.0).all().item()

    assert positions.min().item() >= 0.0
    assert positions.max().item() <= float(domain.max().item())
