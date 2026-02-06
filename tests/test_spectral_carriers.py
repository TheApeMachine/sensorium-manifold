"""Unit tests for the coherence field layer.

The coherence layer is a fixed ω-lattice of a complex field Ψ(ω) evolved by a
(dissipative) Gross–Pitaevskii-style update. There is no conflict-driven carrier
spawning/splitting in the single-model codebase.

Run with:
    pytest tests/test_spectral_carriers.py -v
"""

from __future__ import annotations

import pytest
import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture
def device() -> str:
    return get_device()


@pytest.fixture
def spectral_physics(device):
    """Create a CoherenceDomain instance for testing."""

    class _WrappedCoherence:
        def __init__(self, impl, *, device: str):
            self._impl = impl
            self._device = device

        def __getattr__(self, name):  # pragma: no cover
            return getattr(self._impl, name)

        def step(self, osc_phase, osc_omega, osc_energy):
            N = int(osc_phase.shape[0])
            if N == 0:
                pos = torch.empty((0, 3), device=self._device, dtype=torch.float32)
            else:
                x = torch.linspace(0.25, 7.75, N, device=self._device, dtype=torch.float32)
                pos = torch.stack([x, torch.zeros_like(x) + 0.5, torch.zeros_like(x) + 0.5], dim=1)

            return self._impl.step(
                osc_phase,
                osc_omega,
                osc_energy,
                particle_positions=pos,
            )

    if device == "mps":
        from sensorium.kernels.metal.manifold_physics import CoherenceFieldConfig, CoherenceDomain

        cfg = CoherenceFieldConfig(
            omega_bins=64,
            omega_min=-4.0,
            omega_max=4.0,
            hbar_eff=1.0,
            mass_eff=1.0,
            g_interaction=-0.25,
            energy_decay=0.01,
        )
        impl = CoherenceDomain(cfg, grid_size=(8, 8, 8), dt=0.01, device=device)
        return _WrappedCoherence(impl, device=device)

    pytest.skip("No GPU available for coherence field tests")


class TestCoherenceLattice:
    def test_fixed_lattice_has_expected_size(self, spectral_physics, device):
        N = 128
        osc_phase = torch.rand(N, device=device, dtype=torch.float32) * (2.0 * torch.pi)
        osc_omega = torch.randn(N, device=device, dtype=torch.float32)
        osc_energy = torch.rand(N, device=device, dtype=torch.float32) + 0.1

        assert spectral_physics.num_carriers == spectral_physics.max_carriers
        out = spectral_physics.step(osc_phase, osc_omega, osc_energy)

        assert spectral_physics.num_carriers == spectral_physics.max_carriers
        assert out["frequencies"].numel() == spectral_physics.max_carriers
        assert out["amplitudes"].numel() == spectral_physics.max_carriers

    def test_step_outputs_are_finite(self, spectral_physics, device):
        N = 256
        osc_phase = torch.rand(N, device=device, dtype=torch.float32) * (2.0 * torch.pi)
        osc_omega = torch.randn(N, device=device, dtype=torch.float32)
        osc_energy = torch.rand(N, device=device, dtype=torch.float32) + 0.1

        for _ in range(5):
            out = spectral_physics.step(osc_phase, osc_omega, osc_energy)
            assert torch.isfinite(out["amplitudes"]).all().item()
            assert torch.isfinite(out["phases"]).all().item()

    def test_no_spawning_or_splitting(self, spectral_physics, device):
        N = 256
        osc_phase = torch.rand(N, device=device, dtype=torch.float32) * (2.0 * torch.pi)
        osc_omega = torch.randn(N, device=device, dtype=torch.float32)
        osc_energy = torch.rand(N, device=device, dtype=torch.float32) + 0.1

        spectral_physics.step(osc_phase, osc_omega, osc_energy)
        n0 = int(spectral_physics.num_carriers)
        for _ in range(10):
            spectral_physics.step(osc_phase, osc_omega, osc_energy)
            assert int(spectral_physics.num_carriers) == n0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

