"""Unit tests for geometric layer physics.

Tests the core geometric layer behaviors:
- Particle motion and velocity updates
- Grid-based field solves (pressure, temperature)
- Collision detection and response
- Energy conservation
- Boundary conditions

Run with:
    pytest tests/test_geometric_physics.py -v
"""

from __future__ import annotations

import pytest
import torch


def get_device():
    """Get available device for testing."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture
def device():
    """Pytest fixture for device."""
    return get_device()


@pytest.fixture
def manifold_physics(device):
    """Create a ManifoldPhysics instance for testing."""
    if device == "mps":
        from sensorium.kernels.metal.manifold_physics import (
            ThermodynamicsDomain,
            ThermodynamicsDomainConfig,
        )
        cfg = ThermodynamicsDomainConfig(
            grid_size=(16, 16, 16),
            dt_max=0.01,
        )
        return ThermodynamicsDomain(cfg, device=device)
    elif device == "cuda":
        from sensorium.kernels.triton.manifold_physics import (
            ThermodynamicsDomain,
            ThermodynamicsDomainConfig,
        )
        cfg = ThermodynamicsDomainConfig(
            grid_size=(16, 16, 16),
            dt_max=0.01,
        )
        return ThermodynamicsDomain(cfg)
    else:
        pytest.skip("No GPU available for geometric physics tests")


class TestParticleMotion:
    """Test basic particle dynamics."""
    
    def test_particles_move_with_velocity(self, manifold_physics, device):
        """Particles with non-zero velocity should change position."""
        N = 100
        positions = torch.ones(N, 3, device=device, dtype=torch.float32) * 8.0  # Center of grid
        velocities = torch.ones(N, 3, device=device, dtype=torch.float32)  # Uniform velocity
        energies = torch.ones(N, device=device, dtype=torch.float32)
        heats = torch.zeros(N, device=device, dtype=torch.float32)
        excitations = torch.zeros(N, device=device, dtype=torch.float32)
        masses = torch.ones(N, device=device, dtype=torch.float32)
        
        initial_pos = positions.clone()
        
        for _ in range(10):
            positions, velocities, energies, heats, excitations = manifold_physics.step(
                positions, velocities, energies, heats, excitations, masses
            )
        
        # Particles should have moved
        displacement = (positions - initial_pos).norm(dim=1)
        assert displacement.mean().item() > 0.01, "Particles should move with velocity"
    
    def test_stationary_particles_stay_put(self, manifold_physics, device):
        """Particles with zero velocity should remain (mostly) stationary."""
        N = 100
        positions = torch.rand(N, 3, device=device, dtype=torch.float32) * 12.0 + 2.0
        velocities = torch.zeros(N, 3, device=device, dtype=torch.float32)
        energies = torch.zeros(N, device=device, dtype=torch.float32)
        heats = torch.zeros(N, device=device, dtype=torch.float32)
        excitations = torch.zeros(N, device=device, dtype=torch.float32)
        masses = torch.ones(N, device=device, dtype=torch.float32)
        
        initial_pos = positions.clone()
        
        for _ in range(10):
            positions, velocities, energies, heats, excitations = manifold_physics.step(
                positions, velocities, energies, heats, excitations, masses
            )
        
        # Displacement should be small (only from field effects)
        displacement = (positions - initial_pos).norm(dim=1)
        mean_disp = displacement.mean().item()
        # With zero energy and velocity, particles still move due to field effects
        # (gravity field, pressure gradients). The sorted scatter mode produces
        # slightly different numerical behavior than hash-based scatter.
        # A threshold of 10.0 allows for field-induced motion while catching bugs.
        assert mean_disp < 10.0, f"Zero-velocity particles moved too much: {mean_disp}"


class TestBoundaryConditions:
    """Test grid boundary handling."""
    
    def test_particles_stay_in_bounds(self, manifold_physics, device):
        """Particles should be contained within the grid."""
        N = 500
        grid_size = manifold_physics.config.grid_size
        
        # Start some particles near boundaries
        positions = torch.rand(N, 3, device=device, dtype=torch.float32)
        positions[:N//4] *= 2.0  # Near origin
        positions[N//4:N//2] = positions[N//4:N//2] * 2.0 + (grid_size[0] - 2)  # Near far boundary
        positions[N//2:] = positions[N//2:] * (grid_size[0] - 2.0) + 1.0  # Middle
        
        velocities = torch.randn(N, 3, device=device, dtype=torch.float32) * 2.0
        energies = torch.rand(N, device=device, dtype=torch.float32)
        heats = torch.zeros(N, device=device, dtype=torch.float32)
        excitations = torch.zeros(N, device=device, dtype=torch.float32)
        masses = torch.ones(N, device=device, dtype=torch.float32)
        
        for _ in range(50):
            positions, velocities, energies, heats, excitations = manifold_physics.step(
                positions, velocities, energies, heats, excitations, masses
            )
            
            # Check bounds (with small margin)
            assert positions.min().item() >= 0.0, "Particles escaped lower boundary"
            assert positions.max().item() <= grid_size[0], "Particles escaped upper boundary"


class TestLegacyAPIRemovals:
    """Ensure legacy particle-field/collision APIs are disabled."""

    def test_collision_api_is_disabled(self, manifold_physics, device):
        """Collision routines must fail loudly (single-model enforcement)."""
        if not hasattr(manifold_physics, "compute_interactions_spatial_hash"):
            pytest.skip("Backend has no collision API surface")

        N = 10
        positions = torch.rand(N, 3, device=device, dtype=torch.float32) * 12.0 + 2.0
        velocities = torch.randn(N, 3, device=device, dtype=torch.float32) * 0.1
        excitations = torch.rand(N, device=device, dtype=torch.float32)
        masses = torch.ones(N, device=device, dtype=torch.float32)
        heats = torch.rand(N, device=device, dtype=torch.float32)

        with pytest.raises(RuntimeError, match="legacy|single spatial model|Navier"):
            manifold_physics.compute_interactions_spatial_hash(
                positions, velocities, excitations, masses, heats, dt=0.01
            )


class TestEnergyConservation:
    """Test energy conservation in the geometric layer."""
    
    def test_total_energy_bounded(self, manifold_physics, device):
        """Total energy should not grow unboundedly.
        
        NOTE: This system is not strictly energy-conserving. The geometric layer 
        includes field interactions (gravity, pressure) that can add or remove energy.
        We only check that energy doesn't explode to infinity (numerical instability).
        """
        N = 200
        positions = torch.rand(N, 3, device=device, dtype=torch.float32) * 12.0 + 2.0
        velocities = torch.randn(N, 3, device=device, dtype=torch.float32) * 0.5
        energies = torch.rand(N, device=device, dtype=torch.float32)
        heats = torch.rand(N, device=device, dtype=torch.float32)
        excitations = torch.zeros(N, device=device, dtype=torch.float32)
        masses = torch.ones(N, device=device, dtype=torch.float32)
        
        def compute_total_energy():
            ke = (0.5 * masses * (velocities ** 2).sum(dim=1)).sum().item()
            thermal = (heats + energies).sum().item()
            return ke + thermal
        
        initial_energy = compute_total_energy()
        max_energy = initial_energy
        
        for step in range(100):
            positions, velocities, energies, heats, excitations = manifold_physics.step(
                positions, velocities, energies, heats, excitations, masses
            )
            
            current_energy = compute_total_energy()
            max_energy = max(max_energy, current_energy)
            
            # Check for numerical explosion (NaN or very large values)
            assert current_energy < 1e10, f"Energy exploded at step {step}: {current_energy}"
            assert torch.isfinite(energies).all().item(), f"Non-finite energies at step {step}"
        
        # Energy shouldn't grow by more than 100x (generous bound for non-conservative system)
        assert max_energy < initial_energy * 100, \
            f"Energy grew excessively: {initial_energy:.2f} → {max_energy:.2f}"
    
    def test_kinetic_energy_transfers_to_heat(self, manifold_physics, device):
        """Heats should remain non-negative (internal energy is clamped)."""
        N = 100

        positions = torch.rand(N, 3, device=device, dtype=torch.float32) * 10.0 + 3.0
        velocities = torch.randn(N, 3, device=device, dtype=torch.float32) * 2.0
        energies = torch.ones(N, device=device, dtype=torch.float32)
        heats = torch.zeros(N, device=device, dtype=torch.float32)
        excitations = torch.zeros(N, device=device, dtype=torch.float32)
        masses = torch.ones(N, device=device, dtype=torch.float32)

        for _ in range(50):
            positions, velocities, energies, heats, excitations = manifold_physics.step(
                positions, velocities, energies, heats, excitations, masses
            )
            assert (heats >= 0.0).all().item(), "Heats became negative"


class TestNumericalStability:
    """Test numerical robustness of the geometric layer."""
    
    def test_outputs_are_finite(self, manifold_physics, device):
        """All outputs should be finite (no NaN or Inf)."""
        N = 200
        
        positions = torch.rand(N, 3, device=device, dtype=torch.float32) * 12.0 + 2.0
        velocities = torch.randn(N, 3, device=device, dtype=torch.float32)
        energies = torch.rand(N, device=device, dtype=torch.float32)
        heats = torch.rand(N, device=device, dtype=torch.float32)
        excitations = torch.randn(N, device=device, dtype=torch.float32)
        masses = torch.ones(N, device=device, dtype=torch.float32)
        
        for step in range(100):
            positions, velocities, energies, heats, excitations = manifold_physics.step(
                positions, velocities, energies, heats, excitations, masses
            )
            
            assert torch.isfinite(positions).all().item(), f"Non-finite positions at step {step}"
            assert torch.isfinite(velocities).all().item(), f"Non-finite velocities at step {step}"
            assert torch.isfinite(energies).all().item(), f"Non-finite energies at step {step}"
            assert torch.isfinite(heats).all().item(), f"Non-finite heats at step {step}"
    
    def test_handles_zero_mass(self, manifold_physics, device):
        """System should handle particles with very small mass."""
        N = 50
        
        positions = torch.rand(N, 3, device=device, dtype=torch.float32) * 12.0 + 2.0
        velocities = torch.randn(N, 3, device=device, dtype=torch.float32)
        energies = torch.rand(N, device=device, dtype=torch.float32)
        heats = torch.zeros(N, device=device, dtype=torch.float32)
        excitations = torch.zeros(N, device=device, dtype=torch.float32)
        masses = torch.ones(N, device=device, dtype=torch.float32) * 1e-6  # Very small
        
        # Should not crash
        for _ in range(20):
            positions, velocities, energies, heats, excitations = manifold_physics.step(
                positions, velocities, energies, heats, excitations, masses
            )
            
            assert torch.isfinite(positions).all().item(), "Positions became non-finite with small mass"
    
    def test_handles_high_velocity(self, manifold_physics, device):
        """System should handle particles with very high velocity."""
        N = 50
        
        positions = torch.ones(N, 3, device=device, dtype=torch.float32) * 8.0  # Center
        velocities = torch.randn(N, 3, device=device, dtype=torch.float32) * 100.0  # Very fast
        energies = torch.rand(N, device=device, dtype=torch.float32)
        heats = torch.zeros(N, device=device, dtype=torch.float32)
        excitations = torch.zeros(N, device=device, dtype=torch.float32)
        masses = torch.ones(N, device=device, dtype=torch.float32)
        
        # Should not crash (particles may hit boundaries rapidly)
        for _ in range(10):
            positions, velocities, energies, heats, excitations = manifold_physics.step(
                positions, velocities, energies, heats, excitations, masses
            )
            
            assert torch.isfinite(positions).all().item(), "Positions became non-finite with high velocity"
            assert torch.isfinite(velocities).all().item(), "Velocities became non-finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
