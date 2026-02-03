"""Unit tests for carrier-based energy transport.

Tests each rule of the carrier physics:
1. A particle is created from an input
2. A particle experiences attraction to one or more carriers
3. A particle will form a bond to every carrier it is attracted to, with strength relative to attraction
4. A particle will dump its energy into carriers, split by bond strength
5. Carriers transport energy to other bonded particles
6. Energy share is divided by bond strength
7. Excitation from energy generates heat
"""

import pytest
import torch

from .carriers import CarrierPool, ParticleCarrierBonds


class TestCarrierPool:
    """Tests for CarrierPool - the pool of carriers that mediate energy transport."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def carrier_pool(self, device):
        """Create a small carrier pool for testing."""
        return CarrierPool(
            num_carriers=4,
            embed_dim=8,
            device=device,
            eps=1e-8,
        )
    
    def test_carrier_pool_initialization(self, carrier_pool):
        """Carriers are initialized with positions, zero energy, zero heat."""
        assert carrier_pool.num_carriers == 4
        assert carrier_pool.embed_dim == 8
        assert carrier_pool.position.shape == (4, 8)
        assert carrier_pool.energy.shape == (4,)
        assert carrier_pool.heat.shape == (4,)
        
        # Initially no energy or heat
        assert carrier_pool.energy.sum().item() == 0.0
        assert carrier_pool.heat.sum().item() == 0.0
        
        # Positions are normalized to unit sphere
        norms = carrier_pool.position.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)
    
    def test_carrier_state_snapshot(self, carrier_pool):
        """State snapshot captures current carrier state."""
        carrier_pool.energy[0] = 5.0
        carrier_pool.heat[1] = 3.0
        
        state = carrier_pool.state()
        
        assert state.num_carriers == 4
        assert state.energy[0].item() == 5.0
        assert state.heat[1].item() == 3.0


class TestCarrierAttraction:
    """Tests for Rule 2: Particles experience attraction to carriers."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def carrier_pool(self, device):
        """Create carrier pool with known positions."""
        pool = CarrierPool(num_carriers=3, embed_dim=4, device=device)
        # Set specific positions for predictable attraction
        pool.position = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Carrier 0: points in +x
            [0.0, 1.0, 0.0, 0.0],  # Carrier 1: points in +y
            [0.0, 0.0, 1.0, 0.0],  # Carrier 2: points in +z
        ], device=device, dtype=torch.float32)
        return pool
    
    def test_particle_attracted_to_nearby_carriers(self, carrier_pool, device):
        """Particles are more attracted to carriers with similar positions."""
        # Particle pointing in +x direction (similar to carrier 0)
        particle_pos = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        
        carrier_indices, attractions = carrier_pool.compute_attractions(particle_pos)
        
        # Should return all 3 carriers
        assert carrier_indices.shape == (1, 3)
        assert attractions.shape == (1, 3)
        
        # Carrier 0 should have highest attraction (same direction)
        # Attractions are based on cosine similarity shifted to [0, 1]
        attractions_list = attractions[0].tolist()
        assert attractions_list[0] > attractions_list[1]  # Carrier 0 > Carrier 1
        assert attractions_list[0] > attractions_list[2]  # Carrier 0 > Carrier 2
    
    def test_top_k_carriers_selection(self, carrier_pool, device):
        """When top_k is specified, only k most attractive carriers are returned."""
        particle_pos = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        
        carrier_indices, attractions = carrier_pool.compute_attractions(particle_pos, top_k=2)
        
        assert carrier_indices.shape == (1, 2)
        assert attractions.shape == (1, 2)
        
        # Carrier 0 should be in the top 2
        assert 0 in carrier_indices[0].tolist()
    
    def test_multiple_particles_attractions(self, carrier_pool, device):
        """Multiple particles can compute attractions simultaneously."""
        particle_pos = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Should prefer carrier 0
            [0.0, 1.0, 0.0, 0.0],  # Should prefer carrier 1
        ], device=device)
        
        carrier_indices, attractions = carrier_pool.compute_attractions(particle_pos, top_k=1)
        
        assert carrier_indices.shape == (2, 1)
        # Each particle should prefer the carrier in its direction
        assert carrier_indices[0, 0].item() == 0  # First particle prefers carrier 0
        assert carrier_indices[1, 0].item() == 1  # Second particle prefers carrier 1


class TestParticleCarrierBonds:
    """Tests for Rules 3-6: Bond formation and energy transport."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def bonds(self, device):
        """Create empty bond graph."""
        return ParticleCarrierBonds(
            num_particles=5,
            num_carriers=3,
            device=device,
        )
    
    def test_bond_creation(self, bonds, device):
        """Rule 3: Particles form bonds to carriers with strength relative to attraction."""
        # Particle 0 bonds to carrier 0 with strength 1.0
        bonds.add_bonds(
            particle_ids=torch.tensor([0], device=device),
            carrier_ids=torch.tensor([0], device=device),
            strengths=torch.tensor([1.0], device=device),
        )
        
        assert bonds.num_bonds == 1
        
        # Check the bond exists
        carrier_ids, strengths = bonds.get_particle_bonds(0)
        assert len(carrier_ids) == 1
        assert carrier_ids[0].item() == 0
        assert strengths[0].item() == 1.0
    
    def test_bond_reinforcement(self, bonds, device):
        """Bonds strengthen when the same particle-carrier pair is observed again."""
        # First bond
        bonds.add_bonds(
            particle_ids=torch.tensor([0], device=device),
            carrier_ids=torch.tensor([0], device=device),
            strengths=torch.tensor([1.0], device=device),
        )
        
        # Same bond again
        bonds.add_bonds(
            particle_ids=torch.tensor([0], device=device),
            carrier_ids=torch.tensor([0], device=device),
            strengths=torch.tensor([0.5], device=device),
        )
        
        # Should still be 1 bond, but stronger
        assert bonds.num_bonds == 1
        _, strengths = bonds.get_particle_bonds(0)
        assert strengths[0].item() == 1.5
    
    def test_multiple_bonds_per_particle(self, bonds, device):
        """Rule 3: Particles can bond to multiple carriers."""
        # Particle 0 bonds to carriers 0, 1, 2
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 0, 0], device=device),
            carrier_ids=torch.tensor([0, 1, 2], device=device),
            strengths=torch.tensor([1.0, 0.5, 0.3], device=device),
        )
        
        assert bonds.num_bonds == 3
        
        carrier_ids, strengths = bonds.get_particle_bonds(0)
        assert len(carrier_ids) == 3
    
    def test_multiple_particles_per_carrier(self, bonds, device):
        """Multiple particles can bond to the same carrier."""
        # Particles 0, 1, 2 all bond to carrier 0
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 1, 2], device=device),
            carrier_ids=torch.tensor([0, 0, 0], device=device),
            strengths=torch.tensor([1.0, 0.5, 0.3], device=device),
        )
        
        particle_ids, strengths = bonds.get_carrier_bonds(0)
        assert len(particle_ids) == 3


class TestEnergyFlowParticleToCarrier:
    """Tests for Rule 4: Energy flows from particles to carriers, split by bond strength."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    def test_energy_split_by_bond_strength(self, device):
        """Rule 4: Energy is split proportionally to bond strength."""
        bonds = ParticleCarrierBonds(num_particles=1, num_carriers=2, device=device)
        
        # Particle 0 has bonds to carrier 0 (strength 2) and carrier 1 (strength 1)
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 0], device=device),
            carrier_ids=torch.tensor([0, 1], device=device),
            strengths=torch.tensor([2.0, 1.0], device=device),
        )
        
        # Particle 0 has 3.0 energy to dump
        particle_energy = torch.tensor([3.0], device=device)
        
        carrier_energy_in = bonds.flow_particle_to_carriers(particle_energy)
        
        # Total strength = 3.0, so:
        # Carrier 0 gets 2/3 * 3.0 = 2.0
        # Carrier 1 gets 1/3 * 3.0 = 1.0
        assert carrier_energy_in.shape == (2,)
        assert torch.isclose(carrier_energy_in[0], torch.tensor(2.0), atol=1e-5)
        assert torch.isclose(carrier_energy_in[1], torch.tensor(1.0), atol=1e-5)
    
    def test_multiple_particles_contribute_to_carrier(self, device):
        """Multiple particles can contribute energy to the same carrier."""
        bonds = ParticleCarrierBonds(num_particles=2, num_carriers=1, device=device)
        
        # Both particles bond to carrier 0
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 1], device=device),
            carrier_ids=torch.tensor([0, 0], device=device),
            strengths=torch.tensor([1.0, 1.0], device=device),
        )
        
        # Particle 0 has 2.0 energy, particle 1 has 3.0 energy
        particle_energy = torch.tensor([2.0, 3.0], device=device)
        
        carrier_energy_in = bonds.flow_particle_to_carriers(particle_energy)
        
        # Carrier 0 receives all energy from both particles
        assert torch.isclose(carrier_energy_in[0], torch.tensor(5.0), atol=1e-5)


class TestEnergyFlowCarrierToParticle:
    """Tests for Rules 5-6: Carriers transport energy to bonded particles."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    def test_carrier_distributes_to_particles(self, device):
        """Rule 5: Carriers distribute energy to bonded particles."""
        bonds = ParticleCarrierBonds(num_particles=2, num_carriers=1, device=device)
        
        # Both particles bonded to carrier 0 with equal strength
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 1], device=device),
            carrier_ids=torch.tensor([0, 0], device=device),
            strengths=torch.tensor([1.0, 1.0], device=device),
        )
        
        carrier_energy = torch.tensor([4.0], device=device)
        
        particle_energy_in = bonds.flow_carriers_to_particles(carrier_energy)
        
        # Each particle gets half (equal bond strength)
        assert particle_energy_in.shape == (2,)
        assert torch.isclose(particle_energy_in[0], torch.tensor(2.0), atol=1e-5)
        assert torch.isclose(particle_energy_in[1], torch.tensor(2.0), atol=1e-5)
    
    def test_energy_split_by_bond_strength_to_particles(self, device):
        """Rule 6: Energy share is divided by bond strength."""
        bonds = ParticleCarrierBonds(num_particles=2, num_carriers=1, device=device)
        
        # Particle 0: strength 3, Particle 1: strength 1
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 1], device=device),
            carrier_ids=torch.tensor([0, 0], device=device),
            strengths=torch.tensor([3.0, 1.0], device=device),
        )
        
        carrier_energy = torch.tensor([4.0], device=device)
        
        particle_energy_in = bonds.flow_carriers_to_particles(carrier_energy)
        
        # Particle 0 gets 3/4 = 3.0, Particle 1 gets 1/4 = 1.0
        assert torch.isclose(particle_energy_in[0], torch.tensor(3.0), atol=1e-5)
        assert torch.isclose(particle_energy_in[1], torch.tensor(1.0), atol=1e-5)
    
    def test_exclude_source_particles(self, device):
        """Source particles can be excluded from receiving their own energy back."""
        bonds = ParticleCarrierBonds(num_particles=3, num_carriers=1, device=device)
        
        # All three particles bonded to carrier 0
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 1, 2], device=device),
            carrier_ids=torch.tensor([0, 0, 0], device=device),
            strengths=torch.tensor([1.0, 1.0, 1.0], device=device),
        )
        
        carrier_energy = torch.tensor([3.0], device=device)
        
        # Exclude particle 0 (the source)
        exclude_mask = torch.tensor([True, False, False], device=device)
        particle_energy_in = bonds.flow_carriers_to_particles(carrier_energy, exclude_particles=exclude_mask)
        
        # Particle 0 gets nothing, particles 1 and 2 split the energy
        assert particle_energy_in[0].item() == 0.0
        assert torch.isclose(particle_energy_in[1], torch.tensor(1.5), atol=1e-5)
        assert torch.isclose(particle_energy_in[2], torch.tensor(1.5), atol=1e-5)


class TestBondSnapping:
    """Tests for bond snapping - bonds break when they have no energy flow."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    def test_bonds_snap_when_no_energy_flow(self, device):
        """Bonds with zero energy flow are snapped (removed)."""
        bonds = ParticleCarrierBonds(num_particles=2, num_carriers=1, device=device)
        
        # Two particles bonded to carrier 0
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 1], device=device),
            carrier_ids=torch.tensor([0, 0], device=device),
            strengths=torch.tensor([1.0, 1.0], device=device),
        )
        
        assert bonds.num_bonds == 2
        
        # Only particle 0 has energy to flow
        particle_energy = torch.tensor([5.0, 0.0], device=device)
        bonds.flow_particle_to_carriers(particle_energy)
        
        # Snap bonds with no energy flow
        num_snapped = bonds.snap_dead_bonds()
        
        # Particle 1's bond should snap (no energy flowed)
        assert num_snapped == 1
        assert bonds.num_bonds == 1
        
        # Only particle 0's bond remains
        carrier_ids, _ = bonds.get_particle_bonds(0)
        assert len(carrier_ids) == 1
        carrier_ids, _ = bonds.get_particle_bonds(1)
        assert len(carrier_ids) == 0
    
    def test_bonds_survive_with_energy_flow(self, device):
        """Bonds with energy flow survive snapping."""
        bonds = ParticleCarrierBonds(num_particles=2, num_carriers=1, device=device)
        
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 1], device=device),
            carrier_ids=torch.tensor([0, 0], device=device),
            strengths=torch.tensor([1.0, 1.0], device=device),
        )
        
        # Both particles have energy
        particle_energy = torch.tensor([3.0, 2.0], device=device)
        bonds.flow_particle_to_carriers(particle_energy)
        
        # No bonds should snap
        num_snapped = bonds.snap_dead_bonds()
        
        assert num_snapped == 0
        assert bonds.num_bonds == 2
    
    def test_min_energy_threshold(self, device):
        """Bonds below minimum energy threshold are snapped."""
        bonds = ParticleCarrierBonds(num_particles=2, num_carriers=1, device=device)
        
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 1], device=device),
            carrier_ids=torch.tensor([0, 0], device=device),
            strengths=torch.tensor([1.0, 1.0], device=device),
        )
        
        # Particle 0: high energy, Particle 1: very low energy
        particle_energy = torch.tensor([10.0, 0.001], device=device)
        bonds.flow_particle_to_carriers(particle_energy)
        
        # Snap bonds below threshold 0.01
        num_snapped = bonds.snap_dead_bonds(min_energy_flow=0.01)
        
        # Particle 1's bond should snap (energy flow 0.0005 < 0.01)
        assert num_snapped == 1
        assert bonds.num_bonds == 1


class TestBondDecay:
    """Tests for gradual bond decay."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    def test_bond_decay(self, device):
        """Bonds decay by a factor, and very weak bonds are pruned."""
        bonds = ParticleCarrierBonds(num_particles=2, num_carriers=1, device=device, eps=1e-6)
        
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 1], device=device),
            carrier_ids=torch.tensor([0, 0], device=device),
            strengths=torch.tensor([1.0, 1e-7], device=device),  # Very weak bond
        )
        
        assert bonds.num_bonds == 2
        
        # Decay by 50%
        bonds.decay(0.5)
        
        # Strong bond survives (0.5), weak bond is pruned (0.5e-7 < 1e-6)
        assert bonds.num_bonds == 1
        _, strengths = bonds.get_particle_bonds(0)
        assert torch.isclose(strengths[0], torch.tensor(0.5), atol=1e-5)


class TestEndToEndEnergyTransport:
    """Integration test: full energy transport cycle."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    def test_energy_conserved_in_transport(self, device):
        """Energy is conserved when flowing through carriers (no loss in transport)."""
        bonds = ParticleCarrierBonds(num_particles=3, num_carriers=2, device=device)
        
        # Particle 0 bonds to both carriers
        # Particles 1, 2 bond to carrier 0 and 1 respectively
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 0, 1, 2], device=device),
            carrier_ids=torch.tensor([0, 1, 0, 1], device=device),
            strengths=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
        )
        
        # Particle 0 dumps 10.0 energy
        particle_energy = torch.tensor([10.0, 0.0, 0.0], device=device)
        
        # Step 1: Flow to carriers
        carrier_energy = bonds.flow_particle_to_carriers(particle_energy)
        total_carrier = carrier_energy.sum().item()
        
        # Step 2: Flow from carriers to other particles (exclude particle 0)
        exclude = torch.tensor([True, False, False], device=device)
        received = bonds.flow_carriers_to_particles(carrier_energy, exclude_particles=exclude)
        total_received = received.sum().item()
        
        # Energy should be conserved: what carriers received = what particles receive
        assert torch.isclose(torch.tensor(total_carrier), torch.tensor(total_received), atol=1e-5)
        # And it should equal what particle 0 originally had
        assert torch.isclose(torch.tensor(total_received), torch.tensor(10.0), atol=1e-5)


class TestPreciseEnergyDistribution:
    """Tests for exact energy distribution by bond strength ratios."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    def test_one_particle_three_carriers_exact_distribution(self, device):
        """
        Rule validation: 1 particle with 3 bonds at 10%, 50%, 40% strength.
        Energy introduced to particle must distribute exactly by these ratios.
        Total carrier energy must equal 100% of input energy.
        """
        bonds = ParticleCarrierBonds(num_particles=1, num_carriers=3, device=device)
        
        # Particle 0 bonds to 3 carriers with relative strengths 10%, 50%, 40%
        # Using strengths 0.1, 0.5, 0.4 (they will be normalized)
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 0, 0], device=device),
            carrier_ids=torch.tensor([0, 1, 2], device=device),
            strengths=torch.tensor([0.1, 0.5, 0.4], device=device),
        )
        
        # Introduce exactly 100.0 energy to the particle
        input_energy = 100.0
        particle_energy = torch.tensor([input_energy], device=device)
        
        # Flow energy to carriers
        carrier_energy = bonds.flow_particle_to_carriers(particle_energy)
        
        # Verify exact distribution by ratio
        # Carrier 0: 10% of 100 = 10.0
        # Carrier 1: 50% of 100 = 50.0
        # Carrier 2: 40% of 100 = 40.0
        assert torch.isclose(carrier_energy[0], torch.tensor(10.0), atol=1e-5), \
            f"Carrier 0 expected 10.0, got {carrier_energy[0].item()}"
        assert torch.isclose(carrier_energy[1], torch.tensor(50.0), atol=1e-5), \
            f"Carrier 1 expected 50.0, got {carrier_energy[1].item()}"
        assert torch.isclose(carrier_energy[2], torch.tensor(40.0), atol=1e-5), \
            f"Carrier 2 expected 40.0, got {carrier_energy[2].item()}"
        
        # Verify total equals exactly 100% of input
        total_carrier_energy = carrier_energy.sum().item()
        assert torch.isclose(
            torch.tensor(total_carrier_energy), 
            torch.tensor(input_energy), 
            atol=1e-5
        ), f"Total carrier energy {total_carrier_energy} != input {input_energy}"
    
    def test_unnormalized_strengths_still_distribute_correctly(self, device):
        """
        Bond strengths don't need to sum to 1.0 - they're normalized internally.
        Strengths 1, 5, 4 should produce same 10%, 50%, 40% distribution.
        """
        bonds = ParticleCarrierBonds(num_particles=1, num_carriers=3, device=device)
        
        # Use unnormalized strengths that have same ratios
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 0, 0], device=device),
            carrier_ids=torch.tensor([0, 1, 2], device=device),
            strengths=torch.tensor([1.0, 5.0, 4.0], device=device),  # Sum = 10, not 1
        )
        
        input_energy = 100.0
        particle_energy = torch.tensor([input_energy], device=device)
        carrier_energy = bonds.flow_particle_to_carriers(particle_energy)
        
        # Same expected distribution
        assert torch.isclose(carrier_energy[0], torch.tensor(10.0), atol=1e-5)
        assert torch.isclose(carrier_energy[1], torch.tensor(50.0), atol=1e-5)
        assert torch.isclose(carrier_energy[2], torch.tensor(40.0), atol=1e-5)
        assert torch.isclose(carrier_energy.sum(), torch.tensor(input_energy), atol=1e-5)
    
    def test_arbitrary_energy_amount(self, device):
        """Distribution ratios work for any energy amount, not just 100."""
        bonds = ParticleCarrierBonds(num_particles=1, num_carriers=3, device=device)
        
        bonds.add_bonds(
            particle_ids=torch.tensor([0, 0, 0], device=device),
            carrier_ids=torch.tensor([0, 1, 2], device=device),
            strengths=torch.tensor([0.1, 0.5, 0.4], device=device),
        )
        
        # Use arbitrary energy amount
        input_energy = 73.5
        particle_energy = torch.tensor([input_energy], device=device)
        carrier_energy = bonds.flow_particle_to_carriers(particle_energy)
        
        # Verify ratios
        expected_0 = input_energy * 0.1  # 7.35
        expected_1 = input_energy * 0.5  # 36.75
        expected_2 = input_energy * 0.4  # 29.4
        
        assert torch.isclose(carrier_energy[0], torch.tensor(expected_0), atol=1e-5)
        assert torch.isclose(carrier_energy[1], torch.tensor(expected_1), atol=1e-5)
        assert torch.isclose(carrier_energy[2], torch.tensor(expected_2), atol=1e-5)
        
        # Total must equal input exactly
        assert torch.isclose(carrier_energy.sum(), torch.tensor(input_energy), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
