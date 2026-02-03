"""Carrier-based energy transport for thermodynamic manifolds.

Carriers are intermediary entities that transport energy between particles.
Particles bond to carriers, not directly to each other.

Key concepts:
- Particles bond to carriers based on attraction (proximity in embedding space)
- Bond strength = attraction strength
- Energy flows: Particle → Carrier → Other Particles bonded to that carrier
- Excitation from received energy generates heat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class CarrierState:
    """Snapshot of carrier state for visualization/debugging."""
    num_carriers: int
    positions: torch.Tensor  # [C, D] carrier positions
    energy: torch.Tensor     # [C] energy held by each carrier
    heat: torch.Tensor       # [C] heat at each carrier


class CarrierPool:
    """Pool of carriers that mediate energy transport between particles.
    
    Carriers exist in the same embedding space as particles.
    Particles bond to nearby carriers based on attraction (distance).
    """
    
    def __init__(
        self,
        num_carriers: int,
        embed_dim: int,
        device: torch.device,
        eps: float = 1e-8,
    ):
        self.num_carriers = int(num_carriers)
        self.embed_dim = int(embed_dim)
        self.device = device
        self.eps = eps
        
        # Carrier positions in embedding space (randomly initialized, will adapt)
        self.position = torch.randn(
            self.num_carriers, self.embed_dim, 
            device=device, dtype=torch.float32
        )
        # Normalize to unit sphere initially
        self.position = self.position / (self.position.norm(dim=1, keepdim=True) + eps)
        
        # Carrier energy (what they're currently transporting)
        self.energy = torch.zeros(self.num_carriers, device=device, dtype=torch.float32)
        
        # Carrier heat (accumulated from transport losses)
        self.heat = torch.zeros(self.num_carriers, device=device, dtype=torch.float32)
    
    def state(self) -> CarrierState:
        """Get current carrier state snapshot."""
        return CarrierState(
            num_carriers=self.num_carriers,
            positions=self.position.detach().clone(),
            energy=self.energy.detach().clone(),
            heat=self.heat.detach().clone(),
        )
    
    def compute_attractions(
        self, 
        particle_positions: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attraction between particles and carriers.
        
        Args:
            particle_positions: [N, D] positions of particles
            top_k: If set, only return top-k carriers per particle
            
        Returns:
            carrier_indices: [N, K] indices of carriers each particle is attracted to
            attractions: [N, K] attraction strengths (higher = closer = stronger bond)
        """
        # [N, D] @ [D, C] -> [N, C] similarities (dot product)
        # Normalize both for cosine similarity
        p_norm = particle_positions / (particle_positions.norm(dim=1, keepdim=True) + self.eps)
        c_norm = self.position / (self.position.norm(dim=1, keepdim=True) + self.eps)
        
        similarities = p_norm @ c_norm.T  # [N, C]
        
        # Convert similarity to attraction (0 to 1 range)
        # similarity is in [-1, 1], shift to [0, 1]
        attractions = (similarities + 1.0) / 2.0
        
        if top_k is None or top_k >= self.num_carriers:
            # Return all carriers
            carrier_indices = torch.arange(
                self.num_carriers, device=self.device
            ).unsqueeze(0).expand(particle_positions.shape[0], -1)
            return carrier_indices, attractions
        
        # Select top-k carriers per particle
        top_attractions, top_indices = torch.topk(attractions, k=top_k, dim=1)
        return top_indices, top_attractions


class ParticleCarrierBonds:
    """Sparse bipartite graph of particle ↔ carrier bonds.
    
    Unlike token→token edges, these are undirected bonds where:
    - Bond strength determines energy flow ratio
    - Energy flows both ways (particle→carrier and carrier→particle)
    """
    
    def __init__(
        self,
        num_particles: int,
        num_carriers: int,
        device: torch.device,
        eps: float = 1e-8,
    ):
        self.num_particles = int(num_particles)
        self.num_carriers = int(num_carriers)
        self.device = device
        self.eps = eps
        
        # Sparse bond storage
        # Each bond: (particle_id, carrier_id, strength, energy_flow)
        self.particle_ids = torch.empty(0, device=device, dtype=torch.long)
        self.carrier_ids = torch.empty(0, device=device, dtype=torch.long)
        self.strengths = torch.empty(0, device=device, dtype=torch.float32)
        self.last_energy_flow = torch.empty(0, device=device, dtype=torch.float32)  # Track energy flow
        
        # For fast lookup: which bonds each particle/carrier has
        self._particle_bond_ptr: Optional[torch.Tensor] = None
        self._carrier_bond_ptr: Optional[torch.Tensor] = None
        self._dirty = True
    
    @property
    def num_bonds(self) -> int:
        return int(self.particle_ids.numel())
    
    def add_bonds(
        self,
        particle_ids: torch.Tensor,
        carrier_ids: torch.Tensor,
        strengths: torch.Tensor,
    ) -> None:
        """Add or reinforce bonds between particles and carriers.
        
        Args:
            particle_ids: [B] particle indices
            carrier_ids: [B] carrier indices
            strengths: [B] bond strengths to add
        """
        particle_ids = particle_ids.to(device=self.device, dtype=torch.long).flatten()
        carrier_ids = carrier_ids.to(device=self.device, dtype=torch.long).flatten()
        strengths = strengths.to(device=self.device, dtype=torch.float32).flatten()
        
        if particle_ids.numel() == 0:
            return
        
        # Create unique bond keys
        key_new = particle_ids * self.num_carriers + carrier_ids
        
        # Coalesce duplicates in new batch
        key_sorted, order = torch.sort(key_new)
        particle_ids = particle_ids[order]
        carrier_ids = carrier_ids[order]
        strengths = strengths[order]
        
        key_unique, inverse = torch.unique_consecutive(key_sorted, return_inverse=True)
        if key_unique.numel() != key_sorted.numel():
            # Sum strengths for duplicates
            strengths_unique = torch.zeros(key_unique.numel(), device=self.device, dtype=torch.float32)
            strengths_unique.index_add_(0, inverse, strengths)
            particle_ids = (key_unique // self.num_carriers).to(torch.long)
            carrier_ids = (key_unique % self.num_carriers).to(torch.long)
            strengths = strengths_unique
        
        if self.num_bonds == 0:
            self.particle_ids = particle_ids
            self.carrier_ids = carrier_ids
            self.strengths = strengths
            self.last_energy_flow = torch.zeros_like(strengths)
            self._dirty = True
            return
        
        # Check for existing bonds
        existing_keys = self.particle_ids * self.num_carriers + self.carrier_ids
        
        # Find which new bonds already exist
        pos = torch.searchsorted(existing_keys, key_unique)
        pos_clamped = pos.clamp(max=existing_keys.numel() - 1)
        exists = (pos < existing_keys.numel()) & (existing_keys[pos_clamped] == key_unique)
        
        # Reinforce existing bonds
        if exists.any():
            self.strengths[pos[exists]] = self.strengths[pos[exists]] + strengths[exists]
        
        # Add new bonds
        new_mask = ~exists
        if new_mask.any():
            self.particle_ids = torch.cat([self.particle_ids, particle_ids[new_mask]])
            self.carrier_ids = torch.cat([self.carrier_ids, carrier_ids[new_mask]])
            self.strengths = torch.cat([self.strengths, strengths[new_mask]])
            self.last_energy_flow = torch.cat([
                self.last_energy_flow, 
                torch.zeros(new_mask.sum(), device=self.device, dtype=torch.float32)
            ])
            
            # Re-sort by key for future lookups
            all_keys = self.particle_ids * self.num_carriers + self.carrier_ids
            order = torch.argsort(all_keys)
            self.particle_ids = self.particle_ids[order]
            self.carrier_ids = self.carrier_ids[order]
            self.strengths = self.strengths[order]
            self.last_energy_flow = self.last_energy_flow[order]
        
        self._dirty = True
    
    def get_particle_bonds(self, particle_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all carriers bonded to a particle and their strengths.
        
        Returns:
            carrier_ids: [K] carrier indices
            strengths: [K] bond strengths
        """
        mask = self.particle_ids == particle_id
        return self.carrier_ids[mask], self.strengths[mask]
    
    def get_carrier_bonds(self, carrier_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all particles bonded to a carrier and their strengths.
        
        Returns:
            particle_ids: [K] particle indices
            strengths: [K] bond strengths
        """
        mask = self.carrier_ids == carrier_id
        return self.particle_ids[mask], self.strengths[mask]
    
    def flow_particle_to_carriers(
        self,
        particle_energy: torch.Tensor,
    ) -> torch.Tensor:
        """Flow energy from particles to carriers based on bond strengths.
        
        Each particle dumps its energy to carriers, split by relative bond strength.
        Also tracks energy flow through each bond for snapping.
        
        Args:
            particle_energy: [P] energy at each particle
            
        Returns:
            carrier_energy_in: [C] energy received by each carrier
        """
        if self.num_bonds == 0:
            return torch.zeros(self.num_carriers, device=self.device, dtype=torch.float32)
        
        # For each particle, compute total outgoing bond strength
        particle_total_strength = torch.zeros(
            self.num_particles, device=self.device, dtype=torch.float32
        )
        particle_total_strength.index_add_(0, self.particle_ids, self.strengths)
        
        # Normalized weight for each bond (strength / total for that particle)
        bond_weights = self.strengths / (particle_total_strength[self.particle_ids] + self.eps)
        
        # Energy flowing through each bond
        bond_energy = particle_energy[self.particle_ids] * bond_weights
        
        # Track energy flow through each bond (for snapping logic)
        self.last_energy_flow = bond_energy.detach().clone()
        
        # Sum into carriers
        carrier_energy_in = torch.zeros(
            self.num_carriers, device=self.device, dtype=torch.float32
        )
        carrier_energy_in.index_add_(0, self.carrier_ids, bond_energy)
        
        return carrier_energy_in
    
    def flow_carriers_to_particles(
        self,
        carrier_energy: torch.Tensor,
        exclude_particles: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flow energy from carriers to particles based on bond strengths.
        
        Each carrier distributes its energy to bonded particles, split by relative bond strength.
        
        Args:
            carrier_energy: [C] energy at each carrier
            exclude_particles: Optional [P] mask of particles to exclude (e.g., source particles)
            
        Returns:
            particle_energy_in: [P] energy received by each particle
        """
        if self.num_bonds == 0:
            return torch.zeros(self.num_particles, device=self.device, dtype=torch.float32)
        
        # Effective strengths (zero out excluded particles)
        effective_strengths = self.strengths.clone()
        if exclude_particles is not None:
            excluded_mask = exclude_particles[self.particle_ids]
            effective_strengths = effective_strengths * (~excluded_mask).float()
        
        # For each carrier, compute total outgoing bond strength
        carrier_total_strength = torch.zeros(
            self.num_carriers, device=self.device, dtype=torch.float32
        )
        carrier_total_strength.index_add_(0, self.carrier_ids, effective_strengths)
        
        # Normalized weight for each bond
        bond_weights = effective_strengths / (carrier_total_strength[self.carrier_ids] + self.eps)
        
        # Energy flowing through each bond
        bond_energy = carrier_energy[self.carrier_ids] * bond_weights
        
        # Sum into particles
        particle_energy_in = torch.zeros(
            self.num_particles, device=self.device, dtype=torch.float32
        )
        particle_energy_in.index_add_(0, self.particle_ids, bond_energy)
        
        return particle_energy_in
    
    def snap_dead_bonds(self, min_energy_flow: float = 0.0) -> int:
        """Remove bonds that have no energy flowing through them.
        
        Bonds snap (break) when they have no energy anymore.
        This is the physically meaningful way bonds die - not by arbitrary decay,
        but because they're no longer carrying energy.
        
        Args:
            min_energy_flow: Minimum energy flow to keep a bond alive.
                             Default 0.0 means any non-zero flow keeps the bond.
                             
        Returns:
            Number of bonds snapped.
        """
        if self.num_bonds == 0:
            return 0
        
        # Keep bonds that had energy flowing through them
        keep = self.last_energy_flow > (min_energy_flow + self.eps)
        num_snapped = (~keep).sum().item()
        
        if num_snapped > 0:
            self.particle_ids = self.particle_ids[keep]
            self.carrier_ids = self.carrier_ids[keep]
            self.strengths = self.strengths[keep]
            self.last_energy_flow = self.last_energy_flow[keep]
            self._dirty = True
        
        return int(num_snapped)
    
    def decay(self, factor: float) -> None:
        """Decay all bond strengths by a factor.
        
        Note: prefer snap_dead_bonds() for physically meaningful bond removal.
        This method is for gradual weakening of bonds over time.
        """
        self.strengths = self.strengths * factor
        
        # Prune very weak bonds
        keep = self.strengths > self.eps
        if not keep.all():
            self.particle_ids = self.particle_ids[keep]
            self.carrier_ids = self.carrier_ids[keep]
            self.strengths = self.strengths[keep]
            self.last_energy_flow = self.last_energy_flow[keep]
            self._dirty = True
