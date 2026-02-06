"""Dark particle injector.

Injects dark particles into the simulation for inference queries.
Dark particles:
- Perturb the simulation state but do not couple to modes
- Are invisible to regular observers
- Decay naturally (energy runs out) and are removed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Callable, Any

import torch
import numpy as np

from sensorium.observers.types import PARTICLE_FLAG_DARK


@dataclass
class DarkParticleConfig:
    """Configuration for dark particle injection.
    
    Attributes:
        initial_energy: Starting energy for dark particles (affects lifespan)
        energy_decay_rate: Rate at which dark particles lose energy per step
        removal_threshold: Energy below which dark particles are removed
        coupling_block: If True, dark particles cannot couple to modes
    """
    initial_energy: float = 0.5
    energy_decay_rate: float = 0.02
    removal_threshold: float = 0.01
    coupling_block: bool = True


class DarkParticleInjector:
    """Injects dark particles into the simulation for inference.
    
    Dark particles are marked with PARTICLE_FLAG_DARK and are:
    - Invisible to observers (filtered out by default)
    - Unable to couple to modes (enforced in physics)
    - Subject to natural decay (removed when energy depletes)
    
    Example:
        injector = DarkParticleInjector(config)
        
        # Inject from a dataset
        dark_indices = injector.inject(state, dataset.generate())
        
        # Or inject raw bytes
        dark_indices = injector.inject_bytes(state, query_bytes)
    """
    
    def __init__(self, config: DarkParticleConfig | None = None):
        """Initialize the dark particle injector.
        
        Args:
            config: Configuration for dark particle behavior
        """
        self.config = config or DarkParticleConfig()
        self._injected_indices: list[int] = []
    
    @property
    def injected_indices(self) -> list[int]:
        """Get the indices of currently injected dark particles."""
        return self._injected_indices.copy()
    
    @property
    def n_dark(self) -> int:
        """Get the number of currently tracked dark particles."""
        return len(self._injected_indices)
    
    def inject(
        self,
        state: dict,
        data_source: Iterator[tuple[int, int]] | list[tuple[int, int]],
        device: str = "mps",
        dtype: torch.dtype = torch.float32,
    ) -> list[int]:
        """Inject dark particles from a data source.
        
        Args:
            state: Mutable simulation state dict (will be modified in-place)
            data_source: Iterator yielding (byte_value, sequence_index) tuples
            device: Device for new tensors
            dtype: Data type for new tensors
        
        Returns:
            List of indices where dark particles were added
        """
        # Consume iterator to list
        data_list = list(data_source)
        if not data_list:
            return []
        
        n_new = len(data_list)
        byte_values = [d[0] for d in data_list]
        seq_indices = [d[1] for d in data_list]
        
        # Create dark particle tensors
        new_positions = self._generate_positions(n_new, state, device, dtype)
        new_velocities = torch.zeros(n_new, 3, device=device, dtype=dtype)
        new_energies = torch.full((n_new,), self.config.initial_energy, device=device, dtype=dtype)
        new_heats = torch.zeros(n_new, device=device, dtype=dtype)
        
        # Convert bytes to frequencies (normalized to [0, 1])
        new_excitations = torch.tensor(
            [b / 255.0 for b in byte_values],
            device=device,
            dtype=dtype,
        )
        
        new_masses = new_energies.clone()
        new_phases = torch.rand(n_new, device=device, dtype=dtype) * 2.0 * np.pi
        
        # Create flags tensor (all marked as dark)
        new_flags = torch.full((n_new,), PARTICLE_FLAG_DARK, device=device, dtype=torch.int32)
        
        # Create token IDs from bytes
        new_token_ids = torch.tensor(byte_values, device=device, dtype=torch.int64)
        new_byte_values = torch.tensor(byte_values, device=device, dtype=torch.int64)
        new_sequence_indices = torch.tensor(seq_indices, device=device, dtype=torch.int64)
        # Query particles are not part of training samples.
        new_sample_indices = torch.full((n_new,), -1, device=device, dtype=torch.int64)
        
        # Get current particle count
        n_existing = len(state.get("positions", torch.empty(0, 3)))
        self._ensure_existing_particle_fields(
            state=state,
            n_existing=n_existing,
            device=device,
            dtype=dtype,
        )
        
        # Append to state
        self._append_to_state(
            state,
            positions=new_positions,
            velocities=new_velocities,
            energies=new_energies,
            heats=new_heats,
            excitations=new_excitations,
            masses=new_masses,
            osc_phase=new_phases,
            particle_flags=new_flags,
            token_ids=new_token_ids,
            byte_values=new_byte_values,
            sequence_indices=new_sequence_indices,
            sample_indices=new_sample_indices,
        )
        
        # Track indices
        new_indices = list(range(n_existing, n_existing + n_new))
        self._injected_indices.extend(new_indices)
        
        return new_indices
    
    def inject_bytes(
        self,
        state: dict,
        byte_sequence: bytes | list[int],
        device: str = "mps",
        dtype: torch.dtype = torch.float32,
    ) -> list[int]:
        """Inject dark particles from raw bytes.
        
        Args:
            state: Mutable simulation state dict
            byte_sequence: Raw bytes to inject
            device: Device for new tensors
            dtype: Data type for new tensors
        
        Returns:
            List of indices where dark particles were added
        """
        data_source = [(b, i) for i, b in enumerate(byte_sequence)]
        return self.inject(state, data_source, device, dtype)
    
    def decay_and_remove(
        self,
        state: dict,
    ) -> list[int]:
        """Decay dark particle energy and remove depleted ones.
        
        Args:
            state: Mutable simulation state dict
        
        Returns:
            List of indices that were removed
        """
        if not self._injected_indices:
            return []
        
        particle_flags = state.get("particle_flags")
        if particle_flags is None:
            return []
        
        energies = state.get("energies")
        if energies is None:
            return []
        
        # Find dark particles
        flags_np = particle_flags.cpu().numpy()
        dark_mask = (flags_np & PARTICLE_FLAG_DARK) != 0
        
        if not dark_mask.any():
            self._injected_indices = []
            return []
        
        # Decay energy for dark particles
        energies_np = energies.cpu().numpy()
        energies_np[dark_mask] *= (1.0 - self.config.energy_decay_rate)
        
        # Find depleted dark particles
        depleted_mask = dark_mask & (energies_np < self.config.removal_threshold)
        depleted_indices = depleted_mask.nonzero()[0].tolist()
        
        if depleted_indices:
            # Remove depleted particles
            keep_mask = ~depleted_mask
            self._remove_particles(state, keep_mask)
            
            # Update tracked indices (shift due to removal)
            self._injected_indices = [
                idx for idx in self._injected_indices
                if idx not in depleted_indices
            ]
            # Recompute indices after removal
            self._update_indices_after_removal(depleted_indices)
        else:
            # Update energies in-place
            state["energies"] = torch.tensor(
                energies_np,
                device=energies.device,
                dtype=energies.dtype,
            )
        
        return depleted_indices
    
    def clear(self, state: dict) -> int:
        """Remove all dark particles immediately.
        
        Args:
            state: Mutable simulation state dict
        
        Returns:
            Number of dark particles removed
        """
        if not self._injected_indices:
            return 0
        
        particle_flags = state.get("particle_flags")
        if particle_flags is None:
            self._injected_indices = []
            return 0
        
        flags_np = particle_flags.cpu().numpy()
        dark_mask = (flags_np & PARTICLE_FLAG_DARK) != 0
        n_removed = int(dark_mask.sum())
        
        if n_removed > 0:
            keep_mask = ~dark_mask
            self._remove_particles(state, keep_mask)
        
        self._injected_indices = []
        return n_removed
    
    def _generate_positions(
        self,
        n: int,
        state: dict,
        device: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate positions for new dark particles.
        
        Places dark particles in a region where they can interact
        with existing particles but are spread out.
        """
        # Get existing positions to compute bounds
        existing_pos = state.get("positions")
        
        if existing_pos is not None and len(existing_pos) > 0:
            # Place near existing particles
            pos_np = existing_pos.cpu().numpy()
            center = pos_np.mean(axis=0)
            spread = pos_np.std(axis=0) + 0.1
            
            positions = torch.randn(n, 3, device=device, dtype=dtype)
            positions = positions * torch.tensor(spread, device=device, dtype=dtype)
            positions = positions + torch.tensor(center, device=device, dtype=dtype)
        else:
            # Default positioning
            positions = torch.rand(n, 3, device=device, dtype=dtype) * 10.0
        
        return positions
    
    def _append_to_state(self, state: dict, **tensors):
        """Append new tensors to existing state tensors."""
        for key, new_tensor in tensors.items():
            existing = state.get(key)
            if existing is not None:
                state[key] = torch.cat([existing, new_tensor], dim=0)
            else:
                state[key] = new_tensor

    def _ensure_existing_particle_fields(
        self,
        state: dict,
        n_existing: int,
        device: str,
        dtype: torch.dtype,
    ):
        """Ensure particle-indexed state tensors exist for current particles.

        This prevents shape mismatches when appending/removing dark particles in
        states that do not yet carry all particle metadata fields.
        """
        if n_existing <= 0:
            return

        def ensure(name: str, tensor: torch.Tensor):
            if state.get(name) is None:
                state[name] = tensor

        ensure("positions", torch.zeros((n_existing, 3), device=device, dtype=dtype))
        ensure("velocities", torch.zeros((n_existing, 3), device=device, dtype=dtype))
        ensure("energies", torch.zeros((n_existing,), device=device, dtype=dtype))
        ensure("heats", torch.zeros((n_existing,), device=device, dtype=dtype))
        ensure("excitations", torch.zeros((n_existing,), device=device, dtype=dtype))
        ensure("masses", torch.zeros((n_existing,), device=device, dtype=dtype))
        ensure("osc_phase", torch.zeros((n_existing,), device=device, dtype=dtype))
        ensure("particle_flags", torch.zeros((n_existing,), device=device, dtype=torch.int32))
        ensure("token_ids", torch.zeros((n_existing,), device=device, dtype=torch.int64))
        ensure("byte_values", torch.zeros((n_existing,), device=device, dtype=torch.int64))
        ensure(
            "sequence_indices",
            torch.arange(n_existing, device=device, dtype=torch.int64),
        )
        ensure("sample_indices", torch.zeros((n_existing,), device=device, dtype=torch.int64))
    
    def _remove_particles(self, state: dict, keep_mask: np.ndarray):
        """Remove particles based on a keep mask."""
        keep_mask_torch = torch.tensor(keep_mask, device="cpu")
        
        for key in ["positions", "velocities", "energies", "heats", 
                    "excitations", "masses", "osc_phase", "particle_flags", "token_ids",
                    "byte_values", "sequence_indices", "sample_indices"]:
            tensor = state.get(key)
            if tensor is not None:
                device = tensor.device
                dtype = tensor.dtype
                # Move to CPU for masking
                tensor_cpu = tensor.cpu()
                filtered = tensor_cpu[keep_mask_torch.to(tensor_cpu.device)]
                state[key] = filtered.to(device=device, dtype=dtype if dtype.is_floating_point else dtype)
    
    def _update_indices_after_removal(self, removed_indices: list[int]):
        """Update tracked indices after particle removal."""
        removed_set = set(removed_indices)
        
        new_indices = []
        for idx in self._injected_indices:
            if idx in removed_set:
                continue
            # Count how many removed indices are below this one
            shift = sum(1 for r in removed_indices if r < idx)
            new_indices.append(idx - shift)
        
        self._injected_indices = new_indices


def get_dark_particle_mask(
    particle_flags: torch.Tensor | np.ndarray | None,
    n: int,
) -> np.ndarray:
    """Get a boolean mask where True = dark particle.
    
    Args:
        particle_flags: Tensor of particle flags, or None
        n: Number of particles
    
    Returns:
        Boolean mask where True = dark particle
    """
    if particle_flags is None:
        return np.zeros(n, dtype=bool)
    
    if isinstance(particle_flags, torch.Tensor):
        flags = particle_flags.cpu().numpy()
    else:
        flags = particle_flags
    
    return (flags & PARTICLE_FLAG_DARK) != 0


def get_coupling_mask(
    particle_flags: torch.Tensor | np.ndarray | None,
    n: int,
) -> np.ndarray:
    """Get a boolean mask of particles that can couple to carriers.
    
    Dark particles cannot couple, so they are excluded.
    
    Args:
        particle_flags: Tensor of particle flags, or None
        n: Number of particles
    
    Returns:
        Boolean mask where True = can couple
    """
    if particle_flags is None:
        return np.ones(n, dtype=bool)
    
    if isinstance(particle_flags, torch.Tensor):
        flags = particle_flags.cpu().numpy()
    else:
        flags = particle_flags
    
    # Particles that are NOT dark can couple
    return (flags & PARTICLE_FLAG_DARK) == 0
