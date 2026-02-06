"""Dark particle masking for coupling prevention.

This module provides utilities to mask dark particles from carrier
coupling operations, ensuring they perturb but don't become part of
the learned coherence structure.
"""

from __future__ import annotations

from typing import Any

import torch
import numpy as np

from sensorium.observers.types import PARTICLE_FLAG_DARK


class DarkParticleMask:
    """Manages masking of dark particles for coupling operations.
    
    This class wraps state access to automatically filter out dark
    particles when computing mode coupling, resonance, or any
    operation that would cause dark particles to become entangled
    with the learned coherence structure.
    
    Example:
        mask = DarkParticleMask(state)
        
        # Get only visible (non-dark) particle data
        visible_excitations = mask.visible("excitations")
        visible_positions = mask.visible("positions")
        
        # Get indices that can participate in coupling
        coupling_indices = mask.coupling_indices
    """
    
    def __init__(self, state: dict):
        """Initialize the dark particle mask.
        
        Args:
            state: Simulation state dict
        """
        self.state = state
        self._dark_mask: np.ndarray | None = None
        self._visible_mask: np.ndarray | None = None
    
    @property
    def n_particles(self) -> int:
        """Total number of particles (including dark)."""
        pos = self.state.get("positions")
        if pos is None:
            return 0
        return len(pos)
    
    @property
    def dark_mask(self) -> np.ndarray:
        """Boolean mask where True = dark particle."""
        if self._dark_mask is None:
            self._compute_masks()
        return self._dark_mask
    
    @property
    def visible_mask(self) -> np.ndarray:
        """Boolean mask where True = visible (non-dark) particle."""
        if self._visible_mask is None:
            self._compute_masks()
        return self._visible_mask
    
    @property
    def dark_indices(self) -> np.ndarray:
        """Indices of dark particles."""
        return self.dark_mask.nonzero()[0]
    
    @property
    def visible_indices(self) -> np.ndarray:
        """Indices of visible (non-dark) particles."""
        return self.visible_mask.nonzero()[0]
    
    @property
    def coupling_indices(self) -> np.ndarray:
        """Indices of particles that can couple to modes.
        
        This is the same as visible_indices - dark particles
        cannot couple.
        """
        return self.visible_indices
    
    @property
    def n_dark(self) -> int:
        """Number of dark particles."""
        return int(self.dark_mask.sum())
    
    @property
    def n_visible(self) -> int:
        """Number of visible (non-dark) particles."""
        return int(self.visible_mask.sum())
    
    def _compute_masks(self):
        """Compute dark/visible masks from particle flags."""
        n = self.n_particles
        particle_flags = self.state.get("particle_flags")
        
        if particle_flags is None or n == 0:
            self._dark_mask = np.zeros(n, dtype=bool)
            self._visible_mask = np.ones(n, dtype=bool)
            return
        
        if isinstance(particle_flags, torch.Tensor):
            flags = particle_flags.cpu().numpy()
        else:
            flags = np.asarray(particle_flags)
        
        self._dark_mask = (flags & PARTICLE_FLAG_DARK) != 0
        self._visible_mask = ~self._dark_mask
    
    def visible(self, key: str) -> torch.Tensor | np.ndarray | None:
        """Get a state tensor filtered to only visible particles.
        
        Args:
            key: State key to filter
        
        Returns:
            Filtered tensor with only visible particles, or None if key missing
        """
        tensor = self.state.get(key)
        if tensor is None:
            return None
        
        mask = self.visible_mask
        
        if isinstance(tensor, torch.Tensor):
            mask_torch = torch.tensor(mask, device=tensor.device)
            return tensor[mask_torch]
        else:
            return tensor[mask]
    
    def dark(self, key: str) -> torch.Tensor | np.ndarray | None:
        """Get a state tensor filtered to only dark particles.
        
        Args:
            key: State key to filter
        
        Returns:
            Filtered tensor with only dark particles, or None if key missing
        """
        tensor = self.state.get(key)
        if tensor is None:
            return None
        
        mask = self.dark_mask
        
        if isinstance(tensor, torch.Tensor):
            mask_torch = torch.tensor(mask, device=tensor.device)
            return tensor[mask_torch]
        else:
            return tensor[mask]
    
    def apply_coupling_mask(self, coupling_weights: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Zero out coupling weights for dark particles.
        
        This ensures dark particles don't contribute to carrier updates.
        
        Args:
            coupling_weights: (N,) or (N, K) tensor of coupling weights
        
        Returns:
            Coupling weights with dark particle contributions zeroed
        """
        mask = self.visible_mask
        
        if isinstance(coupling_weights, torch.Tensor):
            mask_torch = torch.tensor(
                mask,
                device=coupling_weights.device,
                dtype=coupling_weights.dtype,
            )
            if coupling_weights.ndim == 1:
                return coupling_weights * mask_torch
            else:
                # (N, K) - mask along particle dimension
                return coupling_weights * mask_torch.unsqueeze(-1)
        else:
            if coupling_weights.ndim == 1:
                return coupling_weights * mask.astype(coupling_weights.dtype)
            else:
                return coupling_weights * mask.astype(coupling_weights.dtype)[:, np.newaxis]
    
    def invalidate(self):
        """Invalidate cached masks (call after state changes)."""
        self._dark_mask = None
        self._visible_mask = None


def prevent_dark_coupling(
    coupling_matrix: torch.Tensor,
    particle_flags: torch.Tensor | None,
) -> torch.Tensor:
    """Zero out coupling matrix rows for dark particles.
    
    This function should be called in the carrier update physics
    to prevent dark particles from contributing to carrier resonance.
    
    Args:
        coupling_matrix: (N, K) coupling strengths between particles and carriers
        particle_flags: (N,) particle flags tensor, or None
    
    Returns:
        Modified coupling matrix with dark particle rows zeroed
    """
    if particle_flags is None:
        return coupling_matrix
    
    # Create mask: 1.0 for visible, 0.0 for dark
    dark_mask = (particle_flags & PARTICLE_FLAG_DARK) != 0
    visible_mask = (~dark_mask).to(coupling_matrix.dtype)
    
    # Apply mask along particle dimension
    return coupling_matrix * visible_mask.unsqueeze(-1)
