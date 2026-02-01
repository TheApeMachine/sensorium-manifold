"""
Unified Spectral Manifold

A dimensionality-agnostic thermodynamic manifold where particles from any modality
can coexist. Text tokens, audio frequencies, image frequencies, video frequencies—
all just particles with positions in a continuous space.

The manifold doesn't know or care about modality. Modality is:
1. Determined by which encoder created the particle
2. Emergent from which region of the space the particle occupies
3. Used by decoders to select which particles to render

This enables true native multimodality: not adapters glued together,
but a unified dynamics where cross-modal relationships emerge naturally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..core.config import PhysicsConfig
from ..core.state import BatchState
from ..physics.engine import ThermodynamicEngine


class Modality(Enum):
    """Modality tags for particles. Used by decoders, not by the manifold itself."""
    TEXT = auto()
    AUDIO = auto()
    IMAGE = auto()
    VIDEO = auto()
    UNKNOWN = auto()


@dataclass
class UnifiedParticle:
    """A particle in the unified manifold.
    
    The position can be any dimensionality:
    - Text: D-dimensional embedding
    - Audio: 1D frequency
    - Image: 2D frequency (u, v)
    - Video: 3D frequency (u, v, t)
    
    The manifold doesn't care—it just computes distances and flows.
    """
    position: torch.Tensor  # Shape: [D] where D is any dimensionality
    energy: torch.Tensor    # Scalar
    heat: torch.Tensor      # Scalar
    modality: Modality = Modality.UNKNOWN
    
    # Optional: additional properties for rendering
    phase: Optional[torch.Tensor] = None  # For frequency-domain particles
    token_id: Optional[int] = None        # For text particles


@dataclass 
class UnifiedOutput:
    """Output from the unified manifold."""
    particles: List[UnifiedParticle]
    by_modality: Dict[Modality, List[int]]  # modality -> particle indices
    meta: Dict[str, Any]


class UnifiedManifold(ThermodynamicEngine):
    """
    Dimensionality-agnostic thermodynamic manifold.
    
    All particles live in a shared space. The manifold processes them
    with the same thermodynamic dynamics regardless of their origin
    or intended output modality.
    
    Key insight: particles of different dimensionalities can coexist
    if we project them into a common embedding space for distance
    computation, while preserving their native coordinates for decoding.
    """
    
    def __init__(
        self, 
        config: PhysicsConfig, 
        device: torch.device,
        embed_dim: int = 256,
    ):
        super().__init__(config, device)
        self.embed_dim = embed_dim
        
        # Storage for heterogeneous particles
        self._particles: List[UnifiedParticle] = []
        self._attractors: List[UnifiedParticle] = []
        
        # Projection matrices for different dimensionalities → common space
        # Lazily initialized as we encounter new dimensionalities
        self._projections: Dict[int, torch.Tensor] = {}
        
        # Modality region centers (learned/adapted over time)
        self._modality_centers: Dict[Modality, torch.Tensor] = {}
        
    def _get_projection(self, dim: int) -> torch.Tensor:
        """Get or create projection matrix for a given dimensionality."""
        if dim not in self._projections:
            if dim == self.embed_dim:
                # Identity projection
                self._projections[dim] = torch.eye(
                    dim, device=self.device, dtype=torch.float32
                )
            elif dim < self.embed_dim:
                # Pad with zeros (or use learned projection)
                proj = torch.zeros(dim, self.embed_dim, device=self.device, dtype=torch.float32)
                proj[:, :dim] = torch.eye(dim, device=self.device, dtype=torch.float32)
                self._projections[dim] = proj
            else:
                # Project down (random projection preserves distances approximately)
                proj = torch.randn(dim, self.embed_dim, device=self.device, dtype=torch.float32)
                proj = proj / (torch.linalg.norm(proj, dim=1, keepdim=True) + self.cfg.eps)
                self._projections[dim] = proj
        return self._projections[dim]
    
    def _to_common_space(self, position: torch.Tensor) -> torch.Tensor:
        """Project a position to the common embedding space."""
        dim = position.numel()
        proj = self._get_projection(dim)
        return position.flatten() @ proj
    
    def _update_modality_center(self, modality: Modality, position: torch.Tensor) -> None:
        """Update running average of modality region center."""
        common_pos = self._to_common_space(position)
        if modality not in self._modality_centers:
            self._modality_centers[modality] = common_pos.clone()
        else:
            # EMA update
            alpha = self.cfg.dt / (self.cfg.tau + self.cfg.dt)
            self._modality_centers[modality] = (
                (1 - alpha) * self._modality_centers[modality] + alpha * common_pos
            )
    
    # =========================================================================
    # Particle Management
    # =========================================================================
    
    def add_particle(
        self,
        position: torch.Tensor,
        energy: float = 1.0,
        heat: float = 0.0,
        modality: Modality = Modality.UNKNOWN,
        phase: Optional[torch.Tensor] = None,
        token_id: Optional[int] = None,
    ) -> int:
        """Add a particle to the manifold. Returns particle index."""
        particle = UnifiedParticle(
            position=position.to(device=self.device, dtype=torch.float32).flatten(),
            energy=torch.tensor(energy, device=self.device, dtype=torch.float32),
            heat=torch.tensor(heat, device=self.device, dtype=torch.float32),
            modality=modality,
            phase=phase.to(device=self.device, dtype=torch.float32) if phase is not None else None,
            token_id=token_id,
        )
        self._particles.append(particle)
        self._update_modality_center(modality, particle.position)
        return len(self._particles) - 1
    
    def add_attractor(
        self,
        position: torch.Tensor,
        energy: float = 1.0,
        modality: Modality = Modality.UNKNOWN,
    ) -> int:
        """Add an attractor to the manifold. Returns attractor index."""
        attractor = UnifiedParticle(
            position=position.to(device=self.device, dtype=torch.float32).flatten(),
            energy=torch.tensor(energy, device=self.device, dtype=torch.float32),
            heat=torch.tensor(0.0, device=self.device, dtype=torch.float32),
            modality=modality,
        )
        self._attractors.append(attractor)
        self._update_modality_center(modality, attractor.position)
        return len(self._attractors) - 1
    
    def clear(self) -> None:
        """Clear all particles and attractors."""
        self._particles.clear()
        self._attractors.clear()
    
    # =========================================================================
    # Batch Encoding (for use with modal encoders)
    # =========================================================================
    
    def encode_text(self, embeddings: torch.Tensor, token_ids: Optional[List[int]] = None) -> List[int]:
        """
        Add text particles from embeddings.
        
        Args:
            embeddings: [N, D] tensor of token embeddings
            token_ids: Optional list of token IDs for decoding
            
        Returns:
            List of particle indices
        """
        embeddings = embeddings.to(device=self.device, dtype=torch.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        
        indices = []
        for i in range(embeddings.shape[0]):
            idx = self.add_particle(
                position=embeddings[i],
                modality=Modality.TEXT,
                token_id=token_ids[i] if token_ids else None,
            )
            indices.append(idx)
        return indices
    
    def encode_image(
        self, 
        image: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[int]:
        """
        Add image particles from an image via 2D FFT.
        
        Args:
            image: [H, W] or [C, H, W] tensor (grayscale or color)
            top_k: Only keep top-k frequency components by magnitude
            
        Returns:
            List of particle indices
        """
        image = image.to(device=self.device, dtype=torch.float32)
        
        # Handle color images by converting to grayscale for now
        if image.ndim == 3:
            if image.shape[0] in (1, 3, 4):
                # Channel first: average channels
                image = image.mean(dim=0)
            else:
                # Assume [H, W, C]
                image = image.mean(dim=-1)
        
        # 2D FFT
        spectrum = torch.fft.fft2(image)
        spectrum_shifted = torch.fft.fftshift(spectrum)
        
        magnitudes = spectrum_shifted.abs()
        phases = spectrum_shifted.angle()
        
        H, W = image.shape
        
        # Create frequency coordinates
        u_coords = torch.arange(H, device=self.device, dtype=torch.float32) - H // 2
        v_coords = torch.arange(W, device=self.device, dtype=torch.float32) - W // 2
        uu, vv = torch.meshgrid(u_coords, v_coords, indexing='ij')
        
        # Flatten everything
        mags_flat = magnitudes.flatten()
        phases_flat = phases.flatten()
        u_flat = uu.flatten()
        v_flat = vv.flatten()
        
        # Select top-k if specified
        if top_k is not None and top_k < mags_flat.numel():
            _, top_indices = torch.topk(mags_flat, top_k)
            mags_flat = mags_flat[top_indices]
            phases_flat = phases_flat[top_indices]
            u_flat = u_flat[top_indices]
            v_flat = v_flat[top_indices]
        
        # Normalize magnitudes to energy
        total_mag = mags_flat.sum() + self.cfg.eps
        energies = mags_flat / total_mag
        
        # Add particles
        indices = []
        for i in range(mags_flat.numel()):
            # Position is 2D frequency coordinate
            position = torch.stack([u_flat[i], v_flat[i]])
            phase = phases_flat[i].unsqueeze(0)
            
            idx = self.add_particle(
                position=position,
                energy=float(energies[i].item()),
                modality=Modality.IMAGE,
                phase=phase,
            )
            indices.append(idx)
        
        return indices
    
    def encode_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 44100,
        top_k: Optional[int] = None,
    ) -> List[int]:
        """
        Add audio particles from a waveform via 1D FFT.
        
        Args:
            waveform: [N] tensor of audio samples
            sample_rate: Sample rate in Hz
            top_k: Only keep top-k frequency components
            
        Returns:
            List of particle indices
        """
        waveform = waveform.to(device=self.device, dtype=torch.float32).flatten()
        
        # 1D FFT
        spectrum = torch.fft.rfft(waveform)
        magnitudes = spectrum.abs()
        phases = spectrum.angle()
        
        # Frequency bins
        n = waveform.numel()
        freqs = torch.fft.rfftfreq(n, d=1.0/sample_rate).to(device=self.device)
        
        # Select top-k if specified
        if top_k is not None and top_k < magnitudes.numel():
            _, top_indices = torch.topk(magnitudes, top_k)
            magnitudes = magnitudes[top_indices]
            phases = phases[top_indices]
            freqs = freqs[top_indices]
        
        # Normalize to energy
        total_mag = magnitudes.sum() + self.cfg.eps
        energies = magnitudes / total_mag
        
        # Add particles
        indices = []
        for i in range(magnitudes.numel()):
            position = freqs[i].unsqueeze(0)  # 1D position
            phase = phases[i].unsqueeze(0)
            
            idx = self.add_particle(
                position=position,
                energy=float(energies[i].item()),
                modality=Modality.AUDIO,
                phase=phase,
            )
            indices.append(idx)
        
        return indices
    
    # =========================================================================
    # Decoding (Attractor-based modality selection)
    # =========================================================================
    
    def decode_image(self, output_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Decode image-modality particles back to an image.
        
        Uses attractor dynamics: particles near the image modality center
        are selected and their frequencies reconstructed via iFFT2D.
        """
        H, W = output_shape
        
        # Initialize spectrum
        spectrum = torch.zeros(H, W, dtype=torch.complex64, device=self.device)
        
        # Collect image particles
        for p in self._particles:
            if p.modality != Modality.IMAGE:
                continue
            if p.position.numel() != 2:
                continue  # Not a 2D frequency
            
            u, v = p.position[0], p.position[1]
            
            # Convert back to spectrum indices
            u_idx = int(u.item() + H // 2)
            v_idx = int(v.item() + W // 2)
            
            if 0 <= u_idx < H and 0 <= v_idx < W:
                magnitude = p.energy.item()
                phase = p.phase[0].item() if p.phase is not None else 0.0
                spectrum[u_idx, v_idx] = magnitude * torch.exp(
                    torch.tensor(1j * phase, device=self.device)
                )
        
        # Inverse shift and FFT
        spectrum_unshifted = torch.fft.ifftshift(spectrum)
        image = torch.fft.ifft2(spectrum_unshifted).real
        
        return image
    
    def decode_audio(self, num_samples: int, sample_rate: int = 44100) -> torch.Tensor:
        """
        Decode audio-modality particles back to a waveform.
        """
        # Number of frequency bins for rfft
        n_freqs = num_samples // 2 + 1
        spectrum = torch.zeros(n_freqs, dtype=torch.complex64, device=self.device)
        
        # Frequency resolution
        freq_resolution = sample_rate / num_samples
        
        for p in self._particles:
            if p.modality != Modality.AUDIO:
                continue
            if p.position.numel() != 1:
                continue
            
            freq = p.position[0].item()
            freq_idx = int(freq / freq_resolution)
            
            if 0 <= freq_idx < n_freqs:
                magnitude = p.energy.item()
                phase = p.phase[0].item() if p.phase is not None else 0.0
                spectrum[freq_idx] = magnitude * torch.exp(
                    torch.tensor(1j * phase, device=self.device)
                )
        
        # Inverse FFT
        waveform = torch.fft.irfft(spectrum, n=num_samples)
        
        return waveform
    
    def decode_text(self, vocab: List[str], top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Decode text-modality particles to tokens.
        
        Returns tokens with highest energy.
        """
        text_particles = [
            (p.token_id, p.energy.item()) 
            for p in self._particles 
            if p.modality == Modality.TEXT and p.token_id is not None
        ]
        
        if not text_particles:
            return []
        
        # Sort by energy
        text_particles.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for token_id, energy in text_particles[:top_k]:
            if 0 <= token_id < len(vocab):
                results.append((vocab[token_id], energy))
        
        return results
    
    # =========================================================================
    # Thermodynamic Step (Override to handle heterogeneous particles)
    # =========================================================================
    
    def _build_batch_state(self) -> None:
        """Convert particle list to BatchState for engine."""
        if not self._particles:
            self.particles = BatchState.empty()
        else:
            # Project all to common space for dynamics
            positions = torch.stack([
                self._to_common_space(p.position) for p in self._particles
            ])
            energies = torch.stack([p.energy for p in self._particles])
            heats = torch.stack([p.heat for p in self._particles])
            
            self.particles = BatchState({
                "position": positions,
                "energy": energies,
                "heat": heats,
            })
        
        if not self._attractors:
            self.attractors = BatchState.empty()
        else:
            positions = torch.stack([
                self._to_common_space(a.position) for a in self._attractors
            ])
            energies = torch.stack([a.energy for a in self._attractors])
            heats = torch.stack([a.heat for a in self._attractors])
            
            self.attractors = BatchState({
                "position": positions,
                "energy": energies,
                "heat": heats,
            })
    
    def _sync_from_batch_state(self) -> None:
        """Sync updates back to particle list."""
        if self.particles.n == 0:
            return
        
        positions = self.particles.get("position")
        energies = self.particles.get("energy")
        heats = self.particles.get("heat")
        
        for i, p in enumerate(self._particles):
            # Update energy and heat (position in common space doesn't 
            # directly map back to native space trivially, so we skip that)
            p.energy = energies[i]
            p.heat = heats[i]
    
    def step(self) -> None:
        """One step of unified thermodynamic dynamics."""
        self._build_batch_state()
        self.step_physics()
        self._sync_from_batch_state()
    
    # =========================================================================
    # Output
    # =========================================================================
    
    def output_state(self) -> UnifiedOutput:
        """Get current state organized by modality."""
        by_modality: Dict[Modality, List[int]] = {m: [] for m in Modality}
        
        for i, p in enumerate(self._particles):
            by_modality[p.modality].append(i)
        
        return UnifiedOutput(
            particles=self._particles.copy(),
            by_modality=by_modality,
            meta={
                "num_particles": len(self._particles),
                "num_attractors": len(self._attractors),
                "modality_counts": {m.name: len(v) for m, v in by_modality.items()},
            },
        )
