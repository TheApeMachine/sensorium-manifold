"""Universal Tokenizer: Modality-agnostic byte-to-token transformation.

The Universal Tokenizer converts arbitrary byte streams into token IDs suitable for
physics-based simulation on the Sensorium Manifold. Unlike vocabulary-based tokenizers
(BPE, WordPiece), it uses deterministic hashing: Hash(byte_value, position) → token_id.

Core Principle: Collision as Compression
----------------------------------------
The key insight is that hash collisions are intentional and beneficial. When tokens
at the same relative position share the same byte value, they hash to identical IDs.

Example with MNIST:
  - Each image is tokenized with position indices 0..783 (28x28 pixels)
  - The first pixel (position 0) of two images with the same intensity → same token ID
  - In the physics simulation, these become the same particle at the same manifold position

This creates "thermodynamic tries" where recurring patterns naturally cluster, enabling:
  - Implicit compression through shared particle representations
  - Pattern recognition via particle collision/proximity in manifold space
  - Hierarchical structure emerging from position-aware hashing

The position reset boundary (per-image, per-sentence, etc.) is a hyperparameter that
controls the granularity of pattern sharing across samples.

Output Format
-------------
Each token batch provides two complementary views of the same data:
  - Oscillator view: Spectral properties (frequency ω, phase, energy) for wave dynamics
  - Particle view: Physical properties (mass, heat, excitation) for collision dynamics

These views share underlying values (ω ↔ excitation, energy) ensuring consistency
between the spectral and geometric layers of the physics simulation.
"""

from dataclasses import dataclass
from typing import Iterator, List

import torch
from tensordict import TensorDict

from sensorium.kernels.runtime import get_device
from sensorium.tokenizer.prototype import Tokenizer
from sensorium.dataset.base import DatasetProtocol


class UniversalTokenizer(Tokenizer):
    """Universal byte-to-token transformer with deterministic hashing.
    
    The hash function is: token_id = (byte_value * prime + position) & mask
    
    This polynomial rolling hash provides:
      - O(1) computation per token
      - Deterministic output (same input → same output, always)
      - Uniform distribution across vocabulary (with good prime choice)
      - Intentional collisions for repeated patterns at same positions
    """
    
    def __init__(self, *, datasets: List[DatasetProtocol]):
        self.datasets = datasets
        self.device = get_device()
        self.sequence_indices = 0
    
    def tokenize(self, cell_ids: list[int], byte_values: list[int], random_positions: bool = False) -> torch.Tensor:
        """Use the raw byte values and sequence indices to create a token ID.
        
        The token ID is a 64-bit integer with the following structure:

        |    High Bits (Spatial)    |   Low Bits (Content)   |
        |---------------------------|------------------------|
        |   Morton Code (Cell ID)   |       Byte Value       |
        |   "Where I am"            |       "What I am"      |
        """
        gx, gy, gz = self.grid_size
        dx = 1.0 / float(max(gx, gy, gz))
        domain = torch.tensor([gx * dx, gy * dx, gz * dx], device=self.device, dtype=torch.float32)
        positions = torch.rand(len(cell_ids), 3, device=self.device, dtype=torch.float32) * domain
        velocities = torch.tensor(0, device=self.device, dtype=torch.float32)

        if random_positions:
            for position in positions:
                """update the position to a random position within the domain"""
                position = torch.rand(3, device=self.device, dtype=torch.float32) * domain
                position = position - position % dx
                positions[i] = position


            for velocity in velocities:
                """update the velocity to a random velocity within the domain"""
                velocity = torch.rand(3, device=self.device, dtype=torch.float32) * domain
                velocity = velocity - velocity % dx
                velocities[i] = velocity

        token = TensorDict(
            {
                "cell_ids": torch.tensor(cell_ids, device=self.device, dtype=torch.int64),
                "byte_values": torch.tensor(byte_values, device=self.device, dtype=torch.int64),
                "token_ids": (cell_ids << 8) | byte_values,
                "sequence_indices": self.sequence_indices,
                "positions": positions,
                "velocities": torch.tensor(0, device=self.device, dtype=torch.float32),
                "heats": torch.tensor(0, device=self.device, dtype=torch.float32),
                "masses": torch.tensor(0, device=self.device, dtype=torch.float32),
                "energies": torch.tensor(0, device=self.device, dtype=torch.float32),
                "excitations": torch.tensor(0, device=self.device, dtype=torch.float32),
                "phase": torch.tensor(0, device=self.device, dtype=torch.float32),
                "omega": torch.tensor(0, device=self.device, dtype=torch.float32),
                "energy_osc": torch.tensor(0, device=self.device, dtype=torch.float32)
            }
        )

        self.sequence_indices += 1
        return token

    def stream(self, random_positions: bool = False) -> Iterator[torch.Tensor]:
        for dataset in self.datasets:
            for cell_ids, byte_values in dataset.generate():
                yield self.tokenize(cell_ids, byte_values, random_positions)
