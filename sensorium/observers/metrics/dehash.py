"""Dehash observer for reversing token ID hashing.

Recovers original byte values from hashed token IDs.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch

from sensorium.observers.types import ObserverProtocol


class DehashObserver(ObserverProtocol):
    """Observer that dehashes token IDs back to bytes for reconstruction.
    
    Reverses the universal tokenizer hash: byte = ((token_id - pos) * inv_prime) & mask
    """
    
    def __init__(self, prime: int = 31, vocab: int = 4096, segment_size: int = 784):
        self.prime = prime
        self.vocab = vocab
        self.mask = vocab - 1
        self.segment_size = segment_size
        self.inv_prime = pow(prime, -1, vocab)
    
    def observe(self, observation=None, **kwargs) -> Dict[str, Any]:
        if observation is None:
            return {}
        
        data = observation.data if hasattr(observation, "data") else observation
        
        token_ids = data.get("token_ids")
        energies = data.get("energies")
        
        if token_ids is None:
            return data
        
        if isinstance(token_ids, torch.Tensor):
            token_ids_t = token_ids
            device = token_ids_t.device
        else:
            token_ids_t = torch.tensor(token_ids, dtype=torch.int64)
            device = torch.device("cpu")
        
        # Calculate positions (wrapping every segment_size bytes)
        n = len(token_ids_t)
        indices = torch.arange(n, device=device, dtype=torch.int64)
        pos = torch.remainder(indices, self.segment_size)
        
        # Dehash: byte = ((token_id - pos) * inv_prime) & mask
        diff = token_ids_t - pos
        target = diff & self.mask
        recovered_vals = (target * self.inv_prime) & self.mask
        
        # Clamp to valid byte range
        recovered_vals_clamped = torch.clamp(recovered_vals, 0, 255)
        prompt_flat = recovered_vals_clamped.cpu().numpy().astype(np.uint8)
        
        # Convert energies to numpy
        if isinstance(energies, torch.Tensor):
            energy_by_tid = energies.cpu().numpy()
        else:
            energy_by_tid = np.array(energies) if energies is not None else np.ones(n)
        
        return {
            "prompt_flat": prompt_flat,
            "energy_by_tid": energy_by_tid,
        }
