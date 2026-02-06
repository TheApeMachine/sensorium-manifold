"""Diffusion/denoising metric observers.

Observers for text reconstruction and pattern matching.
"""

from __future__ import annotations

from typing import List

import numpy as np

from sensorium.observers.types import ObserverProtocol


class TriePatternMatcher(ObserverProtocol):
    """Observer that performs trie-based pattern matching for reconstruction.
    
    Matches context bytes against training patterns to predict masked bytes.
    Uses the thermodynamic trie structure (token IDs and energies) to find
    matching patterns and predict missing bytes.
    
    Requires observation to contain:
    - train_bytes: Training data bytes
    - test_bytes: Test data bytes (with masked positions)
    - token_ids: Token IDs from simulation
    - energies: Energy values from simulation
    - mask_positions: Set of positions to reconstruct
    
    Returns:
    - reconstructed: Reconstructed byte string
    - n_correct: Number of correctly predicted bytes
    - n_masked: Number of masked positions
    - char_accuracy: Accuracy on masked positions
    
    Example:
        matcher = TriePatternMatcher(
            vocab_size=4096,
            prime=31,
            context_length=5,
            segment_size=64,
        )
        result = matcher.observe({
            "train_bytes": train,
            "test_bytes": test,
            "token_ids": token_ids,
            "energies": energies,
            "mask_positions": {10, 20, 30},
        })
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        prime: int = 31,
        context_length: int = 5,
        segment_size: int = 64,
    ):
        """
        Args:
            vocab_size: Vocabulary size for hashing
            prime: Prime for hash function
            context_length: Context window for matching
            segment_size: Segment size for position calculations
        """
        self.vocab_size = vocab_size
        self.prime = prime
        self.context_length = context_length
        self.segment_size = segment_size
        self.mask = vocab_size - 1
        self.inv_prime = pow(prime, -1, vocab_size)
    
    def observe(self, observation=None, **kwargs):
        """Reconstruct masked positions using trie matching."""
        if observation is None:
            return {}
        
        # Handle dict or ObservationResult
        data = observation.data if hasattr(observation, "data") else observation
        
        train_bytes = data.get("train_bytes", b"")
        test_bytes = data.get("test_bytes", b"")
        token_ids = data.get("token_ids")
        energies = data.get("energies")
        mask_positions = data.get("mask_positions", set())
        
        if token_ids is None or not mask_positions:
            return {}
        
        # Convert to numpy
        token_ids_np = token_ids.cpu().numpy() if hasattr(token_ids, 'cpu') else np.array(token_ids)
        energies_np = energies.cpu().numpy() if hasattr(energies, 'cpu') else np.ones(len(token_ids_np))
        
        # Reconstruct
        reconstructed = bytearray(test_bytes)
        
        for pos in sorted(mask_positions):
            # Get context (non-masked neighbors)
            context = []
            for i in range(max(0, pos - self.context_length), pos):
                if i not in mask_positions:
                    context.append((i, test_bytes[i]))
            for i in range(pos + 1, min(len(test_bytes), pos + self.context_length + 1)):
                if i not in mask_positions:
                    context.append((i, reconstructed[i]))
            
            predicted = self._predict_byte(train_bytes, token_ids_np, energies_np, context, pos)
            reconstructed[pos] = predicted
        
        # Calculate accuracy
        correct = sum(1 for pos in mask_positions if reconstructed[pos] == test_bytes[pos])
        accuracy = correct / len(mask_positions) if mask_positions else 0.0
        
        return {
            "reconstructed": bytes(reconstructed),
            "n_correct": correct,
            "n_masked": len(mask_positions),
            "char_accuracy": accuracy,
        }
    
    def _predict_byte(
        self,
        train_bytes: bytes,
        token_ids: np.ndarray,
        energies: np.ndarray,
        context: List[tuple],
        target_pos: int,
    ) -> int:
        """Predict byte using thermodynamic trie pattern matching."""
        target_seg_pos = target_pos % self.segment_size
        
        # Build context token IDs
        context_tids = []
        for ctx_pos, ctx_byte in context:
            ctx_seg_pos = ctx_pos % self.segment_size
            tid = (ctx_byte * self.prime + ctx_seg_pos) & self.mask
            context_tids.append(tid)
        
        if not context_tids:
            return self._frequency_prior(train_bytes, target_seg_pos)
        
        # Score candidates by trie matching
        scores = np.zeros(256, dtype=np.float32)
        n_train = len(token_ids)
        
        for start_idx in range(n_train - len(context_tids)):
            match_count = 0
            for j, ctx_tid in enumerate(context_tids):
                if start_idx + j < n_train and token_ids[start_idx + j] == ctx_tid:
                    match_count += 1
            
            if match_count >= min(2, len(context_tids)):
                next_idx = start_idx + len(context_tids)
                if next_idx < n_train:
                    next_tid = int(token_ids[next_idx])
                    energy = float(energies[next_idx])
                    
                    byte_val = self._dehash(next_tid, target_seg_pos)
                    if 0 <= byte_val < 256:
                        scores[byte_val] += energy * match_count
        
        if scores.sum() > 0:
            return int(np.argmax(scores))
        
        return self._frequency_prior(train_bytes, target_seg_pos)
    
    def _frequency_prior(self, train_bytes: bytes, target_seg_pos: int) -> int:
        """Return most common byte at this segment position."""
        counts = np.zeros(256, dtype=np.int32)
        for i, b in enumerate(train_bytes):
            if i % self.segment_size == target_seg_pos:
                counts[b] += 1
        
        return int(np.argmax(counts)) if counts.sum() > 0 else ord(' ')
    
    def _dehash(self, token_id: int, position: int) -> int:
        """Reverse hash to get byte value."""
        target = (token_id - position) & self.mask
        return (target * self.inv_prime) & self.mask
