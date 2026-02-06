"""Rule shift prediction observer using trie pattern matching.

Predicts next bytes at evaluation points using the learned
pattern distribution from the manifold.
"""

from __future__ import annotations

from typing import List, Tuple, Any, Dict

import numpy as np

from sensorium.observers.types import ObserverProtocol


class RuleShiftPredictor(ObserverProtocol):
    """Observer that predicts next byte using trie pattern matching.
    
    Used to evaluate how well the manifold adapts when patterns shift.
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        prime: int = 31,
        context_length: int = 8,
    ):
        self.vocab_size = vocab_size
        self.prime = prime
        self.context_length = context_length
        self.mask = vocab_size - 1
        self.inv_prime = pow(prime, -1, vocab_size)
    
    def observe(self, observation=None, **kwargs) -> Dict[str, Any]:
        """Predict bytes at evaluation points using only past data."""
        if observation is None:
            return {}
        
        # Extract data from observation
        data = observation.data if hasattr(observation, "data") else observation
        
        token_ids = data.get("token_ids")
        energies = data.get("energies")
        forward_phrase = data.get("forward_phrase", "")
        reverse_phrase = data.get("reverse_phrase", "")
        forward_reps = data.get("forward_reps", 50)
        reverse_reps = data.get("reverse_reps", 50)
        eval_every = data.get("eval_every", 5)
        segment_size = data.get("segment_size", 24)
        phase_switch_byte = data.get("phase_switch_byte", 0)
        
        if token_ids is None:
            return data
        
        # Convert to numpy
        token_ids_np = token_ids.cpu().numpy() if hasattr(token_ids, 'cpu') else np.array(token_ids)
        energies_np = energies.cpu().numpy() if hasattr(energies, 'cpu') else np.ones(len(token_ids_np))
        
        accuracy_history = []
        total_reps = forward_reps + reverse_reps
        
        for rep in range(eval_every, total_reps, eval_every):
            current_byte = rep * segment_size
            phase = "forward" if current_byte < phase_switch_byte else "reverse"
            test_phrase = forward_phrase if phase == "forward" else reverse_phrase
            
            correct = 0
            total = 0
            
            for seg_pos in range(self.context_length, segment_size):
                context_tids = []
                for i in range(self.context_length):
                    ctx_pos = seg_pos - self.context_length + i
                    byte_val = ord(test_phrase[ctx_pos])
                    tid = (byte_val * self.prime + ctx_pos) & self.mask
                    context_tids.append(tid)
                
                predictions = self._predict_next_byte(
                    token_ids_np[:current_byte],
                    energies_np[:current_byte],
                    context_tids,
                    seg_pos,
                )
                
                actual_byte = ord(test_phrase[seg_pos])
                predicted_byte = predictions[0][0] if predictions else 128
                
                if predicted_byte == actual_byte:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            
            accuracy_history.append({
                "rep": rep,
                "byte_position": current_byte,
                "phase": phase,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            })
        
        return {
            "accuracy_history": accuracy_history,
        }
    
    def _predict_next_byte(
        self,
        token_ids: np.ndarray,
        energies: np.ndarray,
        context_tids: List[int],
        target_seg_pos: int,
    ) -> List[Tuple[int, float]]:
        """Predict next byte using trie pattern matching."""
        scores = np.zeros(256, dtype=np.float32)
        n_particles = len(token_ids)
        context_len = len(context_tids)
        
        for start_idx in range(n_particles - context_len):
            match = True
            for j, ctx_tid in enumerate(context_tids):
                if token_ids[start_idx + j] != ctx_tid:
                    match = False
                    break
            
            if match:
                next_idx = start_idx + context_len
                if next_idx < n_particles:
                    next_tid = int(token_ids[next_idx])
                    next_energy = float(energies[next_idx])
                    
                    byte_val = self._dehash(next_tid, target_seg_pos)
                    if 0 <= byte_val < 256:
                        scores[byte_val] += next_energy
        
        if scores.sum() > 0:
            scores = scores / scores.sum()
        
        top_indices = np.argsort(scores)[::-1][:5]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def _dehash(self, token_id: int, position: int) -> int:
        """Reverse the hash to get the original byte value."""
        target = (token_id - position) & self.mask
        byte_val = (target * self.inv_prime) & self.mask
        return byte_val
