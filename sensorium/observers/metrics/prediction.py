"""Prediction metric observers.

Observers for next-token prediction metrics and crystallization monitoring.
"""

from __future__ import annotations

import numpy as np

from sensorium.observers.types import ObserverProtocol


class CrystallizationObserver(ObserverProtocol):
    """Observer that waits for modes to crystallize before stopping.
    
    Used during simulation to determine when the system has stabilized.
    Returns {"done_thinking": True} when criteria are met.
    
    Criteria:
    - At least min_modes modes formed
    - At least min_crystallized modes crystallized (state == 2)
    - OR max_steps have elapsed
    
    Example:
        observer = CrystallizationObserver(
            min_modes=10,
            min_crystallized=5,
            max_steps=500,
        )
    """
    
    def __init__(
        self,
        min_modes: int = 5,
        min_crystallized: int = 1,
        max_steps: int = 200,
        log_interval: int = 50,
    ):
        """
        Args:
            min_modes: Minimum modes required
            min_crystallized: Minimum crystallized modes required
            max_steps: Maximum steps before forcing stop
            log_interval: Steps between diagnostic logging
        """
        self.min_modes = min_modes
        self.min_crystallized = min_crystallized
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.step_count = 0
    
    def reset(self):
        """Reset step counter for new run."""
        self.step_count = 0
    
    def observe(self, observation=None, **kwargs):
        """Check crystallization criteria."""
        self.step_count += 1
        
        if observation is None:
            return {"done_thinking": self.step_count >= self.max_steps}
        
        # Handle dict or ObservationResult
        data = observation.data if hasattr(observation, "data") else observation
        
        # Count modes from amplitudes
        amplitudes = data.get("amplitudes")
        if amplitudes is not None:
            num_modes = int((amplitudes > 1e-6).sum().item())
        else:
            num_modes = 0
        
        # Count crystallized modes
        mode_state = data.get("mode_state")
        num_crystallized = 0
        if mode_state is not None and num_modes > 0:
            num_crystallized = int((mode_state[:num_modes] == 2).sum().item())
        
        # Check if we've met crystallization criteria
        meets_criteria = (
            num_modes >= self.min_modes and 
            num_crystallized >= self.min_crystallized
        )
        
        # Or if we've hit max steps
        done = meets_criteria or (self.step_count >= self.max_steps)
        
        # Periodic diagnostics
        if self.log_interval > 0 and self.step_count % self.log_interval == 0:
            conflict = data.get("conflict")
            max_amp = float(amplitudes.max().item()) if amplitudes is not None and amplitudes.numel() > 0 else 0
            min_conf = float(conflict[:num_modes].min().item()) if conflict is not None and num_modes > 0 else -1
            print(f"  [step {self.step_count}] modes={num_modes}, crystallized={num_crystallized}, "
                  f"max_amp={max_amp:.3f}, min_conflict={min_conf:.3f}")
        
        return {
            "done_thinking": done,
            "num_modes": num_modes,
            "num_crystallized": num_crystallized,
            "step_count": self.step_count,
        }


class NextTokenMetrics(ObserverProtocol):
    """Observer that computes next-token prediction metrics.
    
    Takes a list of predictions and computes:
    - accuracy: Exact match accuracy
    - top3_accuracy, top5_accuracy: Top-k accuracy
    - perplexity: Perplexity from probability scores
    - ambiguous_accuracy: Accuracy on ambiguous (multi-candidate) positions
    - unambiguous_accuracy: Accuracy on clear (single-candidate) positions
    
    Requires observation to contain:
    - predictions: List of dicts with keys:
        - predicted: Predicted byte value
        - actual: Actual byte value
        - top3, top5: Top-k predictions
        - scores: Score array for all 256 bytes
    
    Example:
        metrics = NextTokenMetrics()
        result = metrics.observe({"predictions": predictions_list})
    """
    
    def observe(self, observation=None, **kwargs):
        """Compute prediction metrics."""
        if observation is None:
            return {}
        
        # Handle dict or ObservationResult
        data = observation.data if hasattr(observation, "data") else observation
        
        predictions = data.get("predictions", [])
        if not predictions:
            return {
                "accuracy": 0.0,
                "top3_accuracy": 0.0,
                "top5_accuracy": 0.0,
                "perplexity": float("inf"),
                "total_predictions": 0,
            }
        
        # Calculate metrics
        correct = sum(1 for p in predictions if p["predicted"] == p["actual"])
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # Top-k accuracy
        top3_correct = sum(1 for p in predictions if p["actual"] in p["top3"])
        top5_correct = sum(1 for p in predictions if p["actual"] in p["top5"])
        top3_accuracy = top3_correct / total if total > 0 else 0.0
        top5_accuracy = top5_correct / total if total > 0 else 0.0
        
        # Perplexity approximation
        log_probs = []
        for p in predictions:
            scores = p["scores"]
            actual = p["actual"]
            # Softmax to get probabilities
            max_score = np.max(scores)
            exp_scores = np.exp(scores - max_score)
            probs = exp_scores / (exp_scores.sum() + 1e-10)
            log_probs.append(np.log(probs[actual] + 1e-10))
        
        avg_log_prob = np.mean(log_probs) if log_probs else -10
        perplexity = np.exp(-avg_log_prob)
        
        # Ambiguous vs unambiguous accuracy
        ambiguous_count = 0
        ambiguous_correct = 0
        unambiguous_count = 0
        unambiguous_correct = 0
        
        for p in predictions:
            scores = p["scores"]
            non_zero = (scores > 0.01).sum()
            is_correct = p["predicted"] == p["actual"]
            
            if non_zero > 1:
                ambiguous_count += 1
                if is_correct:
                    ambiguous_correct += 1
            else:
                unambiguous_count += 1
                if is_correct:
                    unambiguous_correct += 1
        
        ambiguous_accuracy = ambiguous_correct / ambiguous_count if ambiguous_count > 0 else 1.0
        unambiguous_accuracy = unambiguous_correct / unambiguous_count if unambiguous_count > 0 else 1.0
        
        return {
            "accuracy": accuracy,
            "top3_accuracy": top3_accuracy,
            "top5_accuracy": top5_accuracy,
            "perplexity": perplexity,
            "total_predictions": total,
            "ambiguous_cases": ambiguous_count,
            "ambiguous_accuracy": ambiguous_accuracy,
            "unambiguous_accuracy": unambiguous_accuracy,
        }
