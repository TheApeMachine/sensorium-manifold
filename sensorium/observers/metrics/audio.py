"""Audio metric observers.

Observers for audio waveform prediction and reconstruction.
"""

from __future__ import annotations

import numpy as np

from sensorium.observers.types import ObserverProtocol
from sensorium.dataset.audio import dequantize_bytes_to_audio


class AudioPeriodicityPredictor(ObserverProtocol):
    """Observer that predicts audio samples using periodic position matching.
    
    Uses the periodicity of audio waveforms to predict masked samples
    by matching position within the waveform period.
    
    Requires observation to contain:
    - train_bytes: Training audio bytes
    - test_bytes: Test audio bytes
    - split_idx: Index where train/test split occurs
    - mask_positions: Set of positions to reconstruct
    
    Returns:
    - reconstructed: Reconstructed byte string
    - mae: Mean absolute error
    - mse: Mean squared error
    - snr: Signal-to-noise ratio in dB
    - accuracy: Exact byte match accuracy
    
    Example:
        predictor = AudioPeriodicityPredictor(period_samples=18)
        result = predictor.observe({
            "train_bytes": train,
            "test_bytes": test,
            "split_idx": 6400,
            "mask_positions": {100, 200, 300},
        })
    """
    
    def __init__(self, period_samples: int):
        """
        Args:
            period_samples: Number of samples per waveform period
        """
        self.period_samples = period_samples
    
    def observe(self, observation=None, **kwargs):
        """Predict masked samples using position periodicity."""
        if observation is None:
            return {}
        
        # Handle dict or ObservationResult
        data = observation.data if hasattr(observation, "data") else observation
        
        train_bytes = data.get("train_bytes", b"")
        test_bytes = data.get("test_bytes", b"")
        split_idx = data.get("split_idx", 0)
        mask_positions = data.get("mask_positions", set())
        
        if not train_bytes or not test_bytes or not mask_positions:
            return {}
        
        reconstructed = bytearray(test_bytes)
        
        for pos in sorted(mask_positions):
            seg_pos = (split_idx + pos) % self.period_samples
            predicted = self._predict_by_period(train_bytes, seg_pos)
            reconstructed[pos] = predicted
        
        # Calculate metrics
        original_audio = dequantize_bytes_to_audio(test_bytes)
        recon_audio = dequantize_bytes_to_audio(bytes(reconstructed))
        
        mae = float(np.mean(np.abs(original_audio - recon_audio)))
        mse = float(np.mean((original_audio - recon_audio) ** 2))
        
        signal_power = np.mean(original_audio ** 2)
        noise_power = mse
        snr = float(10 * np.log10(signal_power / (noise_power + 1e-10)))
        
        correct = sum(1 for pos in mask_positions if reconstructed[pos] == test_bytes[pos])
        accuracy = correct / len(mask_positions) if mask_positions else 0.0
        
        return {
            "reconstructed": bytes(reconstructed),
            "mae": mae,
            "mse": mse,
            "snr": snr,
            "accuracy": float(accuracy),
            "n_masked": len(mask_positions),
            "n_correct": correct,
        }
    
    def _predict_by_period(self, train_bytes: bytes, target_seg_pos: int) -> int:
        """Predict using weighted average of matching periodic positions."""
        values = []
        weights = []
        
        for i in range(len(train_bytes)):
            if i % self.period_samples == target_seg_pos:
                values.append(train_bytes[i])
                recency = (i + 1) / len(train_bytes)
                weights.append(recency)
        
        if not values:
            return 128  # Default to middle of byte range
        
        values_np = np.array(values, dtype=np.float32)
        weights_np = np.array(weights, dtype=np.float32)
        weights_np = weights_np / weights_np.sum()
        
        predicted = np.average(values_np, weights=weights_np)
        return int(np.clip(np.round(predicted), 0, 255))
