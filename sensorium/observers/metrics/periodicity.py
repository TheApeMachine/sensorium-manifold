"""Periodicity metric observers.

Observers for time series prediction using position periodicity.
"""

from __future__ import annotations

import numpy as np

from sensorium.observers.types import ObserverProtocol


class PositionPeriodicityPredictor(ObserverProtocol):
    """Observer that predicts next byte using position periodicity.
    
    The segment_size creates a thermodynamic trie where values at the same
    segment position collide. For time series, this captures periodicity.
    
    Requires observation to contain:
    - train_bytes: Training data bytes
    - test_bytes: Test data bytes  
    - split_idx: Index where train/test split occurs
    
    Returns metrics:
    - mae: Mean absolute error
    - rmse: Root mean squared error
    - direction_accuracy: Fraction of correct trend predictions
    - within_5, within_10: Fraction within N quantization levels
    
    Example:
        predictor = PositionPeriodicityPredictor(segment_size=50)
        result = predictor.observe({
            "train_bytes": train_data,
            "test_bytes": test_data,
            "split_idx": 1600,
        })
    """
    
    def __init__(self, segment_size: int = 50):
        """
        Args:
            segment_size: Period for position-based prediction
        """
        self.segment_size = segment_size
    
    def observe(self, observation=None, **kwargs):
        """Predict test values using position periodicity."""
        if observation is None:
            return {}
        
        # Handle dict or ObservationResult
        data = observation.data if hasattr(observation, "data") else observation
        
        train_bytes = data.get("train_bytes", b"")
        test_bytes = data.get("test_bytes", b"")
        split_idx = data.get("split_idx", 0)
        
        if not train_bytes or not test_bytes:
            return {}
        
        actuals = []
        predictions = []
        
        for test_pos in range(len(test_bytes)):
            actual_byte = test_bytes[test_pos]
            global_pos = split_idx + test_pos
            seg_pos = global_pos % self.segment_size
            
            predicted = self._predict_by_position(train_bytes, seg_pos)
            
            actuals.append(actual_byte)
            predictions.append(predicted)
        
        # Calculate metrics
        actuals_np = np.array(actuals, dtype=np.float32)
        predictions_np = np.array(predictions, dtype=np.float32)
        
        mae = float(np.mean(np.abs(actuals_np - predictions_np)))
        mse = float(np.mean((actuals_np - predictions_np) ** 2))
        rmse = float(np.sqrt(mse))
        
        exact_accuracy = float(np.mean(actuals_np == predictions_np))
        
        # Direction accuracy
        if len(actuals_np) > 1:
            actual_diff = np.diff(actuals_np)
            pred_diff = np.diff(predictions_np)
            direction_accuracy = float(np.mean(np.sign(actual_diff) == np.sign(pred_diff)))
        else:
            direction_accuracy = 0.0
        
        within_5 = float(np.mean(np.abs(actuals_np - predictions_np) <= 5))
        within_10 = float(np.mean(np.abs(actuals_np - predictions_np) <= 10))
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "exact_accuracy": exact_accuracy,
            "direction_accuracy": direction_accuracy,
            "within_5": within_5,
            "within_10": within_10,
            "actuals": actuals,
            "predictions": predictions,
        }
    
    def _predict_by_position(self, train_bytes: bytes, target_seg_pos: int) -> int:
        """Predict using weighted average of matching positions.
        
        Uses recency-weighted averaging: more recent values get higher weight.
        """
        values = []
        weights = []
        
        for i in range(len(train_bytes)):
            if i % self.segment_size == target_seg_pos:
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
