"""Cross-modal observers for tracking dynamics and image reconstruction.

Observers for monitoring carrier dynamics and reconstructing images
from manifold state in cross-modal experiments.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from sensorium.observers.types import ObserverProtocol


class CrossModalDynamicsObserver(ObserverProtocol):
    """Observer that tracks cross-modal dynamics over time."""
    
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.step_count = 0
        self.history = {
            "step": [],
            "n_modes": [],
            "n_crystallized": [],
            "mean_energy": [],
        }
    
    def observe(self, observation=None, **kwargs) -> Dict[str, Any]:
        self.step_count += 1
        
        if observation is not None:
            data = observation.data if hasattr(observation, "data") else observation
            amplitudes = data.get("amplitudes")
            mode_state = data.get("mode_state")
            
            if amplitudes is not None:
                active = amplitudes > 1e-6
                n_modes = int(active.sum().item())
                n_crystallized = 0
                if mode_state is not None and n_modes > 0:
                    n_crystallized = int((mode_state[:n_modes] == 2).sum().item())
            else:
                n_modes = 0
                n_crystallized = 0
            
            self.history["step"].append(self.step_count)
            self.history["n_modes"].append(n_modes)
            self.history["n_crystallized"].append(n_crystallized)
        
        return {"done_thinking": self.step_count >= self.max_steps}


class ImageReconstructor(ObserverProtocol):
    """Observer that reconstructs image from manifold state."""
    
    def observe(self, observation=None, **kwargs) -> Dict[str, Any]:
        if observation is None:
            return {}
        
        data = observation.data if hasattr(observation, "data") else observation
        
        state = data.get("state", {})
        metadata = data.get("metadata", {})
        n_image_bytes = data.get("n_image_bytes", 0)
        original = data.get("original", None)
        
        reconstructed = self._reconstruct_image(state, metadata, n_image_bytes)
        
        # Compute metrics
        if original is not None:
            mse = float(np.mean((original - reconstructed) ** 2))
            psnr = float(10 * np.log10(1.0 / (mse + 1e-10))) if mse > 0 else 100.0
        else:
            mse = 0.0
            psnr = 0.0
        
        return {
            "reconstructed": reconstructed,
            "mse": mse,
            "psnr": psnr,
        }
    
    def _reconstruct_image(self, state: Dict, metadata: Dict, n_image_bytes: int) -> np.ndarray:
        """Reconstruct image from manifold state."""
        H, W = metadata.get("shape", (32, 32))
        topk_indices = metadata.get("topk_indices", np.array([]))
        mag_selected = metadata.get("mag_selected", np.array([]))
        phase_selected = metadata.get("phase_selected", np.array([]))
        
        osc_energy = state.get("osc_energy")
        n_freq = min(len(topk_indices), n_image_bytes // 2)
        
        if osc_energy is not None and len(osc_energy) >= n_image_bytes:
            image_energies = osc_energy[:n_image_bytes].cpu().numpy()
            energy_weights = np.array([image_energies[i * 2] for i in range(n_freq)])
            if energy_weights.max() > 0:
                energy_weights = energy_weights / energy_weights.max()
            else:
                energy_weights = np.ones(n_freq)
        else:
            energy_weights = np.ones(n_freq)
        
        spectrum = np.zeros((H, W), dtype=np.complex64)
        
        for i, idx in enumerate(topk_indices[:n_freq]):
            mag = mag_selected[i] * energy_weights[i]
            phase = phase_selected[i]
            u = idx // W
            v = idx % W
            if 0 <= u < H and 0 <= v < W:
                spectrum[u, v] = mag * np.exp(1j * phase)
        
        spectrum_unshifted = np.fft.ifftshift(spectrum)
        reconstructed = np.fft.ifft2(spectrum_unshifted).real
        
        rmin, rmax = reconstructed.min(), reconstructed.max()
        if rmax > rmin:
            reconstructed = (reconstructed - rmin) / (rmax - rmin)
        else:
            reconstructed = np.clip(reconstructed, 0, 1)
        
        return reconstructed.astype(np.float32)
