"""Image inpainting observer using dual-domain inference.

Reconstructs masked pixels using carrier patterns from the manifold.
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import torch

from sensorium.observers.types import ObserverProtocol


MNIST_IMAGE_SIZE = 28 * 28


class ImageInpainter(ObserverProtocol):
    """Reconstruct masked pixels using dual-domain inference.
    
    For images, leverages:
    - Geometric: spatial locality (nearby pixels cluster together)
    - Spectral: carriers couple similar pixels at same positions
    """
    
    def __init__(self, vocab_size: int = 4096, prime: int = 31, image_size: int = 784):
        self.vocab_size = vocab_size
        self.prime = prime
        self.mask = vocab_size - 1
        self.image_size = image_size
        self.image_width = int(np.sqrt(image_size))
        self.inference = None
    
    def learn_from_manifold(self, geo_state: Dict, spec_state: Dict):
        """Set up dual-domain inference from manifold state."""
        from sensorium.observers.dual_domain import DualDomainInference
        
        self.inference = DualDomainInference(
            geometric_state=geo_state,
            spectral_state=spec_state,
            vocab_size=self.vocab_size,
            prime=self.prime,
        )
    
    def observe(self, observation=None, **kwargs) -> Dict[str, Any]:
        """Inpaint masked image."""
        if observation is None:
            return {}
        
        data = observation.data if hasattr(observation, "data") else observation
        
        corrupted = data.get("corrupted", b"")
        mask_positions = data.get("mask_positions", [])
        original = data.get("original", b"")
        
        if not corrupted or not mask_positions:
            return {}
        
        reconstructed = self.inpaint(corrupted, mask_positions)
        
        # Calculate metrics
        original_np = np.array(list(original), dtype=np.float32)
        recon_np = np.array(list(reconstructed), dtype=np.float32)
        
        mae = float(np.mean(np.abs(original_np - recon_np)))
        mse = float(np.mean((original_np - recon_np) ** 2))
        psnr = float(10 * np.log10(255**2 / (mse + 1e-10)))
        
        return {
            "reconstructed": reconstructed,
            "mae": mae,
            "mse": mse,
            "psnr": psnr,
        }
    
    def inpaint(self, corrupted: bytes, mask_positions: List[int]) -> bytes:
        """Reconstruct masked pixels using dual-domain inference."""
        result = bytearray(corrupted)
        
        if self.inference is None:
            return bytes(result)
        
        for pos in mask_positions:
            neighbors = self._get_neighbors(pos)
            context_positions = [n for n in neighbors if n not in mask_positions]
            
            if context_positions:
                context_indices = torch.tensor(
                    context_positions,
                    device=self.inference.device,
                    dtype=torch.int64
                )
                
                carrier_scores = self.inference.score_candidate_bytes(
                    context_indices=context_indices,
                    target_position=pos % self.image_size,
                    segment_size=self.image_size,
                )
            else:
                carrier_scores = np.ones(256, dtype=np.float32) / 256
            
            # Spatial smoothness prior
            neighbor_scores = np.zeros(256, dtype=np.float32)
            for npos in neighbors:
                if npos not in mask_positions and 0 <= npos < len(result):
                    neighbor_val = result[npos]
                    for pval in range(256):
                        diff = abs(pval - neighbor_val)
                        neighbor_scores[pval] += np.exp(-diff**2 / (2 * 30**2))
            
            if neighbor_scores.sum() > 0:
                neighbor_scores /= neighbor_scores.sum()
            
            combined_scores = 0.6 * carrier_scores + 0.4 * neighbor_scores
            result[pos] = int(np.argmax(combined_scores))
        
        return bytes(result)
    
    def _get_neighbors(self, pos: int) -> List[int]:
        """Get 4-connected neighbors in image."""
        row = pos // self.image_width
        col = pos % self.image_width
        neighbors = []
        
        if row > 0:
            neighbors.append((row - 1) * self.image_width + col)
        if row < self.image_width - 1:
            neighbors.append((row + 1) * self.image_width + col)
        if col > 0:
            neighbors.append(row * self.image_width + (col - 1))
        if col < self.image_width - 1:
            neighbors.append(row * self.image_width + (col + 1))
        
        return neighbors
    
    def evaluate_batch(
        self,
        test_images: List[bytes],
        test_labels: List[int],
        mask_fracs: List[float],
        seed: int = 42,
        max_examples_per_level: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate inpainting on a batch of test images at multiple mask levels.
        
        Args:
            test_images: List of original images as bytes.
            test_labels: List of label indices.
            mask_fracs: List of mask fractions to test (e.g. [0.1, 0.2, 0.3]).
            seed: Random seed for mask generation.
            max_examples_per_level: Max examples to save per mask level.
        
        Returns:
            Dict with 'mask_results' and 'examples' for projectors.
        """
        rng = np.random.RandomState(seed)
        mask_results: Dict[float, Dict[str, Any]] = {}
        examples: List[Dict[str, Any]] = []
        
        for mask_frac in mask_fracs:
            all_maes = []
            all_mses = []
            all_psnrs = []
            
            for img_idx, img in enumerate(test_images):
                n_mask = int(self.image_size * mask_frac)
                mask_positions = list(rng.choice(self.image_size, size=n_mask, replace=False))
                
                masked = bytearray(img)
                for pos in mask_positions:
                    masked[pos] = 128
                
                result = self.observe({
                    "corrupted": bytes(masked),
                    "mask_positions": mask_positions,
                    "original": img,
                })
                
                reconstructed = result.get("reconstructed", bytes(masked))
                mae = result.get("mae", 0)
                mse = result.get("mse", 0)
                psnr = result.get("psnr", 0)
                
                all_maes.append(mae)
                all_mses.append(mse)
                all_psnrs.append(psnr)
                
                # Save examples (up to max per mask level)
                n_examples_at_level = len([e for e in examples if e.get("mask_frac") == mask_frac])
                if n_examples_at_level < max_examples_per_level:
                    examples.append({
                        "original": img,
                        "masked": bytes(masked),
                        "reconstructed": reconstructed,
                        "mask_frac": mask_frac,
                        "mae": mae,
                        "psnr": psnr,
                        "label": test_labels[img_idx] if img_idx < len(test_labels) else -1,
                    })
            
            mask_results[mask_frac] = {
                "mae": float(np.mean(all_maes)),
                "mse": float(np.mean(all_mses)),
                "psnr": float(np.mean(all_psnrs)),
                "n_images": len(test_images),
            }
        
        return {
            "mask_results": mask_results,
            "examples": examples,
        }
