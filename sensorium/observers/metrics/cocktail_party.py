"""Cocktail party separation observer using spectral clustering.

Clusters frequency bins to separate different speakers.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from sensorium.observers.types import ObserverProtocol


class SpectralClusteringObserver(ObserverProtocol):
    """Observer that clusters frequency bins for speaker separation."""
    
    def __init__(self, num_speakers: int = 2, n_iterations: int = 30):
        self.num_speakers = num_speakers
        self.n_iterations = n_iterations
    
    def observe(self, observation=None, **kwargs) -> Dict[str, Any]:
        """Cluster frequency bins and return labels."""
        if observation is None:
            return {}
        
        data = observation.data if hasattr(observation, "data") else observation
        
        frequencies = data.get("frequencies")
        energies = data.get("energies")
        
        if frequencies is None or energies is None:
            return {}
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        freq_tensor = torch.tensor(frequencies, dtype=torch.float32, device=device)
        energy_tensor = torch.tensor(energies, dtype=torch.float32, device=device)
        
        x = freq_tensor.view(-1, 1)
        
        # Initialize centroids
        sorted_x, _ = torch.sort(x.squeeze())
        q1_idx = len(sorted_x) // 4
        q3_idx = 3 * len(sorted_x) // 4
        centroids = torch.tensor(
            [[sorted_x[q1_idx].item()], [sorted_x[q3_idx].item()]], 
            device=device
        )
        
        # K-means iterations
        for _ in range(self.n_iterations):
            dists = torch.abs(x - centroids.T)
            labels = torch.argmin(dists, dim=1)
            
            new_centroids = []
            for i in range(self.num_speakers):
                mask = (labels == i)
                if mask.sum() > 0:
                    weights = energy_tensor[mask]
                    weighted_mean = (x[mask].squeeze() * weights).sum() / (weights.sum() + 1e-10)
                    new_centroids.append(weighted_mean)
                else:
                    new_centroids.append(centroids[i, 0])
            centroids = torch.stack(new_centroids).view(self.num_speakers, 1)
        
        labels_np = labels.cpu().numpy()
        
        # Compute cluster statistics
        cluster_counts = [(labels_np == i).sum() for i in range(self.num_speakers)]
        cluster_means = [
            float(frequencies[labels_np == i].mean()) if cluster_counts[i] > 0 else 0 
            for i in range(self.num_speakers)
        ]
        cluster_energy = [
            float(energies[labels_np == i].sum()) if cluster_counts[i] > 0 else 0 
            for i in range(self.num_speakers)
        ]
        
        # Separation score
        inter_dist = abs(cluster_means[0] - cluster_means[1])
        intra_dists = []
        for i in range(self.num_speakers):
            if cluster_counts[i] > 1:
                cluster_freqs = frequencies[labels_np == i]
                intra_dists.append(float(np.std(cluster_freqs)))
        mean_intra = np.mean(intra_dists) if intra_dists else 0.01
        separation_score = float(inter_dist / (mean_intra + 1e-6))
        
        return {
            "labels": labels_np,
            "cluster_counts": cluster_counts,
            "cluster_means": cluster_means,
            "cluster_energy": cluster_energy,
            "separation_score": separation_score,
        }
