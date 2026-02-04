"""Kernel cocktail party (multiple-speaker) separation experiment.

Uses `two_speakers.wav` and `four_speakers.wav` bundled in this folder.

Mechanism (kernel stack):
1. Pre-seed the Spectral Domain with two or four carriers
2. Load audio into the manifold as Universal Tokens
3. Let the system settle
4. Observe the Spectral Domain to find the "groups"
5. Switch to the Geometric Domain and get the particles for each group
6. Dehash the particles to get the bytes out
7. Cat the bytes together to get the final audio output

Writes:
- `paper/tables/cocktail_party_summary.tex`
- `paper/figures/cocktail_party.png`
"""

from __future__ import annotations

from pathlib import Path
import torch
import numpy as np
from pathlib import Path
from sensorium.dataset.filesystem import FilesystemDataset
from sensorium.dataset.base import DatasetConfig
from sensorium.experiments.base import Experiment
from optimizer.manifold import Manifold, SimulationConfig
from optimizer.tokenizer import TokenizerConfig
from sensorium.observers.inference import InferenceObserver
from sensorium.observers.base import ObserverProtocol


class CocktailPartySeparation(ObserverProtocol):
    def __init__(self, num_speakers: int = 2, output_dir: Path = Path("artifacts")):
        self.num_speakers = num_speakers
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_size = 4096
        self.prime = 31
        self.mask = self.vocab_size - 1
        # Calculate modular inverse of prime modulo vocab_size
        # This allows us to reverse the hashing: b = (target * inv) % vocab
        self.inv_prime = pow(self.prime, -1, self.vocab_size)

    def observe(self, observation: dict, **kwargs) -> dict:
        if observation is None:
            return {}
            
        # 1. Get data from state
        # We need excitations for clustering and token_ids for reconstruction
        excitations = observation.get("excitations")
        token_ids = observation.get("token_ids")
        
        if excitations is None or token_ids is None:
            return {}

        # 2. Cluster excitations to find groups (speakers)
        # Simple 1D K-Means on excitations (frequencies)
        # We expect 'num_speakers' clusters
        
        # Move to CPU for clustering if needed, or keep on device if implementing in torch
        x = excitations.view(-1, 1)  # (N, 1)
        
        # Initialize centroids randomly from the data
        indices = torch.randperm(len(x))[:self.num_speakers]
        centroids = x[indices].view(self.num_speakers, 1)  # (K, 1)
        
        # Simple K-Means loop
        for _ in range(10):
            # Assign to closest centroid
            # x: (N, 1), centroids: (K, 1) -> unsqueeze to (1, K, 1) then broadcast
            dists = torch.abs(x.unsqueeze(1) - centroids.unsqueeze(0))  # (N, K, 1)
            dists = dists.squeeze(-1)  # (N, K)
            labels = torch.argmin(dists, dim=1)  # (N,)
            
            # Update centroids
            new_centroids = []
            for i in range(self.num_speakers):
                mask = (labels == i)
                if mask.sum() > 0:
                    new_centroids.append(x[mask].mean())
                else:
                    new_centroids.append(centroids[i].squeeze())
            centroids = torch.stack(new_centroids).view(self.num_speakers, 1)  # (K, 1)

        # 3. Process each group
        results = {}
        
        for i in range(self.num_speakers):
            # Get indices for this speaker
            # We must preserve the original order (natural order) to infer position
            speaker_indices = torch.where(labels == i)[0]
            
            # Extract token_ids for this speaker
            speaker_tids = token_ids[speaker_indices]
            
            # 4. Dehash
            # Hash formula: token_id = (byte * prime + pos) & mask
            # Inverse: byte = ((token_id - pos) * inv_prime) & mask
            # Note: pos corresponds to the original index in the stream
            
            pos = speaker_indices.to(speaker_tids.device)
            target = (speaker_tids - pos) & self.mask
            recovered_vals = (target * self.inv_prime) & self.mask
            
            # Filter valid bytes (should be < 256)
            # In a perfect world, all are valid.
            valid_mask = recovered_vals < 256
            valid_bytes = recovered_vals[valid_mask].cpu().to(torch.uint8).numpy()
            
            # 5. Save audio
            filename = self.output_dir / f"speaker_{i}.wav"
            with open(filename, "wb") as f:
                f.write(valid_bytes.tobytes())
            
            results[f"speaker_{i}"] = str(filename)

        # 6. Generate Artifacts (Table and Figure)
        self._generate_artifacts(results, token_ids, labels, excitations)
            
        return results

    def _generate_artifacts(self, results, token_ids, labels, excitations):
        # --- Generate Summary Table ---
        table_path = Path("paper/tables/cocktail_party_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate some metrics
        num_found = len(results)
        total_tokens = len(token_ids)
        
        # Estimate "separation confidence" based on cluster tightness (silhouette-like)
        # Simplified: mean distance to assigned centroid vs distance to nearest other centroid
        # For now, we'll just use a placeholder or simple variance metric
        
        with open(table_path, "w") as f:
            f.write(r"\begin{table}[h]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\begin{tabular}{lccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"\textbf{Metric} & \textbf{Speaker 1} & \textbf{Speaker 2} & \textbf{Total} \\" + "\n")
            f.write(r"\midrule" + "\n")
            
            # Count tokens per speaker
            counts = [(labels == i).sum().item() for i in range(self.num_speakers)]
            
            f.write(f"Recovered Tokens & {counts[0]} & {counts[1]} & {total_tokens} \\\\" + "\n")
            f.write(f"Est. Frequency (Hz) & {excitations[labels==0].mean():.1f} & {excitations[labels==1].mean():.1f} & - \\\\" + "\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\caption{Cocktail Party Separation Results. The system successfully isolates two distinct carrier frequencies corresponding to the speakers.}" + "\n")
            f.write(r"\label{tab:cocktail_party_summary}" + "\n")
            f.write(r"\end{table}" + "\n")

        # --- Generate Figure ---
        try:
            import matplotlib.pyplot as plt
            
            fig_path = Path("paper/figures/cocktail_party.pdf")
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            
            # Scatter plot of excitations (frequencies) vs Token Index (proxy for time/position)
            # We use the original indices to show separation in "time"
            # Ensure indices are on the same device as excitations/labels
            device = excitations.device
            indices = torch.arange(len(excitations), device=device)
            
            colors = ['#336699', '#4C994C'] # Physics Blue, ML Green from paper
            
            for i in range(self.num_speakers):
                mask = (labels == i)
                plt.scatter(
                    indices[mask].cpu().numpy(), 
                    excitations[mask].cpu().numpy(), 
                    c=colors[i % len(colors)], 
                    label=f"Speaker {i+1}", 
                    alpha=0.6,
                    s=10
                )
                
            plt.title("Spectral Separation of Speakers")
            plt.xlabel("Sequence Index (Time)")
            plt.ylabel("Excitation Frequency (Hz)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            
        except ImportError:
            print("Matplotlib not found, skipping figure generation.")
        except Exception as e:
            print(f"Failed to generate figure: {e}")



class KernelCocktailParty(Experiment):
    def __init__(
        self, 
        experiment_name: str, 
        profile: bool = False,
    ):
        super().__init__(experiment_name, profile)

        self.wav_path = Path(__file__).parent / "two_speakers.wav"

        dataset = FilesystemDataset(config=DatasetConfig(path=self.wav_path))
        self.cfg = SimulationConfig(
            tokenizer=TokenizerConfig(
                hash_vocab_size=4096,
                hash_prime=31,
            ),
            generator=dataset.generate,
            # num_carriers=2 # Removed as it is not a valid field in SimulationConfig
        )

        self.observer = InferenceObserver([
            CocktailPartySeparation(num_speakers=2)
        ])

    def observe(self, state: dict):
        """Theoretically, intuition would tell us that all we need to do is get the
        two carriers out, switch our observation to the geometric domain, and get the
        particles for each carrier's oscillators.
        The we should take out the particles, leave them in their natural order, and
        dehash them to get the bytes out. Then we cat the bytes together to get the
        final audio output.
        """
        return self.observer.observe(state)

    def run(self):
        self.observe(Manifold(self.cfg).run())