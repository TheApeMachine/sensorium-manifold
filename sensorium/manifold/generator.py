import torch
from typing import Dict, Optional


class SyntheticDataGenerator:
    """Generates synthetic 'files' - bursts of related particles.
    
    Each 'file' represents a coherent input (like an image, text chunk, etc.)
    with particles that have related positions and energies.
    """
    
    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        grid_size: tuple[int, int, int] = (32, 32, 32),
    ):
        self.device = device
        self.dtype = dtype
        self.grid_size = grid_size
        self.file_count = 0
        
        # Pattern types for variety
        self.patterns = ["cluster", "line", "sphere", "random", "grid"]
    
    def generate_file(
        self,
        num_particles: int = 50,
        pattern: Optional[str] = None,
        energy_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Generate a synthetic 'file' - a burst of related particles.
        
        Returns dict with positions, energies, etc. for new particles.
        """
        import random
        
        self.file_count += 1
        pattern = pattern or random.choice(self.patterns)
        
        gx, gy, gz = self.grid_size
        grid_center = torch.tensor([gx/2, gy/2, gz/2], device=self.device, dtype=self.dtype)
        
        if pattern == "cluster":
            # Tight cluster around a random point
            center = torch.rand(3, device=self.device, dtype=self.dtype) * torch.tensor(
                [gx * 0.6, gy * 0.6, gz * 0.6], device=self.device, dtype=self.dtype
            ) + torch.tensor([gx * 0.2, gy * 0.2, gz * 0.2], device=self.device, dtype=self.dtype)
            spread = min(gx, gy, gz) * 0.15
            positions = center + torch.randn(num_particles, 3, device=self.device, dtype=self.dtype) * spread
            
        elif pattern == "line":
            # Particles along a random line
            start = torch.rand(3, device=self.device, dtype=self.dtype) * torch.tensor(
                [gx, gy, gz], device=self.device, dtype=self.dtype
            ) * 0.3
            direction = torch.randn(3, device=self.device, dtype=self.dtype)
            direction = direction / (direction.norm() + 1e-8)
            t = torch.linspace(0, min(gx, gy, gz) * 0.7, num_particles, device=self.device, dtype=self.dtype)
            positions = start + t.unsqueeze(1) * direction
            # Add some noise
            positions += torch.randn(num_particles, 3, device=self.device, dtype=self.dtype) * 0.5
            
        elif pattern == "sphere":
            # Hollow sphere shell
            center = grid_center + torch.randn(3, device=self.device, dtype=self.dtype) * 3
            radius = min(gx, gy, gz) * 0.2 + torch.rand(1).item() * 5
            # Random points on unit sphere
            theta = torch.rand(num_particles, device=self.device, dtype=self.dtype) * 2 * 3.14159
            phi = torch.acos(2 * torch.rand(num_particles, device=self.device, dtype=self.dtype) - 1)
            x = torch.sin(phi) * torch.cos(theta)
            y = torch.sin(phi) * torch.sin(theta)
            z = torch.cos(phi)
            positions = center + radius * torch.stack([x, y, z], dim=1)
            
        elif pattern == "grid":
            # Regular grid pattern
            side = int(num_particles ** (1/3)) + 1
            coords = torch.stack(torch.meshgrid(
                torch.linspace(2, gx-2, side, device=self.device),
                torch.linspace(2, gy-2, side, device=self.device),
                torch.linspace(2, gz-2, side, device=self.device),
                indexing='ij'
            ), dim=-1).reshape(-1, 3)[:num_particles]
            positions = coords + torch.randn(len(coords), 3, device=self.device, dtype=self.dtype) * 0.3
            # Pad if needed
            if len(positions) < num_particles:
                extra = num_particles - len(positions)
                positions = torch.cat([positions, torch.rand(extra, 3, device=self.device, dtype=self.dtype) * 
                                       torch.tensor([gx, gy, gz], device=self.device, dtype=self.dtype)])
            
        else:  # random
            positions = torch.rand(num_particles, 3, device=self.device, dtype=self.dtype) * torch.tensor(
                [gx - 2, gy - 2, gz - 2], device=self.device, dtype=self.dtype
            ) + 1
        
        # Clamp to valid range
        positions = positions.clamp(0.5, min(gx, gy, gz) - 1.5)
        
        # Generate energies - higher near the pattern center
        if pattern in ["cluster", "sphere"]:
            center = positions.mean(dim=0)
            dist_from_center = (positions - center).norm(dim=1)
            max_dist = dist_from_center.max() + 1e-8
            energies = (1 - dist_from_center / max_dist) * energy_scale + 0.1
        else:
            energies = torch.rand(num_particles, device=self.device, dtype=self.dtype) * energy_scale * 0.5 + 0.5
        
        # Small initial velocities pointing toward center
        center = positions.mean(dim=0)
        velocities = (center - positions) * 0.01
        velocities += torch.randn_like(velocities) * 0.05
        
        return {
            "positions": positions,
            "velocities": velocities,
            "energies": energies,
            "heats": torch.zeros(num_particles, device=self.device, dtype=self.dtype),
            "excitations": torch.rand(num_particles, device=self.device, dtype=self.dtype) * 0.1,
            "masses": energies.clone(),
            "pattern": pattern,
            "file_id": self.file_count,
        }
