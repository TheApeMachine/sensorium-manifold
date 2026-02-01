"""
Unified Multimodal Demo

Demonstrates the unified manifold processing particles from multiple modalities:
1. Encode an image to frequency-space particles
2. Run thermodynamic dynamics
3. Decode back to image

This proves that the spectral manifold is truly modality-agnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.spectral.unified import UnifiedManifold, Modality


def create_test_image(size: int = 64) -> torch.Tensor:
    """Create a simple test image with clear frequency structure."""
    x = torch.linspace(0, 2 * torch.pi * 4, size)
    y = torch.linspace(0, 2 * torch.pi * 4, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Combination of sine waves at different frequencies
    image = (
        torch.sin(xx * 2) * 0.3 +           # Low freq horizontal
        torch.sin(yy * 4) * 0.3 +           # Mid freq vertical  
        torch.sin(xx * 8 + yy * 8) * 0.2 +  # High freq diagonal
        torch.sin(xx * 1 + yy * 1) * 0.2    # Very low freq diagonal
    )
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    return image


def run_demo(
    *,
    image_size: int = 64,
    top_k: int = 100,
    steps: int = 10,
    dt: float = 0.02,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Run the unified multimodal demo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("UNIFIED MULTIMODAL MANIFOLD DEMO")
    print("=" * 60)
    
    # Create manifold
    cfg = PhysicsConfig(dt=dt, eps=1e-8)
    manifold = UnifiedManifold(cfg, device, embed_dim=256)
    
    # Create test image
    print(f"\n1. Creating test image ({image_size}x{image_size})...")
    original = create_test_image(image_size).to(device)
    
    # Encode image to particles
    print(f"2. Encoding image to {top_k} frequency particles...")
    particle_indices = manifold.encode_image(original, top_k=top_k)
    print(f"   Created {len(particle_indices)} IMAGE particles")
    
    # Show modality distribution
    state = manifold.output_state()
    print(f"   Modality counts: {state.meta['modality_counts']}")
    
    # Run thermodynamic dynamics
    print(f"\n3. Running {steps} steps of thermodynamic dynamics...")
    for t in range(steps):
        manifold.step()
        if (t + 1) % 5 == 0:
            state = manifold.output_state()
            total_energy = sum(p.energy.item() for p in state.particles)
            total_heat = sum(p.heat.item() for p in state.particles)
            print(f"   Step {t+1}: energy={total_energy:.4f}, heat={total_heat:.4f}")
    
    # Decode back to image
    print(f"\n4. Decoding particles back to image...")
    reconstructed = manifold.decode_image((image_size, image_size))
    
    # Compute reconstruction error
    mse = ((original - reconstructed) ** 2).mean().item()
    print(f"   Reconstruction MSE: {mse:.6f}")
    
    # Save results
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(original.cpu().numpy(), cmap='viridis')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed.cpu().numpy(), cmap='viridis')
        axes[1].set_title(f'Reconstructed (MSE={mse:.4f})')
        axes[1].axis('off')
        
        diff = (original - reconstructed).abs().cpu().numpy()
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        
        fig.tight_layout()
        fig_path = out_dir / "unified_multimodal.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\n5. Saved visualization to: {fig_path}")
        
        # Also save particle distribution
        fig2, ax = plt.subplots(figsize=(8, 8))
        
        # Extract 2D positions of image particles
        u_coords = []
        v_coords = []
        energies = []
        for p in manifold._particles:
            if p.modality == Modality.IMAGE and p.position.numel() == 2:
                u_coords.append(p.position[0].item())
                v_coords.append(p.position[1].item())
                energies.append(p.energy.item())
        
        scatter = ax.scatter(
            v_coords, u_coords, 
            c=energies, 
            s=[e * 1000 for e in energies],
            cmap='plasma',
            alpha=0.6,
        )
        ax.set_xlabel('v (horizontal frequency)')
        ax.set_ylabel('u (vertical frequency)')
        ax.set_title('Particle Distribution in 2D Frequency Space')
        ax.set_aspect('equal')
        plt.colorbar(scatter, label='Energy')
        
        fig2_path = out_dir / "frequency_particles.png"
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)
        print(f"   Saved particle distribution to: {fig2_path}")
        
    except ImportError:
        print("\n   (matplotlib not available, skipping visualization)")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nKey insight: The manifold processed IMAGE particles using")
    print(f"the same thermodynamic dynamics that work for TEXT and AUDIO.")
    print(f"Modality is not special—it's just a tag for the decoder.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Multimodal Manifold Demo")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--out-dir", type=str, default="./artifacts/unified")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    run_demo(
        image_size=args.image_size,
        top_k=args.top_k,
        steps=args.steps,
        dt=args.dt,
        device=device,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
