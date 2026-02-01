"""
Cross-Modal Demo

Demonstrates text and image particles living in the same unified manifold,
with thermodynamic dynamics operating on both simultaneously.

This proves the core claim: native multimodality where modality is emergent
from position in the space, not bolted on as separate modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.spectral.unified import UnifiedManifold, Modality


def create_stripe_image(size: int = 32, orientation: str = "horizontal") -> torch.Tensor:
    """Create a simple stripe pattern."""
    coords = torch.linspace(0, 2 * torch.pi * 4, size)
    rows, cols = torch.meshgrid(coords, coords, indexing='ij')
    # rows varies along vertical axis (row index)
    # cols varies along horizontal axis (column index)
    
    if orientation == "horizontal":
        # Horizontal stripes = bands that run left-right = vary with row (vertical position)
        image = torch.sin(rows * 4)
    elif orientation == "vertical":
        # Vertical stripes = bands that run up-down = vary with column (horizontal position)
        image = torch.sin(cols * 4)
    elif orientation == "diagonal":
        image = torch.sin(rows * 4 + cols * 4)
    else:
        # Checkerboard
        image = torch.sin(rows * 4) * torch.sin(cols * 4)
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image


def create_simple_embeddings(vocab: List[str], embed_dim: int, device: torch.device) -> torch.Tensor:
    """Create simple word embeddings.
    
    In a real system, these would come from a pretrained model.
    Here we create structured embeddings where semantically related
    words are closer together.
    """
    embeddings = torch.zeros(len(vocab), embed_dim, device=device, dtype=torch.float32)
    
    # Semantic clusters (hand-designed for demo)
    # Pattern words cluster together
    pattern_words = {"stripes", "lines", "pattern", "bars", "waves"}
    # Direction words cluster together  
    direction_words = {"horizontal", "vertical", "diagonal"}
    # Shape words
    shape_words = {"grid", "checkerboard", "cross"}
    
    for i, word in enumerate(vocab):
        # Base: random direction
        torch.manual_seed(hash(word) % (2**32))
        base = torch.randn(embed_dim, device=device)
        base = base / (base.norm() + 1e-8)
        
        # Add cluster bias
        if word in pattern_words:
            bias = torch.zeros(embed_dim, device=device)
            bias[0:10] = 1.0  # Pattern words have high values in dims 0-10
            base = base + 0.5 * bias
        elif word in direction_words:
            bias = torch.zeros(embed_dim, device=device)
            bias[10:20] = 1.0  # Direction words in dims 10-20
            base = base + 0.5 * bias
        elif word in shape_words:
            bias = torch.zeros(embed_dim, device=device)
            bias[20:30] = 1.0  # Shape words in dims 20-30
            base = base + 0.5 * bias
        
        embeddings[i] = base / (base.norm() + 1e-8)
    
    return embeddings


def run_demo(
    *,
    image_size: int = 32,
    top_k_freq: int = 50,
    steps: int = 20,
    dt: float = 0.02,
    embed_dim: int = 64,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Run the cross-modal demo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CROSS-MODAL MANIFOLD DEMO")
    print("=" * 60)
    
    # Create manifold
    cfg = PhysicsConfig(dt=dt, eps=1e-8)
    manifold = UnifiedManifold(cfg, device, embed_dim=embed_dim)
    
    # =========================================================================
    # 1. Create vocabulary and embeddings
    # =========================================================================
    print("\n1. Setting up vocabulary and embeddings...")
    vocab = [
        "stripes", "lines", "pattern", "bars", "waves",
        "horizontal", "vertical", "diagonal",
        "grid", "checkerboard", "cross",
        "image", "picture", "visual",
    ]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    embeddings = create_simple_embeddings(vocab, embed_dim, device)
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Embedding dim: {embed_dim}")
    
    # =========================================================================
    # 2. Encode text: "horizontal stripes"
    # =========================================================================
    print("\n2. Encoding text: 'horizontal stripes'...")
    text_tokens = ["horizontal", "stripes"]
    text_ids = [word_to_id[w] for w in text_tokens]
    text_embeddings = embeddings[text_ids]
    
    text_indices = manifold.encode_text(text_embeddings, token_ids=text_ids)
    print(f"   Created {len(text_indices)} TEXT particles")
    
    # =========================================================================
    # 3. Encode image: horizontal stripe pattern
    # =========================================================================
    print("\n3. Encoding image: horizontal stripe pattern...")
    image = create_stripe_image(image_size, orientation="horizontal").to(device)
    
    image_indices = manifold.encode_image(image, top_k=top_k_freq)
    print(f"   Created {len(image_indices)} IMAGE particles")
    
    # =========================================================================
    # 4. Show initial state
    # =========================================================================
    state = manifold.output_state()
    print(f"\n4. Manifold state (before dynamics):")
    print(f"   Total particles: {state.meta['num_particles']}")
    print(f"   By modality: {state.meta['modality_counts']}")
    
    # =========================================================================
    # 5. Run thermodynamic dynamics
    # =========================================================================
    print(f"\n5. Running {steps} steps of thermodynamic dynamics...")
    
    energy_history = []
    heat_history = []
    
    for t in range(steps):
        manifold.step()
        
        # Track metrics
        state = manifold.output_state()
        total_energy = sum(p.energy.item() for p in state.particles)
        total_heat = sum(p.heat.item() for p in state.particles)
        energy_history.append(total_energy)
        heat_history.append(total_heat)
        
        if (t + 1) % 5 == 0:
            print(f"   Step {t+1}: energy={total_energy:.4f}, heat={total_heat:.4f}")
    
    # =========================================================================
    # 6. Analyze cross-modal relationships
    # =========================================================================
    print("\n6. Analyzing cross-modal relationships...")
    
    # Get positions in common space for all particles
    text_positions = []
    image_positions = []
    
    for i, p in enumerate(manifold._particles):
        common_pos = manifold._to_common_space(p.position)
        if p.modality == Modality.TEXT:
            text_positions.append((i, common_pos))
        elif p.modality == Modality.IMAGE:
            image_positions.append((i, common_pos))
    
    # Compute distances between text and image particles
    if text_positions and image_positions:
        print(f"\n   Cross-modal distances (text <-> image):")
        for ti, (text_idx, text_pos) in enumerate(text_positions):
            token_id = manifold._particles[text_idx].token_id
            word = vocab[token_id] if token_id is not None else "?"
            
            # Find closest image particles
            distances = []
            for img_idx, img_pos in image_positions:
                dist = torch.linalg.norm(text_pos - img_pos).item()
                distances.append((img_idx, dist))
            
            distances.sort(key=lambda x: x[1])
            closest_dist = distances[0][1]
            mean_dist = sum(d for _, d in distances) / len(distances)
            
            print(f"   '{word}': closest_img_dist={closest_dist:.4f}, mean_img_dist={mean_dist:.4f}")
    
    # =========================================================================
    # 7. Decode back
    # =========================================================================
    print("\n7. Decoding outputs...")
    
    # Decode image
    reconstructed = manifold.decode_image((image_size, image_size))
    mse = ((image - reconstructed) ** 2).mean().item()
    print(f"   Image reconstruction MSE: {mse:.6f}")
    
    # Decode text
    decoded_text = manifold.decode_text(vocab, top_k=5)
    print(f"   Top decoded tokens: {decoded_text}")
    
    # =========================================================================
    # 8. Visualization
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        print("\n8. Generating visualizations...")
        
        # Figure 1: Image comparison
        fig1, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(image.cpu().numpy(), cmap='viridis')
        axes[0].set_title('Original (horizontal stripes)')
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed.cpu().numpy(), cmap='viridis')
        axes[1].set_title(f'Reconstructed (MSE={mse:.4f})')
        axes[1].axis('off')
        
        diff = (image - reconstructed).abs().cpu().numpy()
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        
        fig1.tight_layout()
        fig1.savefig(out_dir / "cross_modal_images.png", dpi=150)
        plt.close(fig1)
        print(f"   Saved: cross_modal_images.png")
        
        # Figure 2: Particle positions in common space (first 3 dims)
        fig2 = plt.figure(figsize=(10, 8))
        ax = fig2.add_subplot(111, projection='3d')
        
        text_coords = []
        text_labels = []
        image_coords = []
        image_energies = []
        
        for p in manifold._particles:
            common = manifold._to_common_space(p.position).cpu().numpy()
            if p.modality == Modality.TEXT:
                text_coords.append(common[:3])
                token_id = p.token_id
                text_labels.append(vocab[token_id] if token_id is not None else "?")
            elif p.modality == Modality.IMAGE:
                image_coords.append(common[:3])
                image_energies.append(p.energy.item())
        
        if image_coords:
            image_coords = torch.tensor(image_coords)
            ax.scatter(
                image_coords[:, 0], image_coords[:, 1], image_coords[:, 2],
                c=image_energies, cmap='Blues', alpha=0.3, s=20, label='Image frequencies'
            )
        
        if text_coords:
            text_coords = torch.tensor(text_coords)
            ax.scatter(
                text_coords[:, 0], text_coords[:, 1], text_coords[:, 2],
                c='red', s=200, marker='*', label='Text tokens'
            )
            for i, label in enumerate(text_labels):
                ax.text(text_coords[i, 0], text_coords[i, 1], text_coords[i, 2], 
                       f'  {label}', fontsize=12, color='red')
        
        ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 1')
        ax.set_zlabel('Dim 2')
        ax.set_title('Particles in Common Embedding Space (first 3 dims)')
        ax.legend()
        
        fig2.savefig(out_dir / "cross_modal_space.png", dpi=150)
        plt.close(fig2)
        print(f"   Saved: cross_modal_space.png")
        
        # Figure 3: Energy dynamics
        fig3, ax = plt.subplots(figsize=(8, 4))
        ax.plot(energy_history, label='Total Energy')
        ax.plot(heat_history, label='Total Heat')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title('Thermodynamic Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig3.savefig(out_dir / "cross_modal_dynamics.png", dpi=150)
        plt.close(fig3)
        print(f"   Saved: cross_modal_dynamics.png")
        
    except ImportError:
        print("\n   (matplotlib not available, skipping visualization)")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("""
KEY INSIGHTS:

1. Text particles ("horizontal", "stripes") and image particles
   (2D frequency components) coexist in the SAME manifold.

2. The thermodynamic dynamics operate on BOTH simultaneously,
   with no special handling for either modality.

3. Modality is just a tag for the decoder—the manifold itself
   is modality-agnostic.

4. Cross-modal relationships can emerge from proximity in the
   shared embedding space (with proper training/alignment).

This is NATIVE multimodality, not adapters bolted together.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Modal Manifold Demo")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--top-k-freq", type=int, default=50)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="./artifacts/cross_modal")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    run_demo(
        image_size=args.image_size,
        top_k_freq=args.top_k_freq,
        steps=args.steps,
        dt=args.dt,
        embed_dim=args.embed_dim,
        device=device,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
