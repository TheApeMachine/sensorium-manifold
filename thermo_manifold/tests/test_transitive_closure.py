from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.semantic.manifold import SemanticManifold


def test_transitive_closure_adds_shortcut_edge() -> None:
    device = torch.device("cpu")
    cfg = PhysicsConfig(dt=1e-2, tau=1.0)
    vocab = ["A", "B", "C"]
    brain = SemanticManifold(cfg, device, vocab=vocab, embed_dim=3)

    # Seed A->B and B->C edges.
    a, b, c = 0, 1, 2
    brain.graph.add_edges(torch.tensor([a], device=device), torch.tensor([b], device=device), torch.tensor(1.0, device=device))
    brain.graph.add_edges(torch.tensor([b], device=device), torch.tensor([c], device=device), torch.tensor(1.0, device=device))

    # Make A active so closure considers it.
    exc = brain.attractors.get("excitation")
    exc[a] = 1.0
    brain.attractors.set("excitation", exc)

    brain.idle_think(steps=3, dream_steps=2)

    w, _, exists = brain.graph.get_edges(torch.tensor([a], device=device), torch.tensor([c], device=device))
    assert bool(exists.item())
    assert float(w.item()) > 0.0
