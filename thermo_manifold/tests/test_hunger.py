from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.semantic.manifold import SemanticManifold


def test_hunger_increases_on_dead_end_dreams() -> None:
    device = torch.device("cpu")
    cfg = PhysicsConfig(dt=1e-2, tau=1.0)
    vocab = ["A", "B", "C"]
    brain = SemanticManifold(cfg, device, vocab=vocab, embed_dim=3)

    # No edges => immediate dead end.
    a = 0
    exc = brain.attractors.get("excitation")
    exc[a] = 1.0
    brain.attractors.set("excitation", exc)

    h0 = float(brain.hunger[a].item())
    brain.idle_think(steps=2, dream_steps=4)
    h1 = float(brain.hunger[a].item())
    assert h1 > h0
