from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.semantic.hierarchical import HierarchicalSemanticManifold


def test_multi_resolution_nucleation_creates_variable_length_chunks() -> None:
    device = torch.device("cpu")
    cfg = PhysicsConfig(dt=1e-2, tau=1.0)
    vocab = ["<bos>", "a", "b", "c", "<eos>"]
    brain = HierarchicalSemanticManifold(cfg, device, vocab=vocab, embed_dim=5, chunk_min_len=2, chunk_max_len=4)

    # Build a context long enough for 2/3/4 candidates.
    ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    brain.ingest_ids(ids)

    # Seed strong bigram bonds for the last 4-tokens: a->b, b->c, c-><eos> missing so 4-gram should not win.
    a, b, c = 1, 2, 3
    brain.graph.add_edges(torch.tensor([a, b], device=device), torch.tensor([b, c], device=device), torch.tensor(5.0, device=device))

    # Force baselines low so condensation can occur on first call.
    brain.chunks._binding_baseline_by_len[2] = torch.tensor(0.0, device=device)
    brain.chunks._binding_baseline_by_len[3] = torch.tensor(0.0, device=device)
    brain.chunks._binding_baseline_by_len[4] = torch.tensor(0.0, device=device)

    ratio = brain._homeostasis_ratio().to(torch.float32)
    brain._chunk_condensation(ratio=ratio)

    assert brain.chunks.num_chunks >= 1
    # Stored sequences are variable-length, tracked by seq_len.
    assert brain.chunks.seq_len.numel() == brain.chunks.num_chunks
