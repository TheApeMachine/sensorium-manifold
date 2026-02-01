from __future__ import annotations

import math
from typing import List, Tuple

import torch

from thermo_manifold import BridgeManifold, PhysicsConfig, SemanticManifold, SpectralManifold


def make_vocab() -> List[str]:
    return ["the", "cat", "sat", "on", "mat", ".", "dog", "ran"]


def token_frequencies(vocab_size: int, *, device: torch.device) -> torch.Tensor:
    """Environment mapping: assign each token a distinct carrier frequency.

    This is not used by the learning rules directly; it only generates sensory co-occurrence
    (text token + audio frequency) to let the bridge bonds emerge.
    """
    # Frequencies span a range implied by the population statistics (no hand-tuned per-token values).
    idx = torch.arange(vocab_size, device=device, dtype=torch.float32)
    frac = idx / (vocab_size - 1 + 1e-8)
    # Use a log-scale spread so low and high are both represented.
    f_min = torch.tensor(110.0, device=device)  # A2 as a conventional audible anchor
    f_max = torch.tensor(1760.0, device=device)  # A6
    freqs = f_min * (f_max / f_min) ** frac
    return freqs


def train_ring_grammar(brain: SemanticManifold, seq: List[int], steps_per_context: int) -> None:
    """Train simple ring transitions by repeatedly stepping on short contexts."""
    device = brain.device
    seq_t = torch.tensor(seq, device=device, dtype=torch.long)

    for i in range(len(seq)):
        ctx = seq_t[i : i + 2]
        if ctx.numel() < 2:
            continue
        brain.ingest_ids(ctx)
        for _ in range(steps_per_context):
            brain.step_grammar()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhysicsConfig(dt=1e-2)

    vocab = make_vocab()
    brain = SemanticManifold(cfg, device, vocab=vocab, embed_dim=min(16, len(vocab)))

    # Train a ring: a->b->c->...
    ring = list(range(len(vocab)))
    steps_per_context = int(1.0 / cfg.dt)  # simulate ~1 unit time per observation
    train_ring_grammar(brain, ring + [ring[0]], steps_per_context=steps_per_context)

    # Bridge: semantic vector space -> spectral bins.
    freqs = token_frequencies(len(vocab), device=device)
    bridge = BridgeManifold(sem_dim=brain.embed_dim, spec_bins=freqs, dt=cfg.dt, device=device)

    # Co-activation training (no hard-coded mapping inside the bridge):
    # present each token's semantic embedding concurrently with its sensory frequency.
    emb = brain.attractors.get("position")
    for tid in range(len(vocab)):
        bridge.observe(
            sem_pos=emb[tid],
            spec_pos=freqs[tid : tid + 1],
        )

    # Inference: given a short context, predict next token and synthesize audio.
    context = torch.tensor([0, 1], device=device)  # "the cat"
    brain.ingest_ids(context)
    while not brain.thinking_complete():
        brain.step_grammar()
    out = brain.output_state()
    print("Predicted next token:", out.token, "(index", out.token_index, ")")
    print("Meta:", {k: round(v, 4) for k, v in out.meta.items() if isinstance(v, (int, float))})

    # Bridge the semantic state into spectral energy.
    # Use a continuous "thought" vector: E[pos] under the semantic distribution.
    sem_vec = out.probs @ brain.attractors.get("position")
    b_out = bridge.forward(sem_vec)
    spec_energy = b_out.spec_probs

    voice = SpectralManifold(cfg, device)
    voice.set_targets(freqs, energy=spec_energy)
    voice.seed_particles(n=int(1.0 / cfg.dt))  # particle budget proportional to simulated time

    # Diffuse for ~1 simulated time unit.
    steps = int(1.0 / cfg.dt)
    for _ in range(steps):
        voice.step_physics()

    audio = voice.output_state(topk=5)
    print("Top synthesized frequencies (Hz):", [round(float(f), 2) for f in audio.frequencies.tolist()])
    print("Amplitudes:", [round(float(a), 4) for a in audio.amplitudes.tolist()])
    print("Bridge meta:", b_out.meta)


if __name__ == "__main__":
    main()
