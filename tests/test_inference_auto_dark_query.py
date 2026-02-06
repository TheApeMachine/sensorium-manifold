from __future__ import annotations

import pytest


def test_inference_observer_auto_dark_query_injection_and_cleanup():
    torch = pytest.importorskip("torch")

    from sensorium.observers.inference import InferenceConfig, InferenceObserver
    from sensorium.observers.types import PARTICLE_FLAG_DARK

    class QueryObserver:
        def resolve_dark_query(self, state: dict, metadata: dict | None = None):
            md = metadata or {}
            return md.get("test_bytes")

        def observe(self, state: dict, **kwargs):
            flags = state.get("particle_flags")
            if flags is None:
                return {"n_dark_visible_during_observe": 0}
            n_dark = int(((flags & PARTICLE_FLAG_DARK) != 0).sum().item())
            return {"n_dark_visible_during_observe": n_dark}

    state = {
        "positions": torch.zeros((2, 3), dtype=torch.float32),
        "velocities": torch.zeros((2, 3), dtype=torch.float32),
        "energies": torch.ones((2,), dtype=torch.float32),
        "heats": torch.zeros((2,), dtype=torch.float32),
        "excitations": torch.zeros((2,), dtype=torch.float32),
        "masses": torch.ones((2,), dtype=torch.float32),
        "token_ids": torch.tensor([1, 2], dtype=torch.int64),
        "byte_values": torch.tensor([1, 2], dtype=torch.int64),
        "sequence_indices": torch.tensor([0, 1], dtype=torch.int64),
        "sample_indices": torch.tensor([0, 0], dtype=torch.int64),
    }

    inference = InferenceObserver(
        [QueryObserver()],
        config=InferenceConfig(
            steps=0,
            cleanup_mode="decay",
            observer_query_cleanup_mode="immediate",
        ),
    )

    n_before = int(state["positions"].shape[0])
    result = inference.observe(state, test_bytes=b"AB")

    assert result["dark_query_source"] == "observer"
    assert result["n_dark_injected"] == 2
    assert result["n_dark_visible_during_observe"] == 2
    assert int(state["positions"].shape[0]) == n_before
