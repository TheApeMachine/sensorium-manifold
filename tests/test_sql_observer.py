from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_sql_observer():
    path = Path(__file__).resolve().parents[1] / "sensorium" / "observers" / "sql.py"
    spec = importlib.util.spec_from_file_location("sql_observer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.SQLObserver, mod.SQLObserverConfig


def test_sql_observer_builds_simulation_view():
    SQLObserver, SQLObserverConfig = _load_sql_observer()

    state = {
        "byte_values": np.array([65, 66, 65, 66], dtype=np.int64),
        "sequence_indices": np.array([0, 1, 0, 1], dtype=np.int64),
        "sample_indices": np.array([0, 0, 1, 1], dtype=np.int64),
        "token_ids": np.array([(0 << 8) | 65, (1 << 8) | 66, (0 << 8) | 65, (1 << 8) | 66], dtype=np.int64),
        "masses": np.ones((4,), dtype=np.float32),
    }

    observer = SQLObserver(
        """
        SELECT n_particles, n_unique_tokens, n_transition_edges, total_mass
        FROM simulation;
        """,
        config=SQLObserverConfig(row_limit=1),
    )
    result = observer.observe(state)

    assert result["sql_n_particles"] == 4
    assert result["sql_n_unique_tokens"] == 2
    assert result["sql_n_transition_edges"] >= 1
    assert float(result["sql_total_mass"]) == 4.0


def test_sql_observer_transition_query_returns_rows():
    SQLObserver, SQLObserverConfig = _load_sql_observer()

    state = {
        "byte_values": np.array([65, 66, 65, 66, 65, 67], dtype=np.int64),
        "sequence_indices": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
        "sample_indices": np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        "token_ids": np.array([65, 322, 65, 322, 65, 323], dtype=np.int64),
        "masses": np.ones((6,), dtype=np.float32),
    }

    observer = SQLObserver(
        """
        SELECT src_byte, dst_byte, edge_count
        FROM transitions
        ORDER BY edge_count DESC, src_byte ASC, dst_byte ASC
        LIMIT 2;
        """,
        config=SQLObserverConfig(row_limit=2),
    )
    result = observer.observe(state)

    assert result["sql_row_count"] == 2
    top = result["sql_rows"][0]
    assert top["src_byte"] == 65
    assert top["dst_byte"] in (66, 67)
    assert top["edge_count"] >= 1


def test_sql_observer_ignores_dark_particles_in_simulation_view():
    SQLObserver, SQLObserverConfig = _load_sql_observer()

    state = {
        "byte_values": np.array([65, 66, 255], dtype=np.int64),
        "sequence_indices": np.array([0, 1, 0], dtype=np.int64),
        "sample_indices": np.array([0, 0, -1], dtype=np.int64),
        "token_ids": np.array([65, 322, 255], dtype=np.int64),
        "particle_flags": np.array([0, 0, 1], dtype=np.int64),  # last one is dark
        "masses": np.array([1.0, 1.0, 5.0], dtype=np.float32),
    }

    observer = SQLObserver(
        """
        SELECT n_particles, n_unique_tokens, total_mass
        FROM simulation;
        """,
        config=SQLObserverConfig(row_limit=1),
    )
    result = observer.observe(state)

    assert result["sql_n_particles"] == 2
    assert result["sql_n_unique_tokens"] == 2
    assert float(result["sql_total_mass"]) == 2.0


def test_sql_observer_resolves_query_payload_from_metadata_and_comments():
    SQLObserver, SQLObserverConfig = _load_sql_observer()

    observer = SQLObserver("SELECT * FROM simulation;", config=SQLObserverConfig())
    payload = observer.resolve_dark_query({}, metadata={"test_bytes": b"ABCD"})
    assert payload == b"ABCD"

    observer_with_comment = SQLObserver(
        """
        -- dark_query_text: hello
        SELECT n_particles FROM simulation;
        """,
        config=SQLObserverConfig(),
    )
    payload2 = observer_with_comment.resolve_dark_query({}, metadata={})
    assert payload2 == b"hello"
