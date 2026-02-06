"""SQL observer backed by an in-memory SQLite snapshot.

This observer turns an arbitrary simulation state dict into queryable tables, then
executes SQL over that snapshot. The goal is to let experiments express complex
physics/inference probes declaratively instead of embedding ad-hoc Python logic.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np
try:
    import sqlparse
except Exception:  # pragma: no cover - optional dependency fallback.
    sqlparse = None  # type: ignore[assignment]

try:  # Optional import for environments where torch is unavailable at import time.
    import torch
except Exception:  # pragma: no cover - exercised only in minimal envs.
    torch = None  # type: ignore[assignment]


@dataclass
class SQLObserverConfig:
    """Execution and materialization knobs for SQLObserver."""

    row_limit: int = 256
    max_transition_edges: int = 50_000
    max_fold_rows: int = 50_000
    include_query_in_result: bool = False
    query_keys: tuple[str, ...] = (
        "query_bytes",
        "query",
        "test_bytes",
        "test_item",
        "input_bytes",
        "item",
    )
    allow_query_comment_directives: bool = True


class SQLObserver:
    """Run SQL against a state snapshot and return query results."""

    def __init__(self, sql_query: str, config: SQLObserverConfig | None = None):
        self.sql_query = sql_query.strip()
        self.config = config or SQLObserverConfig()

    def resolve_dark_query(
        self,
        state: dict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bytes | list[int] | list[tuple[int, int]] | None:
        """Resolve an inference query payload from metadata/state/SQL directives.

        This lets InferenceObserver manage dark-particle injection automatically:
        users pass a test item to inference, SQLObserver resolves it, and the
        dark injector handles perturbation/cleanup transparently.
        """
        state_dict = state if isinstance(state, dict) else {}
        meta = metadata if isinstance(metadata, dict) else {}

        for key in self.config.query_keys:
            if key in meta:
                payload = self._normalize_dark_query(meta[key])
                if payload is not None:
                    return payload
            if key in state_dict:
                payload = self._normalize_dark_query(state_dict[key])
                if payload is not None:
                    return payload

        if self.config.allow_query_comment_directives:
            payload = self._dark_query_from_sql_comments(self.sql_query)
            if payload is not None:
                return payload
        return None

    def observe(self, state: dict | None = None, **kwargs) -> dict:
        """Materialize state into SQLite, execute query, and return rows + scalars."""
        if not isinstance(state, dict):
            state = {}

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            self._create_tables(conn)
            self._load_state(conn, state)
            rows = self._execute_query(conn, self.sql_query)
        finally:
            conn.close()

        result: dict[str, Any] = {
            "sql_rows": rows,
            "sql_row_count": len(rows),
        }
        if self.config.include_query_in_result:
            result["sql_query"] = self.sql_query

        # Convenience: surface scalar columns from a single-row result directly.
        if len(rows) == 1:
            for key, value in rows[0].items():
                if isinstance(value, (int, float, str)) or value is None:
                    result[f"sql_{key}"] = value

        return result

    def _execute_query(self, conn: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if sqlparse is None:
            statements = [s.strip() for s in query.split(";") if s.strip()]
        else:
            statements = [s.strip() for s in sqlparse.split(query) if s.strip()]
        for statement in statements:
            cur = conn.execute(statement)
            if cur.description:
                fetched = cur.fetchmany(int(max(1, self.config.row_limit)))
                rows = [dict(row) for row in fetched]
        return rows

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE particles (
                idx INTEGER PRIMARY KEY,
                byte_value INTEGER,
                token_id INTEGER,
                sequence_index INTEGER,
                sample_index INTEGER,
                particle_flags INTEGER,
                mass REAL,
                heat REAL,
                energy REAL,
                excitation REAL,
                x REAL,
                y REAL,
                z REAL,
                vx REAL,
                vy REAL,
                vz REAL
            );

            CREATE TABLE modes (
                idx INTEGER PRIMARY KEY,
                amplitude REAL,
                mode_state INTEGER,
                omega REAL,
                phase REAL,
                psi_amplitude REAL
            );

            CREATE TABLE scalars (
                key TEXT PRIMARY KEY,
                value_text TEXT,
                value_real REAL,
                value_int INTEGER
            );

            CREATE TABLE transitions (
                src_key INTEGER,
                dst_key INTEGER,
                src_token_id INTEGER,
                dst_token_id INTEGER,
                src_byte INTEGER,
                dst_byte INTEGER,
                edge_count INTEGER
            );

            CREATE TABLE folds (
                key INTEGER,
                token_id INTEGER,
                sequence_index INTEGER,
                byte_value INTEGER,
                count INTEGER,
                total_mass REAL
            );

            CREATE VIEW particles_visible AS
            SELECT *
            FROM particles
            WHERE (COALESCE(particle_flags, 0) & 1) = 0;

            CREATE VIEW simulation AS
            SELECT
                (SELECT COUNT(*) FROM particles_visible) AS n_particles,
                (SELECT COUNT(DISTINCT token_id) FROM particles_visible WHERE token_id IS NOT NULL) AS n_unique_tokens,
                (SELECT COUNT(*) FROM transitions) AS n_transition_edges,
                (SELECT COUNT(*) FROM folds) AS n_fold_rows,
                (SELECT COALESCE(SUM(mass), 0.0) FROM particles_visible) AS total_mass;
            """
        )

    def _load_state(self, conn: sqlite3.Connection, state: dict) -> None:
        self._insert_scalars(conn, state)
        arrays = self._extract_arrays(state)
        self._insert_particles(conn, arrays)
        self._insert_modes(conn, arrays)
        self._insert_transitions(conn, arrays)
        self._insert_folds(conn, arrays)
        conn.commit()

    def _insert_scalars(self, conn: sqlite3.Connection, state: dict) -> None:
        rows: list[tuple[str, str | None, float | None, int | None]] = []
        for key, value in state.items():
            if isinstance(value, bool):
                rows.append((str(key), str(value), float(int(value)), int(value)))
            elif isinstance(value, int):
                rows.append((str(key), str(value), float(value), int(value)))
            elif isinstance(value, float):
                rows.append((str(key), str(value), float(value), None))
            elif isinstance(value, str):
                rows.append((str(key), value, None, None))
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO scalars(key, value_text, value_real, value_int) VALUES (?, ?, ?, ?)",
                rows,
            )

    def _insert_particles(self, conn: sqlite3.Connection, arrays: dict[str, np.ndarray]) -> None:
        n = self._particle_count(arrays)
        if n <= 0:
            return

        rows: list[tuple[Any, ...]] = []
        for i in range(n):
            rows.append(
                (
                    i,
                    self._take1(arrays.get("byte_values"), i),
                    self._take1(arrays.get("token_ids"), i),
                    self._take1(arrays.get("sequence_indices"), i),
                    self._take1(arrays.get("sample_indices"), i),
                    self._take1(arrays.get("particle_flags"), i),
                    self._take1(arrays.get("masses"), i, as_float=True),
                    self._take1(arrays.get("heats"), i, as_float=True),
                    self._take1(arrays.get("energies"), i, as_float=True),
                    self._take1(arrays.get("excitations"), i, as_float=True),
                    self._take2(arrays.get("positions"), i, 0),
                    self._take2(arrays.get("positions"), i, 1),
                    self._take2(arrays.get("positions"), i, 2),
                    self._take2(arrays.get("velocities"), i, 0),
                    self._take2(arrays.get("velocities"), i, 1),
                    self._take2(arrays.get("velocities"), i, 2),
                )
            )

        conn.executemany(
            """
            INSERT INTO particles(
                idx, byte_value, token_id, sequence_index, sample_index, particle_flags,
                mass, heat, energy, excitation, x, y, z, vx, vy, vz
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_modes(self, conn: sqlite3.Connection, arrays: dict[str, np.ndarray]) -> None:
        amps = arrays.get("amplitudes")
        psi_amp = arrays.get("psi_amplitude")
        mode_state = arrays.get("mode_state")
        omega = arrays.get("omega")
        phase = arrays.get("phase")

        n = 0
        for arr in (amps, psi_amp, mode_state, omega, phase):
            if arr is None:
                continue
            if arr.ndim == 1:
                n = max(n, int(arr.shape[0]))
        if n <= 0:
            return

        rows: list[tuple[Any, ...]] = []
        for i in range(n):
            rows.append(
                (
                    i,
                    self._take1(amps, i, as_float=True),
                    self._take1(mode_state, i),
                    self._take1(omega, i, as_float=True),
                    self._take1(phase, i, as_float=True),
                    self._take1(psi_amp, i, as_float=True),
                )
            )

        conn.executemany(
            "INSERT INTO modes(idx, amplitude, mode_state, omega, phase, psi_amplitude) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def _insert_transitions(self, conn: sqlite3.Connection, arrays: dict[str, np.ndarray]) -> None:
        seq = arrays.get("sequence_indices")
        samples = arrays.get("sample_indices")
        bytes_ = arrays.get("byte_values")
        token_ids = arrays.get("token_ids")
        flags = arrays.get("particle_flags")
        if seq is None or samples is None:
            return
        if seq.ndim != 1 or samples.ndim != 1:
            return
        n = int(min(seq.shape[0], samples.shape[0]))
        if n <= 1:
            return

        if bytes_ is None or bytes_.ndim != 1:
            bytes_ = np.zeros((n,), dtype=np.int64)
        else:
            bytes_ = bytes_[:n].astype(np.int64, copy=False)

        if token_ids is None or token_ids.ndim != 1:
            token_ids = ((seq[:n].astype(np.int64) << 8) | (bytes_.astype(np.int64) & 0xFF)).astype(np.int64)
        else:
            token_ids = token_ids[:n].astype(np.int64, copy=False)

        seq = seq[:n].astype(np.int64, copy=False)
        samples = samples[:n].astype(np.int64, copy=False)

        if flags is not None and flags.ndim == 1:
            vis = (flags[:n].astype(np.int64, copy=False) & np.int64(1)) == 0
            seq = seq[vis]
            samples = samples[vis]
            token_ids = token_ids[vis]
            bytes_ = bytes_[vis]

        order = np.lexsort((seq, samples))
        s = samples[order]
        t = seq[order]
        tok = token_ids[order]
        byt = bytes_[order]

        ok = (s[1:] == s[:-1]) & (t[1:] == t[:-1] + 1)
        if not np.any(ok):
            return

        src_tok = tok[:-1][ok]
        dst_tok = tok[1:][ok]
        src_byte = byt[:-1][ok]
        dst_byte = byt[1:][ok]

        packed = (
            (src_tok.astype(np.uint64) << np.uint64(32))
            | (dst_tok.astype(np.uint64) & np.uint64(0xFFFFFFFF))
        )
        uniq, counts = np.unique(packed, return_counts=True)

        if uniq.size > int(self.config.max_transition_edges):
            top = np.argsort(counts)[::-1][: int(self.config.max_transition_edges)]
            uniq = uniq[top]
            counts = counts[top]

        dst_tok_u = (uniq & np.uint64(0xFFFFFFFF)).astype(np.int64)
        src_tok_u = (uniq >> np.uint64(32)).astype(np.int64)

        byte_lookup: dict[int, int] = {}
        for tid, b in zip(token_ids.tolist(), bytes_.tolist()):
            byte_lookup.setdefault(int(tid), int(b))

        rows: list[tuple[Any, ...]] = []
        for st, dt, c in zip(src_tok_u.tolist(), dst_tok_u.tolist(), counts.tolist()):
            sb = byte_lookup.get(int(st))
            db = byte_lookup.get(int(dt))
            sk = (int(st) << 1) ^ int(sb or 0)
            dk = (int(dt) << 1) ^ int(db or 0)
            rows.append((sk, dk, int(st), int(dt), sb, db, int(c)))

        conn.executemany(
            """
            INSERT INTO transitions(
                src_key, dst_key, src_token_id, dst_token_id, src_byte, dst_byte, edge_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_folds(self, conn: sqlite3.Connection, arrays: dict[str, np.ndarray]) -> None:
        seq = arrays.get("sequence_indices")
        bytes_ = arrays.get("byte_values")
        tok = arrays.get("token_ids")
        mass = arrays.get("masses")
        flags = arrays.get("particle_flags")
        if seq is None or bytes_ is None:
            return
        if seq.ndim != 1 or bytes_.ndim != 1:
            return
        n = int(min(seq.shape[0], bytes_.shape[0]))
        if n <= 0:
            return

        seq = seq[:n].astype(np.int64, copy=False)
        bytes_ = bytes_[:n].astype(np.int64, copy=False)
        if tok is None or tok.ndim != 1:
            tok = ((seq << 8) | (bytes_ & 0xFF)).astype(np.int64)
        else:
            tok = tok[:n].astype(np.int64, copy=False)

        if mass is None or mass.ndim != 1:
            mass = np.ones((n,), dtype=np.float64)
        else:
            mass = mass[:n].astype(np.float64, copy=False)

        if flags is not None and flags.ndim == 1:
            vis = (flags[:n].astype(np.int64, copy=False) & np.int64(1)) == 0
            seq = seq[vis]
            bytes_ = bytes_[vis]
            tok = tok[vis]
            mass = mass[vis]
            n = int(seq.shape[0])
            if n <= 0:
                return

        key = tok
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        tok_s = tok[order]
        seq_s = seq[order]
        byte_s = bytes_[order]
        mass_s = mass[order]
        uniq, start, counts = np.unique(key_s, return_index=True, return_counts=True)
        mass_sum = np.add.reduceat(mass_s, start)

        if uniq.size > int(self.config.max_fold_rows):
            top = np.argsort(mass_sum)[::-1][: int(self.config.max_fold_rows)]
            uniq = uniq[top]
            start = start[top]
            counts = counts[top]
            mass_sum = mass_sum[top]

        rows = []
        for i, k in enumerate(uniq.tolist()):
            j = int(start[i])
            rows.append(
                (
                    int(k),
                    int(tok_s[j]),
                    int(seq_s[j]),
                    int(byte_s[j]),
                    int(counts[i]),
                    float(mass_sum[i]),
                )
            )

        conn.executemany(
            "INSERT INTO folds(key, token_id, sequence_index, byte_value, count, total_mass) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def _extract_arrays(self, state: dict) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        keys = (
            "byte_values",
            "token_ids",
            "sequence_indices",
            "sample_indices",
            "particle_flags",
            "masses",
            "heats",
            "energies",
            "excitations",
            "positions",
            "velocities",
            "amplitudes",
            "mode_state",
            "omega",
            "phase",
            "psi_amplitude",
        )
        for key in keys:
            arr = self._to_numpy(state.get(key))
            if arr is not None:
                arrays[key] = arr
        return arrays

    def _particle_count(self, arrays: dict[str, np.ndarray]) -> int:
        n = 0
        for key in (
            "byte_values",
            "token_ids",
            "sequence_indices",
            "sample_indices",
            "particle_flags",
            "masses",
            "heats",
            "energies",
            "excitations",
        ):
            arr = arrays.get(key)
            if arr is None or arr.ndim != 1:
                continue
            n = max(n, int(arr.shape[0]))

        for key in ("positions", "velocities"):
            arr = arrays.get(key)
            if arr is None or arr.ndim != 2:
                continue
            n = max(n, int(arr.shape[0]))
        return n

    def _normalize_dark_query(self, payload: Any) -> bytes | list[int] | list[tuple[int, int]] | None:
        if payload is None:
            return None

        if isinstance(payload, bytes):
            return payload

        if isinstance(payload, str):
            return payload.encode("utf-8")

        if torch is not None and isinstance(payload, torch.Tensor):
            arr = payload.detach().to("cpu").numpy()
            return self._normalize_dark_query(arr)

        if isinstance(payload, np.ndarray):
            if payload.ndim == 1 and np.issubdtype(payload.dtype, np.integer):
                return [int(x) & 0xFF for x in payload.tolist()]
            if payload.ndim == 2 and payload.shape[1] == 2:
                return [(int(r[0]) & 0xFF, int(r[1])) for r in payload.tolist()]
            return None

        if isinstance(payload, (list, tuple)):
            if len(payload) == 0:
                return []
            first = payload[0]
            if isinstance(first, tuple) and len(first) == 2:
                out: list[tuple[int, int]] = []
                for pair in payload:
                    if not (isinstance(pair, tuple) and len(pair) == 2):
                        return None
                    out.append((int(pair[0]) & 0xFF, int(pair[1])))
                return out
            try:
                return [int(x) & 0xFF for x in payload]
            except Exception:
                return None

        return None

    def _dark_query_from_sql_comments(self, query: str) -> bytes | list[int] | None:
        for raw in query.splitlines():
            line = raw.strip()
            low = line.lower()
            if low.startswith("-- dark_query_text:"):
                value = line.split(":", 1)[1].strip()
                return value.encode("utf-8")
            if low.startswith("-- dark_query_hex:"):
                value = line.split(":", 1)[1].strip().replace(" ", "")
                if value:
                    try:
                        return bytes.fromhex(value)
                    except ValueError:
                        return None
            if low.startswith("-- dark_query_bytes:"):
                value = line.split(":", 1)[1].strip()
                if not value:
                    return []
                try:
                    return [int(x.strip()) & 0xFF for x in value.split(",") if x.strip()]
                except ValueError:
                    return None
        return None

    def _to_numpy(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None

        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().to("cpu").numpy()

        if isinstance(value, np.ndarray):
            return value

        if isinstance(value, (list, tuple)):
            try:
                return np.asarray(value)
            except Exception:
                return None

        return None

    def _take1(self, arr: np.ndarray | None, i: int, *, as_float: bool = False) -> Any:
        if arr is None or arr.ndim != 1 or i >= int(arr.shape[0]):
            return None
        v = arr[i]
        if as_float:
            return float(v)
        if np.issubdtype(arr.dtype, np.integer):
            return int(v)
        if np.issubdtype(arr.dtype, np.floating):
            return float(v)
        return None

    def _take2(self, arr: np.ndarray | None, i: int, j: int) -> float | None:
        if arr is None or arr.ndim != 2:
            return None
        if i >= int(arr.shape[0]) or j >= int(arr.shape[1]):
            return None
        return float(arr[i, j])
