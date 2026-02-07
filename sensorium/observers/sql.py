"""SQL observer via in-memory transpilation to observer tables.

This observer does not use SQLite. It treats SQL as a compact query language
over observer-derived in-memory tables (`simulation`, `transitions`, `folds`,
`particles`, `modes`, `scalars`) and executes a supported SQL subset directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
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


@dataclass(frozen=True, slots=True)
class _SelectItem:
    expr: str
    alias: str | None


@dataclass(frozen=True, slots=True)
class _OrderTerm:
    expr: str
    descending: bool


@dataclass(frozen=True, slots=True)
class _ParsedSelect:
    items: list[_SelectItem]
    source: str | None
    where_expr: str | None
    group_by: list[str]
    order_by: list[_OrderTerm]
    limit: int | None


class SQLObserver:
    """Run SQL-like queries against observer-derived in-memory tables."""

    def __init__(self, sql_query: str, config: SQLObserverConfig | None = None):
        self.sql_query = sql_query.strip()
        self.config = config or SQLObserverConfig()

    def resolve_dark_query(
        self,
        state: dict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bytes | list[int] | list[tuple[int, int]] | None:
        """Resolve an inference query payload from metadata/state/SQL directives."""
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
        """Execute SQL-like query against observer tables."""
        if not isinstance(state, dict):
            state = {}

        context = self._build_context(state)
        rows: list[dict[str, Any]] = []
        for statement in self._split_statements(self.sql_query):
            rows = self._execute_statement(statement, context, state)

        result: dict[str, Any] = {
            "sql_rows": rows,
            "sql_row_count": len(rows),
        }
        if self.config.include_query_in_result:
            result["sql_query"] = self.sql_query

        if len(rows) == 1:
            for key, value in rows[0].items():
                if isinstance(value, (int, float, str)) or value is None:
                    result[f"sql_{key}"] = value

        return result

    # ---------------------------------------------------------------------
    # Query parsing / execution
    # ---------------------------------------------------------------------

    def _execute_statement(
        self,
        statement: str,
        context: dict[str, list[dict[str, Any]]],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sql = self._strip_comments(statement).strip().rstrip(";").strip()
        if not sql:
            return []
        parsed = self._parse_select(sql)
        rows = self._resolve_source_rows(parsed.source, context, state)

        if parsed.where_expr:
            rows = [
                r
                for r in rows
                if bool(self._truthy(self._eval_condition(parsed.where_expr, row=r, rows=None, context=context, state=state)))
            ]

        projected = self._project_rows(parsed, rows, context, state)
        projected = self._apply_order_by(projected, parsed.order_by, context, state)

        if parsed.limit is not None:
            projected = projected[: max(0, int(parsed.limit))]
        max_rows = int(max(1, self.config.row_limit))
        if len(projected) > max_rows:
            projected = projected[:max_rows]
        return projected

    def _parse_select(self, statement: str) -> _ParsedSelect:
        stmt = statement.strip()
        if not stmt.lower().startswith("select"):
            raise ValueError(f"Only SELECT statements are supported, got: {statement!r}")

        body = stmt[6:].strip()
        from_pos = self._find_top_level_keyword(body, "from")

        if from_pos < 0:
            select_part = body
            tail = ""
        else:
            select_part = body[:from_pos].strip()
            tail = body[from_pos + len("from") :].strip()

        items = self._parse_select_items(select_part)
        source: str | None = None
        where_expr: str | None = None
        group_by: list[str] = []
        order_by: list[_OrderTerm] = []
        limit: int | None = None

        if tail:
            clauses = self._split_clauses(tail)
            source = clauses.get("source")
            where_expr = clauses.get("where")
            grp = clauses.get("group by")
            ordc = clauses.get("order by")
            lim = clauses.get("limit")

            if grp:
                group_by = [x.strip() for x in self._split_top_level(grp, ",") if x.strip()]
            if ordc:
                order_by = self._parse_order_by(ordc)
            if lim:
                lim_text = lim.strip()
                try:
                    limit = int(lim_text.split()[0])
                except Exception:
                    raise ValueError(f"Invalid LIMIT clause: {lim_text!r}")

        return _ParsedSelect(
            items=items,
            source=source.strip() if isinstance(source, str) and source.strip() else None,
            where_expr=where_expr.strip() if isinstance(where_expr, str) and where_expr.strip() else None,
            group_by=group_by,
            order_by=order_by,
            limit=limit,
        )

    def _split_clauses(self, tail: str) -> dict[str, str]:
        clauses: dict[str, str] = {}
        keys = ("where", "group by", "order by", "limit")
        found: list[tuple[int, str]] = []
        for key in keys:
            pos = self._find_top_level_keyword(tail, key)
            if pos >= 0:
                found.append((pos, key))
        found.sort(key=lambda x: x[0])

        if not found:
            clauses["source"] = tail.strip()
            return clauses

        source_end = found[0][0]
        clauses["source"] = tail[:source_end].strip()
        for i, (pos, key) in enumerate(found):
            start = pos + len(key)
            end = found[i + 1][0] if i + 1 < len(found) else len(tail)
            clauses[key] = tail[start:end].strip()
        return clauses

    def _parse_select_items(self, select_part: str) -> list[_SelectItem]:
        out: list[_SelectItem] = []
        for raw in self._split_top_level(select_part, ","):
            expr_raw = raw.strip()
            if not expr_raw:
                continue
            expr, alias = self._split_alias(expr_raw)
            out.append(_SelectItem(expr=expr, alias=alias))
        if not out:
            raise ValueError("SELECT list is empty")
        return out

    def _parse_order_by(self, clause: str) -> list[_OrderTerm]:
        out: list[_OrderTerm] = []
        for raw in self._split_top_level(clause, ","):
            part = raw.strip()
            if not part:
                continue
            m = re.match(r"(?is)^(.*?)(?:\s+(asc|desc))?$", part)
            if not m:
                continue
            expr = (m.group(1) or "").strip()
            desc = (m.group(2) or "").strip().lower() == "desc"
            if expr:
                out.append(_OrderTerm(expr=expr, descending=desc))
        return out

    def _resolve_source_rows(
        self,
        source: str | None,
        context: dict[str, list[dict[str, Any]]],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if source is None:
            return [{}]

        src = source.strip()
        if not src:
            return [{}]

        if src.startswith("("):
            inner_sql, remainder = self._extract_parenthesized(src)
            inner_rows = self._execute_statement(inner_sql, context, state)
            alias = self._parse_alias_name(remainder)
            if alias:
                return [self._with_aliases(row, alias) for row in inner_rows]
            return [dict(row) for row in inner_rows]

        table, alias = self._parse_table_source(src)
        rows = context.get(table.lower(), [])
        out = [dict(r) for r in rows]
        if alias:
            out = [self._with_aliases(r, alias, table_name=table) for r in out]
        return out

    def _parse_table_source(self, source: str) -> tuple[str, str | None]:
        tokens = source.strip().split()
        if not tokens:
            raise ValueError("Missing source table")
        table = tokens[0]
        alias: str | None = None
        if len(tokens) >= 3 and tokens[1].lower() == "as":
            alias = tokens[2]
        elif len(tokens) >= 2:
            alias = tokens[1]
        return table, alias

    def _project_rows(
        self,
        parsed: _ParsedSelect,
        rows: list[dict[str, Any]],
        context: dict[str, list[dict[str, Any]]],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        has_agg = any(self._expr_has_aggregate(item.expr) for item in parsed.items)

        if parsed.group_by:
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            order: list[tuple[Any, ...]] = []
            for row in rows:
                key = tuple(
                    self._eval_expr(expr, row=row, rows=None, context=context, state=state)
                    for expr in parsed.group_by
                )
                if key not in grouped:
                    grouped[key] = []
                    order.append(key)
                grouped[key].append(row)
            out: list[dict[str, Any]] = []
            for key in order:
                group_rows = grouped[key]
                out.append(self._project_one(parsed.items, row=group_rows[0], rows=group_rows, context=context, state=state))
            return out

        if has_agg:
            return [self._project_one(parsed.items, row=rows[0] if rows else {}, rows=rows, context=context, state=state)]

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(self._project_one(parsed.items, row=row, rows=None, context=context, state=state))
        return out

    def _project_one(
        self,
        items: list[_SelectItem],
        *,
        row: dict[str, Any] | None,
        rows: list[dict[str, Any]] | None,
        context: dict[str, list[dict[str, Any]]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        row_local = row or {}
        for item in items:
            expr = item.expr.strip()
            if expr == "*":
                for k, v in row_local.items():
                    out[k] = v
                continue
            key = item.alias or self._infer_output_name(expr)
            out[key] = self._eval_expr(expr, row=row_local, rows=rows, context=context, state=state)
        return out

    def _apply_order_by(
        self,
        rows: list[dict[str, Any]],
        order_by: list[_OrderTerm],
        context: dict[str, list[dict[str, Any]]],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not order_by or len(rows) <= 1:
            return rows
        out = list(rows)
        for term in reversed(order_by):
            out.sort(
                key=lambda r: self._order_key(
                    self._eval_expr(term.expr, row=r, rows=None, context=context, state=state)
                ),
                reverse=term.descending,
            )
        return out

    # ---------------------------------------------------------------------
    # Expression evaluation
    # ---------------------------------------------------------------------

    def _eval_expr(
        self,
        expr: str,
        *,
        row: dict[str, Any] | None,
        rows: list[dict[str, Any]] | None,
        context: dict[str, list[dict[str, Any]]],
        state: dict[str, Any],
    ) -> Any:
        raw = expr.strip()
        if self._is_select_subquery(raw):
            inner = raw[1:-1].strip()
            inner_rows = self._execute_statement(inner, context, state)
            if not inner_rows:
                return None
            first = inner_rows[0]
            if not first:
                return None
            return next(iter(first.values()))

        text = self._strip_outer_parens(raw)
        if not text:
            return None

        cast_arg = self._parse_function(text, "cast")
        if cast_arg is not None:
            idx = self._find_top_level_keyword(cast_arg, "as")
            if idx < 0:
                return None
            lhs = cast_arg[:idx].strip()
            typ = cast_arg[idx + len("as") :].strip().lower()
            val = self._eval_expr(lhs, row=row, rows=rows, context=context, state=state)
            if val is None:
                return None
            if typ in {"text", "string", "varchar", "char"}:
                return str(val)
            if typ in {"int", "integer"}:
                return int(val)
            if typ in {"float", "double", "real"}:
                return float(val)
            return val

        coalesce_arg = self._parse_function(text, "coalesce")
        if coalesce_arg is not None:
            for arg in self._split_top_level(coalesce_arg, ","):
                val = self._eval_expr(arg, row=row, rows=rows, context=context, state=state)
                if val is not None:
                    return val
            return None

        case_val = self._eval_case(text, row=row, rows=rows, context=context, state=state)
        if case_val is not _NO_CASE_MATCH:
            return case_val

        agg = self._parse_aggregate(text)
        if agg is not None:
            fn, arg = agg
            return self._eval_aggregate(fn, arg, rows=rows, context=context, state=state)

        num = self._parse_numeric_literal(text)
        if num is not _NO_NUMERIC:
            return num

        s_lit = self._parse_string_literal(text)
        if s_lit is not _NO_STRING:
            return s_lit

        return self._lookup_value(text, row=row, state=state)

    def _eval_case(
        self,
        expr: str,
        *,
        row: dict[str, Any] | None,
        rows: list[dict[str, Any]] | None,
        context: dict[str, list[dict[str, Any]]],
        state: dict[str, Any],
    ) -> Any:
        m = re.match(r"(?is)^case\s+when\s+(.+?)\s+then\s+(.+?)\s+else\s+(.+?)\s+end$", expr)
        if not m:
            return _NO_CASE_MATCH
        cond = (m.group(1) or "").strip()
        then_expr = (m.group(2) or "").strip()
        else_expr = (m.group(3) or "").strip()
        if self._truthy(self._eval_condition(cond, row=row, rows=rows, context=context, state=state)):
            return self._eval_expr(then_expr, row=row, rows=rows, context=context, state=state)
        return self._eval_expr(else_expr, row=row, rows=rows, context=context, state=state)

    def _eval_condition(
        self,
        cond: str,
        *,
        row: dict[str, Any] | None,
        rows: list[dict[str, Any]] | None,
        context: dict[str, list[dict[str, Any]]],
        state: dict[str, Any],
    ) -> Any:
        text = self._strip_outer_parens(cond.strip())
        for op in (">=", "<=", "<>", "!=", ">", "<", "="):
            idx = self._find_top_level_operator(text, op)
            if idx >= 0:
                lhs = text[:idx].strip()
                rhs = text[idx + len(op) :].strip()
                lv = self._eval_expr(lhs, row=row, rows=rows, context=context, state=state)
                rv = self._eval_expr(rhs, row=row, rows=rows, context=context, state=state)
                return self._compare(lv, rv, op)
        return self._eval_expr(text, row=row, rows=rows, context=context, state=state)

    def _eval_aggregate(
        self,
        fn: str,
        arg: str,
        *,
        rows: list[dict[str, Any]] | None,
        context: dict[str, list[dict[str, Any]]],
        state: dict[str, Any],
    ) -> Any:
        data = rows or []
        name = fn.lower()
        arg_text = arg.strip()

        if name == "count":
            if arg_text == "*":
                return int(len(data))
            m = re.match(r"(?is)^distinct\s+(.+)$", arg_text)
            if m:
                expr = (m.group(1) or "").strip()
                vals = {
                    self._hashable(
                        self._eval_expr(expr, row=r, rows=None, context=context, state=state)
                    )
                    for r in data
                }
                vals.discard(None)
                return int(len(vals))
            return int(
                sum(
                    1
                    for r in data
                    if self._eval_expr(arg_text, row=r, rows=None, context=context, state=state) is not None
                )
            )

        vals = [
            self._eval_expr(arg_text, row=r, rows=None, context=context, state=state)
            for r in data
        ]
        vals = [v for v in vals if v is not None]
        if not vals:
            if name in {"sum"}:
                return 0
            return None

        if name == "sum":
            nums = [self._to_float(v) for v in vals]
            nums = [v for v in nums if v is not None]
            return float(sum(nums)) if nums else 0.0
        if name == "avg":
            nums = [self._to_float(v) for v in vals]
            nums = [v for v in nums if v is not None]
            if not nums:
                return None
            return float(sum(nums) / len(nums))
        if name == "max":
            return max(vals)
        if name == "min":
            return min(vals)
        raise ValueError(f"Unsupported aggregate function: {fn}")

    # ---------------------------------------------------------------------
    # Context (observer table) construction
    # ---------------------------------------------------------------------

    def _build_context(self, state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        arrays = self._extract_arrays(state)
        particles = self._build_particles_rows(arrays)
        particles_visible = [r for r in particles if (int(r.get("particle_flags") or 0) & 1) == 0]
        modes = self._build_modes_rows(arrays)
        transitions = self._build_transitions_rows(arrays)
        folds = self._build_folds_rows(arrays)
        scalars = self._build_scalar_rows(state)
        simulation_row = self._build_simulation_row(
            state=state,
            particles_visible=particles_visible,
            transitions=transitions,
            folds=folds,
            modes=modes,
        )
        return {
            "particles": particles,
            "particles_visible": particles_visible,
            "modes": modes,
            "scalars": scalars,
            "transitions": transitions,
            "folds": folds,
            "simulation": [simulation_row],
        }

    def _build_simulation_row(
        self,
        *,
        state: dict[str, Any],
        particles_visible: list[dict[str, Any]],
        transitions: list[dict[str, Any]],
        folds: list[dict[str, Any]],
        modes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        token_ids = [r.get("token_id") for r in particles_visible if r.get("token_id") is not None]
        unique_tokens = {int(t) for t in token_ids}
        total_mass = 0.0
        for row in particles_visible:
            m = row.get("mass")
            if m is not None:
                total_mass += float(m)

        out: dict[str, Any] = {
            "n_particles": int(len(particles_visible)),
            "n_unique_tokens": int(len(unique_tokens)),
            "n_transition_edges": int(len(transitions)),
            "n_fold_rows": int(len(folds)),
            "total_mass": float(total_mass),
            "particles": particles_visible,
            "modes": modes,
        }

        if "load_count" in state and isinstance(state["load_count"], (int, float)):
            out["load_count"] = int(state["load_count"])
        elif "step" in state and isinstance(state["step"], (int, float)):
            out["load_count"] = int(state["step"])

        for key, value in state.items():
            if key in out:
                continue
            if isinstance(value, bool):
                out[key] = int(value)
            elif isinstance(value, (int, float, str)):
                out[key] = value
        return out

    def _build_scalar_rows(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, value in state.items():
            if isinstance(value, bool):
                rows.append(
                    {
                        "key": str(key),
                        "value_text": str(value),
                        "value_real": float(int(value)),
                        "value_int": int(value),
                    }
                )
            elif isinstance(value, int):
                rows.append(
                    {
                        "key": str(key),
                        "value_text": str(value),
                        "value_real": float(value),
                        "value_int": int(value),
                    }
                )
            elif isinstance(value, float):
                rows.append(
                    {
                        "key": str(key),
                        "value_text": str(value),
                        "value_real": float(value),
                        "value_int": None,
                    }
                )
            elif isinstance(value, str):
                rows.append(
                    {
                        "key": str(key),
                        "value_text": value,
                        "value_real": None,
                        "value_int": None,
                    }
                )
        return rows

    def _build_particles_rows(self, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        n = self._particle_count(arrays)
        if n <= 0:
            return []
        out: list[dict[str, Any]] = []
        for i in range(n):
            out.append(
                {
                    "idx": int(i),
                    "byte_value": self._take1(arrays.get("byte_values"), i),
                    "token_id": self._take1(arrays.get("token_ids"), i),
                    "sequence_index": self._take1(arrays.get("sequence_indices"), i),
                    "sample_index": self._take1(arrays.get("sample_indices"), i),
                    "particle_flags": self._take1(arrays.get("particle_flags"), i),
                    "mass": self._take1(arrays.get("masses"), i, as_float=True),
                    "heat": self._take1(arrays.get("heats"), i, as_float=True),
                    "energy": self._take1(arrays.get("energies"), i, as_float=True),
                    "excitation": self._take1(arrays.get("excitations"), i, as_float=True),
                    "x": self._take2(arrays.get("positions"), i, 0),
                    "y": self._take2(arrays.get("positions"), i, 1),
                    "z": self._take2(arrays.get("positions"), i, 2),
                    "vx": self._take2(arrays.get("velocities"), i, 0),
                    "vy": self._take2(arrays.get("velocities"), i, 1),
                    "vz": self._take2(arrays.get("velocities"), i, 2),
                }
            )
        return out

    def _build_modes_rows(self, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
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
            return []

        out: list[dict[str, Any]] = []
        for i in range(n):
            out.append(
                {
                    "idx": int(i),
                    "amplitude": self._take1(amps, i, as_float=True),
                    "mode_state": self._take1(mode_state, i),
                    "omega": self._take1(omega, i, as_float=True),
                    "phase": self._take1(phase, i, as_float=True),
                    "psi_amplitude": self._take1(psi_amp, i, as_float=True),
                }
            )
        return out

    def _build_transitions_rows(self, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        seq = arrays.get("sequence_indices")
        samples = arrays.get("sample_indices")
        bytes_ = arrays.get("byte_values")
        token_ids = arrays.get("token_ids")
        flags = arrays.get("particle_flags")
        if seq is None or samples is None:
            return []
        if seq.ndim != 1 or samples.ndim != 1:
            return []

        n = int(min(seq.shape[0], samples.shape[0]))
        if n <= 1:
            return []

        seq = seq[:n].astype(np.int64, copy=False)
        samples = samples[:n].astype(np.int64, copy=False)
        if bytes_ is None or bytes_.ndim != 1:
            bytes_ = np.zeros((n,), dtype=np.int64)
        else:
            bytes_ = bytes_[:n].astype(np.int64, copy=False)

        if token_ids is None or token_ids.ndim != 1:
            token_ids = ((seq << 8) | (bytes_ & 0xFF)).astype(np.int64)
        else:
            token_ids = token_ids[:n].astype(np.int64, copy=False)

        if flags is not None and flags.ndim == 1:
            vis = (flags[:n].astype(np.int64, copy=False) & np.int64(1)) == 0
            seq = seq[vis]
            samples = samples[vis]
            token_ids = token_ids[vis]
            bytes_ = bytes_[vis]

        if seq.size <= 1:
            return []

        order = np.lexsort((seq, samples))
        s = samples[order]
        t = seq[order]
        tok = token_ids[order]
        byt = bytes_[order]

        ok = (s[1:] == s[:-1]) & (t[1:] == t[:-1] + 1)
        if not np.any(ok):
            return []

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
        for tid, b in zip(token_ids.tolist(), bytes_.tolist(), strict=False):
            byte_lookup.setdefault(int(tid), int(b))

        out: list[dict[str, Any]] = []
        for st, dt, c in zip(src_tok_u.tolist(), dst_tok_u.tolist(), counts.tolist(), strict=False):
            sb = byte_lookup.get(int(st))
            db = byte_lookup.get(int(dt))
            sk = (int(st) << 1) ^ int(sb or 0)
            dk = (int(dt) << 1) ^ int(db or 0)
            out.append(
                {
                    "src_key": int(sk),
                    "dst_key": int(dk),
                    "src_token_id": int(st),
                    "dst_token_id": int(dt),
                    "src_byte": int(sb) if sb is not None else None,
                    "dst_byte": int(db) if db is not None else None,
                    "edge_count": int(c),
                }
            )
        return out

    def _build_folds_rows(self, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        seq = arrays.get("sequence_indices")
        bytes_ = arrays.get("byte_values")
        tok = arrays.get("token_ids")
        mass = arrays.get("masses")
        flags = arrays.get("particle_flags")
        if seq is None or bytes_ is None:
            return []
        if seq.ndim != 1 or bytes_.ndim != 1:
            return []

        n = int(min(seq.shape[0], bytes_.shape[0]))
        if n <= 0:
            return []

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
                return []

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

        out: list[dict[str, Any]] = []
        for i, k in enumerate(uniq.tolist()):
            j = int(start[i])
            out.append(
                {
                    "key": int(k),
                    "token_id": int(tok_s[j]),
                    "sequence_index": int(seq_s[j]),
                    "byte_value": int(byte_s[j]),
                    "count": int(counts[i]),
                    "total_mass": float(mass_sum[i]),
                }
            )
        return out

    # ---------------------------------------------------------------------
    # SQL parsing helpers
    # ---------------------------------------------------------------------

    def _split_statements(self, query: str) -> list[str]:
        if sqlparse is not None:
            return [s.strip() for s in sqlparse.split(query) if s.strip()]
        parts = self._split_top_level(query, ";")
        return [p.strip() for p in parts if p.strip()]

    def _split_alias(self, expr: str) -> tuple[str, str | None]:
        idx = self._rfind_top_level_keyword(expr, "as")
        if idx < 0:
            return expr.strip(), None
        lhs = expr[:idx].strip()
        rhs = expr[idx + len("as") :].strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", rhs):
            return lhs, rhs
        return expr.strip(), None

    def _infer_output_name(self, expr: str) -> str:
        e = self._strip_outer_parens(expr.strip())
        if e == "*":
            return "*"
        m = re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", e)
        if m:
            return e.split(".")[-1]
        agg = self._parse_aggregate(e)
        if agg is not None:
            return agg[0].lower()
        return e

    def _find_top_level_keyword(self, text: str, keyword: str, start: int = 0) -> int:
        target = keyword.lower()
        lower = text.lower()
        depth = 0
        in_single = False
        in_double = False
        i = max(0, int(start))
        while i <= len(text) - len(target):
            ch = text[i]
            if ch == "'" and not in_double:
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue
            if in_single or in_double:
                i += 1
                continue
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if depth == 0 and lower.startswith(target, i):
                before_ok = i == 0 or not self._is_word_char(lower[i - 1])
                after_idx = i + len(target)
                after_ok = after_idx >= len(lower) or not self._is_word_char(lower[after_idx])
                if before_ok and after_ok:
                    return i
            i += 1
        return -1

    def _rfind_top_level_keyword(self, text: str, keyword: str) -> int:
        pos = -1
        start = 0
        while True:
            idx = self._find_top_level_keyword(text, keyword, start=start)
            if idx < 0:
                return pos
            pos = idx
            start = idx + 1

    def _extract_parenthesized(self, text: str) -> tuple[str, str]:
        if not text.startswith("("):
            raise ValueError(f"Expected parenthesized source, got: {text!r}")
        depth = 0
        in_single = False
        in_double = False
        end = -1
        for i, ch in enumerate(text):
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if in_single or in_double:
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            raise ValueError(f"Unclosed parenthesized source: {text!r}")
        inner = text[1:end].strip()
        remainder = text[end + 1 :].strip()
        return inner, remainder

    def _parse_alias_name(self, remainder: str) -> str | None:
        if not remainder:
            return None
        tokens = remainder.split()
        if not tokens:
            return None
        if tokens[0].lower() == "as":
            if len(tokens) >= 2 and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tokens[1]):
                return tokens[1]
            return None
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tokens[0]):
            return tokens[0]
        return None

    def _split_top_level(self, text: str, sep: str) -> list[str]:
        out: list[str] = []
        depth = 0
        in_single = False
        in_double = False
        start = 0
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'" and not in_double:
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue
            if in_single or in_double:
                i += 1
                continue
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if depth == 0 and text.startswith(sep, i):
                out.append(text[start:i])
                i += len(sep)
                start = i
                continue
            i += 1
        out.append(text[start:])
        return out

    def _strip_outer_parens(self, expr: str) -> str:
        out = expr.strip()
        while out.startswith("(") and out.endswith(")"):
            inner, remainder = self._extract_parenthesized(out)
            if remainder:
                break
            out = inner.strip()
        return out

    def _is_select_subquery(self, expr: str) -> bool:
        s = expr.strip()
        if not (s.startswith("(") and s.endswith(")")):
            return False
        inner, remainder = self._extract_parenthesized(s)
        if remainder:
            return False
        return inner.strip().lower().startswith("select")

    def _parse_function(self, expr: str, name: str) -> str | None:
        s = expr.strip()
        prefix = f"{name.lower()}("
        if not s.lower().startswith(prefix) or not s.endswith(")"):
            return None
        inner = s[len(prefix) : -1]
        return inner.strip()

    def _parse_aggregate(self, expr: str) -> tuple[str, str] | None:
        s = expr.strip()
        m = re.match(r"(?is)^(count|max|min|avg|sum)\s*\((.*)\)$", s)
        if not m:
            return None
        return (m.group(1), (m.group(2) or "").strip())

    def _find_top_level_operator(self, text: str, op: str) -> int:
        lower = text.lower()
        depth = 0
        in_single = False
        in_double = False
        i = 0
        while i <= len(text) - len(op):
            ch = text[i]
            if ch == "'" and not in_double:
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue
            if in_single or in_double:
                i += 1
                continue
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if depth == 0 and lower.startswith(op.lower(), i):
                return i
            i += 1
        return -1

    def _strip_comments(self, statement: str) -> str:
        lines: list[str] = []
        for raw in statement.splitlines():
            line = raw
            idx = line.find("--")
            if idx >= 0:
                line = line[:idx]
            if line.strip():
                lines.append(line)
        return "\n".join(lines)

    def _with_aliases(self, row: dict[str, Any], alias: str, table_name: str | None = None) -> dict[str, Any]:
        out = dict(row)
        for key, value in row.items():
            out[f"{alias}.{key}"] = value
            if table_name:
                out[f"{table_name}.{key}"] = value
        return out

    # ---------------------------------------------------------------------
    # Value helpers
    # ---------------------------------------------------------------------

    def _lookup_value(self, key: str, *, row: dict[str, Any] | None, state: dict[str, Any]) -> Any:
        token = key.strip()
        if row:
            if token in row:
                return row[token]
            low = token.lower()
            for k, v in row.items():
                if k.lower() == low:
                    return v
            if "." in token:
                short = token.split(".")[-1]
                if short in row:
                    return row[short]
                low_short = short.lower()
                for k, v in row.items():
                    if k.lower() == low_short:
                        return v
        if token in state and isinstance(state[token], (int, float, str, bool)):
            return state[token]
        return None

    def _parse_numeric_literal(self, text: str) -> Any:
        t = text.strip()
        if re.match(r"^[+-]?\d+$", t):
            try:
                return int(t)
            except Exception:
                return _NO_NUMERIC
        if re.match(r"^[+-]?\d*\.\d+(?:[eE][+-]?\d+)?$", t) or re.match(r"^[+-]?\d+[eE][+-]?\d+$", t):
            try:
                return float(t)
            except Exception:
                return _NO_NUMERIC
        return _NO_NUMERIC

    def _parse_string_literal(self, text: str) -> Any:
        t = text.strip()
        if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
            return t[1:-1]
        return _NO_STRING

    def _compare(self, lhs: Any, rhs: Any, op: str) -> bool:
        if op in {"=", "=="}:
            return lhs == rhs
        if op in {"!=", "<>"}:
            return lhs != rhs
        lv = self._to_float(lhs)
        rv = self._to_float(rhs)
        if lv is None or rv is None:
            return False
        if op == ">":
            return lv > rv
        if op == "<":
            return lv < rv
        if op == ">=":
            return lv >= rv
        if op == "<=":
            return lv <= rv
        return False

    def _truthy(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value != ""
        return bool(value)

    def _expr_has_aggregate(self, expr: str) -> bool:
        return re.search(r"(?is)\b(count|max|min|avg|sum)\s*\(", expr) is not None

    def _order_key(self, value: Any) -> tuple[int, Any]:
        if value is None:
            return (1, 0)
        if isinstance(value, bool):
            return (0, int(value))
        if isinstance(value, (int, float, str)):
            return (0, value)
        return (0, str(value))

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except Exception:
            return None

    def _hashable(self, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(self._hashable(x) for x in value)
        if isinstance(value, dict):
            return tuple(sorted((k, self._hashable(v)) for k, v in value.items()))
        return value

    @staticmethod
    def _is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    # ---------------------------------------------------------------------
    # Data extraction (numpy conversion) and row builders
    # ---------------------------------------------------------------------

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


_NO_CASE_MATCH = object()
_NO_NUMERIC = object()
_NO_STRING = object()
