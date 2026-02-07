"""Rule shift prediction observer using trie pattern matching.

Predicts next bytes at evaluation points using the learned
pattern distribution from the manifold.
"""

from __future__ import annotations

from typing import List, Tuple, Any, Dict, Sequence

import numpy as np

from sensorium.observers.types import ObservationResult


class RuleShiftPredictor:
    """Observer that predicts next byte using trie pattern matching.

    Used to evaluate how well the manifold adapts when patterns shift.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        prime: int = 31,
        context_length: int = 8,
    ):
        self.vocab_size = vocab_size
        self.prime = prime
        self.context_length = context_length
        self.mask = vocab_size - 1
        self.inv_prime = pow(prime, -1, vocab_size)

    def observe(self, state=None, **kwargs) -> Dict[str, Any]:
        """Predict bytes at evaluation points using only past data."""
        if state is None:
            return {}

        # Extract data from observation
        if isinstance(state, ObservationResult):
            data = state.data
        elif isinstance(state, dict):
            data = state
        else:
            return {}

        token_ids = data.get("token_ids")
        sequence_indices = data.get("sequence_indices")
        energies = data.get("energies")
        phase_schedule = data.get("phase_schedule", [])
        forward_phrase = data.get("forward_phrase", "")
        reverse_phrase = data.get("reverse_phrase", "")
        forward_reps = int(data.get("forward_reps", 50))
        reverse_reps = int(data.get("reverse_reps", 50))
        eval_every = data.get("eval_every", 5)
        segment_size = data.get("segment_size", 24)

        if not phase_schedule:
            phase_schedule = [
                {
                    "name": "forward",
                    "phrase": str(forward_phrase),
                    "start_rep": 0,
                    "end_rep": int(forward_reps),
                },
                {
                    "name": "reverse",
                    "phrase": str(reverse_phrase),
                    "start_rep": int(forward_reps),
                    "end_rep": int(forward_reps + reverse_reps),
                },
            ]

        if token_ids is None:
            return {}

        # Convert to numpy
        token_ids_np = (
            token_ids.cpu().numpy()
            if hasattr(token_ids, "cpu")
            else np.array(token_ids)
        )
        if energies is None:
            energies_np = np.ones(len(token_ids_np), dtype=np.float32)
        else:
            energies_np = (
                energies.cpu().numpy()
                if hasattr(energies, "cpu")
                else np.asarray(energies)
            )

        if sequence_indices is None:
            seq_np = np.arange(len(token_ids_np), dtype=np.int64)
        else:
            seq_np = (
                sequence_indices.cpu().numpy()
                if hasattr(sequence_indices, "cpu")
                else np.asarray(sequence_indices)
            ).astype(np.int64)

        accuracy_history = []
        total_reps = int(max(int(item.get("end_rep", 0)) for item in phase_schedule))

        start_rep = max(int(eval_every), 2)
        eval_reps = set(range(start_rep, total_reps + 1, int(eval_every)))
        for phase in phase_schedule[1:]:
            shift_rep = int(phase.get("start_rep", 0)) + 1
            eval_reps.add(shift_rep)
            eval_reps.add(shift_rep + 1)

        for rep in sorted(r for r in eval_reps if 1 <= int(r) <= int(total_reps)):
            # Evaluate repetition `rep` using only prefix up to end of repetition rep-1.
            current_byte = (rep - 1) * segment_size
            prefix_mask = seq_np < int(current_byte)
            if not np.any(prefix_mask):
                continue
            prefix_seq = seq_np[prefix_mask]
            order = np.argsort(prefix_seq, kind="stable")
            prefix_token_ids = token_ids_np[prefix_mask][order]
            prefix_energies = energies_np[prefix_mask][order]

            phase_name, test_phrase, phase_idx = self._phase_for_rep(
                rep,
                phase_schedule,
            )

            correct = 0
            top3_correct = 0
            mrr_sum = 0.0
            total = 0

            for seg_pos in range(self.context_length, segment_size):
                context_bytes = []
                for i in range(self.context_length):
                    ctx_pos = seg_pos - self.context_length + i
                    byte_val = ord(test_phrase[ctx_pos])
                    context_bytes.append(byte_val)

                predictions = self._predict_next_byte(
                    prefix_token_ids,
                    prefix_energies,
                    context_bytes,
                )

                actual_byte = ord(test_phrase[seg_pos])
                predicted_byte = predictions[0][0] if predictions else 128
                top3 = {byte for byte, _ in predictions[:3]}
                rank = None
                for idx, (byte, _prob) in enumerate(predictions):
                    if int(byte) == int(actual_byte):
                        rank = int(idx + 1)
                        break

                if predicted_byte == actual_byte:
                    correct += 1
                if int(actual_byte) in top3:
                    top3_correct += 1
                if rank is not None:
                    mrr_sum += 1.0 / float(rank)
                total += 1

            accuracy = correct / total if total > 0 else 0.0
            top3_acc = (top3_correct / total) if total > 0 else 0.0
            mrr = (mrr_sum / total) if total > 0 else 0.0

            accuracy_history.append(
                {
                    "rep": rep,
                    "byte_position": current_byte,
                    "phase": phase_name,
                    "phase_idx": int(phase_idx),
                    "accuracy": accuracy,
                    "top1": accuracy,
                    "top3": top3_acc,
                    "mrr": mrr,
                    "correct": correct,
                    "total": total,
                }
            )

        summary = self._summarize_history(accuracy_history, phase_schedule)

        return {
            "accuracy_history": accuracy_history,
            **summary,
        }

    def _phase_for_rep(
        self,
        rep: int,
        phase_schedule: Sequence[Dict[str, Any]],
    ) -> tuple[str, str, int]:
        rep_idx = int(max(0, rep - 1))
        for idx, phase in enumerate(phase_schedule):
            st = int(phase.get("start_rep", 0))
            en = int(phase.get("end_rep", st))
            if st <= rep_idx < en:
                return (
                    str(phase.get("name", f"phase_{idx}")),
                    str(phase.get("phrase", "")),
                    int(idx),
                )
        last = phase_schedule[-1]
        return (
            str(last.get("name", "phase_last")),
            str(last.get("phrase", "")),
            int(len(phase_schedule) - 1),
        )

    def _summarize_history(
        self,
        history: Sequence[Dict[str, Any]],
        phase_schedule: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not history:
            return {
                "phase_stats": [],
                "shift_stats": [],
                "worst_drop_top1": 0.0,
                "mean_recovery_reps": 0.0,
                "final_top1": 0.0,
                "final_top3": 0.0,
                "final_mrr": 0.0,
            }

        phase_stats: list[dict[str, Any]] = []
        shift_stats: list[dict[str, Any]] = []
        for idx, phase in enumerate(phase_schedule):
            name = str(phase.get("name", f"phase_{idx}"))
            rows = [r for r in history if str(r.get("phase")) == name]
            if not rows:
                continue
            top1_vals = [float(r.get("top1", 0.0)) for r in rows]
            top3_vals = [float(r.get("top3", 0.0)) for r in rows]
            mrr_vals = [float(r.get("mrr", 0.0)) for r in rows]
            phase_stats.append(
                {
                    "name": name,
                    "phase_idx": int(idx),
                    "mean_top1": float(np.mean(top1_vals)),
                    "mean_top3": float(np.mean(top3_vals)),
                    "mean_mrr": float(np.mean(mrr_vals)),
                    "initial_top1": float(top1_vals[0]),
                    "final_top1": float(top1_vals[-1]),
                }
            )

        for idx in range(1, len(phase_stats)):
            prev_s = phase_stats[idx - 1]
            cur_s = phase_stats[idx]
            drop = float(cur_s["initial_top1"] - prev_s["final_top1"])
            target = float(prev_s["final_top1"]) * 0.8
            cur_name = str(cur_s["name"])
            cur_rows = [r for r in history if str(r.get("phase")) == cur_name]
            recovery = None
            if cur_rows:
                rep0 = int(cur_rows[0].get("rep", 0))
                for r in cur_rows:
                    if float(r.get("top1", 0.0)) >= target:
                        recovery = int(r.get("rep", rep0)) - rep0
                        break
            shift_stats.append(
                {
                    "from": str(prev_s["name"]),
                    "to": cur_name,
                    "drop_top1": float(drop),
                    "recovery_reps": recovery,
                }
            )

        recovery_vals = [
            float(item["recovery_reps"])
            for item in shift_stats
            if item.get("recovery_reps") is not None
        ]
        worst_drop = min(
            [float(item["drop_top1"]) for item in shift_stats], default=0.0
        )
        last = history[-1]
        return {
            "phase_stats": phase_stats,
            "shift_stats": shift_stats,
            "worst_drop_top1": float(worst_drop),
            "mean_recovery_reps": float(np.mean(recovery_vals))
            if recovery_vals
            else -1.0,
            "final_top1": float(last.get("top1", 0.0)),
            "final_top3": float(last.get("top3", 0.0)),
            "final_mrr": float(last.get("mrr", 0.0)),
        }

    def _predict_next_byte(
        self,
        token_ids: np.ndarray,
        energies: np.ndarray,
        context_bytes: List[int],
    ) -> List[Tuple[int, float]]:
        """Predict next byte using observed byte-context matching."""
        if len(token_ids) < self.context_length + 1:
            return []
        if len(context_bytes) != self.context_length:
            return []

        obs_bytes = (token_ids.astype(np.uint64) & np.uint64(0xFF)).astype(np.int64)
        target = np.asarray(context_bytes, dtype=np.int64)
        scores = np.zeros(256, dtype=np.float32)
        n_particles = len(obs_bytes)
        context_len = len(context_bytes)

        for start_idx in range(n_particles - context_len):
            window = obs_bytes[start_idx : start_idx + context_len]
            if not np.array_equal(window, target):
                continue
            next_idx = start_idx + context_len
            if next_idx >= n_particles:
                continue
            next_tid = int(token_ids[next_idx])
            next_energy = float(energies[next_idx]) if next_idx < len(energies) else 1.0
            byte_val = int(next_tid & 0xFF)
            if 0 <= byte_val < 256:
                scores[byte_val] += next_energy

        if scores.sum() > 0:
            scores = scores / scores.sum()

        top_indices = np.argsort(scores)[::-1][:5]
        return [
            (int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0
        ]

    def _dehash(self, token_id: int, position: int) -> int:
        """Compatibility shim for legacy callers."""
        _ = position
        return int(token_id) & 0xFF
