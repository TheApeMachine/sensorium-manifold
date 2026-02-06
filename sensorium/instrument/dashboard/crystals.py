"""Crystal Content panel -- decoded bytes + compact metrics (dark theme)."""

from __future__ import annotations

import numpy as np


_VOCAB_SIZE = 2 ** 16
_PRIME = 1315423911
_MASK = _VOCAB_SIZE - 1
_INV_PRIME = pow(_PRIME, -1, _VOCAB_SIZE)
_MODE_ANCHORS = 8

_MAX_CRYSTAL_LINES = 10
_MAX_METRIC_LINES = 5


def _dehash(token_id: int, position: int) -> int:
    target = (token_id - position) & _MASK
    byte_val = (target * _INV_PRIME) & _MASK
    return byte_val if byte_val < 256 else -1


def _bytes_repr(raw: list[int], max_len: int = 16) -> str:
    if not raw:
        return "(empty)"
    hex_part = " ".join(f"{b:02x}" for b in raw[:max_len])
    ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in raw[:max_len])
    suffix = " ..." if len(raw) > max_len else ""
    return f"{hex_part}  |{ascii_part}|{suffix}"


class CrystalContentPlot:
    """Text panel showing decoded crystal content and simulation metrics."""

    def __init__(self, ax) -> None:
        self.ax = ax
        ax.axis("off")
        ax.set_facecolor("#0e0e1a")

        self._title = ax.text(
            0.01, 0.96, "Crystals", fontsize=9, fontweight="bold",
            color="#f39c12", va="top", ha="left",
            transform=ax.transAxes, fontfamily="monospace",
        )

        self._crystal_texts: list[object] = []
        for i in range(_MAX_CRYSTAL_LINES):
            y = 0.88 - i * 0.072
            txt = ax.text(
                0.02, y, "", fontsize=6.5, color="#ddd",
                va="top", ha="left", transform=ax.transAxes,
                fontfamily="monospace",
            )
            self._crystal_texts.append(txt)

        sep_y = 0.88 - _MAX_CRYSTAL_LINES * 0.072 - 0.005
        self._sep = ax.text(
            0.01, sep_y, "",
            fontsize=5, color="#555", va="top", ha="left",
            transform=ax.transAxes, fontfamily="monospace",
        )

        self._metric_texts: list[object] = []
        y_start = sep_y - 0.025
        for i in range(_MAX_METRIC_LINES):
            y = y_start - i * 0.05
            txt = ax.text(
                0.02, y, "", fontsize=6.5, color="#aaa",
                va="top", ha="left", transform=ax.transAxes,
                fontfamily="monospace",
            )
            self._metric_texts.append(txt)

    def update(self, state: dict) -> None:
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            return v

        def get_scalar(key, default=0.0):
            v = state.get(key, default)
            if v is None:
                return default
            if hasattr(v, "item"):
                return float(v.item())
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        mode_state_arr = get("mode_state")
        omega = get("omega_lattice")
        psi_amp = get("psi_amplitude")
        anchor_idx = get("mode_anchor_idx")
        anchor_w = get("mode_anchor_weight")
        token_ids = get("token_ids")
        seq_indices = get("sequence_indices")

        crystal_lines: list[str] = []

        if (
            mode_state_arr is not None
            and omega is not None
            and psi_amp is not None
            and anchor_idx is not None
            and token_ids is not None
            and seq_indices is not None
        ):
            ms = mode_state_arr.astype(np.int32)
            amp = psi_amp.astype(np.float64)
            omega_arr = omega.astype(np.float64)
            a_idx = anchor_idx.astype(np.int32)
            a_w = np.abs(anchor_w.astype(np.float64)) if anchor_w is not None else np.zeros_like(a_idx, dtype=np.float64)
            tid = token_ids.astype(np.int64)
            sid = seq_indices.astype(np.int64)
            M = len(ms)
            N = len(tid)

            crystal_mask = np.where(ms == 2)[0]
            crystal_sorted = crystal_mask[np.argsort(amp[crystal_mask])[::-1]] if len(crystal_mask) > 0 else np.array([], dtype=np.int32)

            slots = len(a_idx) // max(M, 1) if M > 0 else _MODE_ANCHORS

            for k in crystal_sorted[:_MAX_CRYSTAL_LINES]:
                k = int(k)
                base = k * slots
                decoded: list[int] = []
                n_anchors = 0
                for s in range(min(slots, _MODE_ANCHORS)):
                    pidx = int(a_idx[base + s])
                    if pidx < 0 or pidx >= N:
                        continue
                    w = float(a_w[base + s])
                    if w <= 0:
                        continue
                    n_anchors += 1
                    byte_val = _dehash(int(tid[pidx]), int(sid[pidx]))
                    if 0 <= byte_val < 256:
                        decoded.append(byte_val)

                crystal_lines.append(
                    f"[{k:>3}] w={omega_arr[k]:.3f} |Y|={amp[k]:.1f} "
                    f"({n_anchors}p): {_bytes_repr(decoded)}"
                )

        n_crystals = len(crystal_lines)
        self._title.set_text(f"Crystallized Modes ({n_crystals})" if n_crystals else "Crystallized Modes (none yet)")

        for i, txt in enumerate(self._crystal_texts):
            txt.set_text(crystal_lines[i] if i < len(crystal_lines) else "")

        self._sep.set_text("- " * 40 if n_crystals else "")

        # Metrics
        step = get_scalar("step", 0)
        positions = get("positions")
        n_particles = int(positions.shape[0]) if positions is not None and hasattr(positions, "shape") else 0

        energy = get("energies", np.array([]))
        heat = get("heats", np.array([]))
        masses = get("masses")
        velocities = get("velocities", np.zeros((0, 3)))
        e_mode = float(energy.sum()) if len(energy) > 0 else 0.0
        eheat = float(heat.sum()) if len(heat) > 0 else 0.0
        m_eff = masses if masses is not None else energy
        v2 = (velocities ** 2).sum(axis=1) if len(velocities) > 0 else np.array([])
        ekin = float(0.5 * np.sum(m_eff * v2)) if len(v2) > 0 else 0.0

        dt = get_scalar("dt", 0.0)
        kuramoto_R = 0.0
        psi_real = get("psi_real")
        psi_imag = get("psi_imag")
        if psi_real is not None and psi_imag is not None:
            phases = np.arctan2(psi_imag.astype(np.float64), psi_real.astype(np.float64))
            kuramoto_R = float(np.abs(np.mean(np.exp(1j * phases))))

        n_modes = len(mode_state_arr) if mode_state_arr is not None else 0
        n_stable = int(np.sum(mode_state_arr == 1)) if mode_state_arr is not None else 0
        n_crystal_m = int(np.sum(mode_state_arr == 2)) if mode_state_arr is not None else 0

        metric_lines = [
            f"Step {int(step):>6}  |  {n_particles}p  {n_modes}m  |  dt={dt:.4g}",
            f"E: mode={e_mode:.1f}  heat={eheat:.1f}  kin={ekin:.1f}  tot={e_mode+eheat+ekin:.1f}",
            f"R={kuramoto_R:.3f}  |  {n_modes-n_stable-n_crystal_m}n {n_stable}s {n_crystal_m}c",
        ]

        for i, txt in enumerate(self._metric_texts):
            txt.set_text(metric_lines[i] if i < len(metric_lines) else "")
