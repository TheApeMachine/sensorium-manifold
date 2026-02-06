"""Labelled ω-spectrum probes (visualization support).

Goal: "identify the waves of a word within a sentence" by annotating the current
ω-spectrum with human-readable labels derived from (sequence_position, byte_value)
for one sample.

This is measurement-only and intentionally small (exports <= ~512 floats).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import zlib

import torch


def _byte_label(b: int) -> str:
    b = int(b) & 0xFF
    if 32 <= b <= 126 and b not in (92,):  # printable; avoid "\" in labels
        return chr(b)
    return f"0x{b:02x}"


@dataclass(frozen=True, slots=True)
class OmegaLabelProbeConfig:
    # If None, pick a deterministic pseudo-random sample per run (seeded by run_name).
    sample_id: int | None = 0
    max_labels: int = 16
    # If True, label "pos:char"; else just "char@pos"
    verbose: bool = False
    # If True, shuffle which tokens get labelled (still deterministic if sample_id is None).
    shuffle: bool = True


class CoherenceSpectrumSnapshot:
    """Export ω-lattice + |Ψ(ω)| for plotting (small CPU lists)."""

    def __init__(self, *, max_modes: int = 512):
        self.max_modes = int(max_modes)

    def observe(self, state: dict | None = None, **_kwargs) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return {}
        omega = state.get("omega_lattice", None)
        amp = state.get("psi_amplitude", None)
        if omega is None or amp is None:
            return {}
        if not hasattr(omega, "detach") or not hasattr(amp, "detach"):
            return {}
        if int(omega.numel()) == 0 or int(amp.numel()) == 0:
            return {}
        if int(omega.numel()) != int(amp.numel()):
            return {}
        if int(omega.numel()) > self.max_modes:
            return {}

        om = omega.detach().to("cpu").to(torch.float32).tolist()
        a = amp.detach().to("cpu").to(torch.float32).tolist()
        return {
            "spectrum_omega": om,
            "spectrum_amp": a,
        }


class OmegaLabelProbe:
    """Return labelled ω markers for a chosen sample.

    Labels are derived from the literal byte stream, so they remain meaningful even
    when token_ids are hashed.
    """

    def __init__(self, config: OmegaLabelProbeConfig | None = None):
        self.config = config or OmegaLabelProbeConfig()

    def observe(self, state: dict | None = None, **kwargs) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return {}

        omega_lattice = state.get("omega_lattice", None)
        if omega_lattice is None or not hasattr(omega_lattice, "detach"):
            return {}
        omega_lattice_cpu = omega_lattice.detach().to("cpu").to(torch.float32)
        if int(omega_lattice_cpu.numel()) == 0:
            return {}

        b = state.get("byte_values", None)
        t = state.get("sequence_indices", None)
        s = state.get("sample_indices", None)
        omega_p = state.get("omega", None)
        if b is None or t is None or s is None or omega_p is None:
            return {}
        if not (hasattr(b, "detach") and hasattr(t, "detach") and hasattr(s, "detach") and hasattr(omega_p, "detach")):
            return {}

        b_cpu = b.detach().to("cpu").to(torch.int64)
        t_cpu = t.detach().to("cpu").to(torch.int64)
        s_cpu = s.detach().to("cpu").to(torch.int64)
        om_cpu = omega_p.detach().to("cpu").to(torch.float32)

        n = int(b_cpu.numel())
        if n == 0:
            return {}

        # Choose sample id.
        if self.config.sample_id is None:
            # Deterministic pseudo-random choice (reproducible, anti-cherrypick).
            run = kwargs.get("run_name", "")
            run_s = run if isinstance(run, str) else ""
            # Bound by observed max sample id.
            n_samples = int(torch.max(s_cpu).item()) + 1 if int(s_cpu.numel()) else 1
            if n_samples <= 0:
                return {}
            sid = int(zlib.adler32(run_s.encode("utf-8")) % int(n_samples))
        else:
            sid = int(self.config.sample_id)
        mask = (s_cpu == sid)
        if not bool(mask.any().item()):
            return {}

        idx = torch.nonzero(mask, as_tuple=False).flatten()
        # Candidate ordering.
        # - For readability: start from sequence order
        # - For anti-cherrypick: optional deterministic shuffle within the sample.
        order = torch.argsort(t_cpu[idx])
        idx = idx[order]
        if bool(self.config.shuffle) and int(idx.numel()) > 1:
            run = kwargs.get("run_name", "")
            run_s = run if isinstance(run, str) else ""
            # Use a stable seed derived from run name + sample id.
            seed = int(zlib.adler32(f"{run_s}:{sid}".encode("utf-8")) & 0xFFFFFFFF)
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            perm = torch.randperm(int(idx.numel()), generator=g)
            idx = idx[perm]

        # Take at most max_labels unique (pos, byte) pairs.
        max_labels = int(max(1, self.config.max_labels))
        seen: set[tuple[int, int]] = set()
        labels: List[Dict[str, Any]] = []

        om_min = float(omega_lattice_cpu.min().item())
        om_max = float(omega_lattice_cpu.max().item())
        M = int(omega_lattice_cpu.numel())
        domega = (om_max - om_min) / float(max(M - 1, 1))
        if not (domega > 0.0):
            return {}

        for i in idx.tolist():
            pos_i = int(t_cpu[i].item())
            byte_i = int(b_cpu[i].item()) & 0xFF
            key = (pos_i, byte_i)
            if key in seen:
                continue
            seen.add(key)

            omega_i = float(om_cpu[i].item())
            # Nearest ω-bin index on the lattice.
            k = int(round((omega_i - om_min) / domega))
            k = 0 if k < 0 else (M - 1 if k >= M else k)
            om_k = float(omega_lattice_cpu[k].item())

            ch = _byte_label(byte_i)
            if self.config.verbose:
                lab = f"pos{pos_i}:{ch}"
            else:
                lab = f"{ch}@{pos_i}"

            labels.append(
                {
                    "label": lab,
                    "pos": pos_i,
                    "byte": byte_i,
                    "omega": om_k,
                    "k": k,
                }
            )
            if len(labels) >= max_labels:
                break

        return {"omega_labels": labels, "omega_labels_sample_id": int(sid)}

