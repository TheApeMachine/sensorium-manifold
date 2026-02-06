"""Wave-space observables for the coherence field."""

from __future__ import annotations

from typing import Any, Dict

import torch


class WaveFieldMetrics:
    """Participation ratio / entropy over |Psi|^2 and settling signal."""

    def observe(self, state: dict | None = None, **kwargs) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return {}

        amp = state.get("psi_amplitude", None)
        if amp is None or not hasattr(amp, "numel") or int(amp.numel()) == 0:
            # Still forward psi_delta_rel if present.
            out: Dict[str, Any] = {}
            if "psi_delta_rel" in state:
                try:
                    out["psi_delta_rel"] = float(state["psi_delta_rel"])
                except Exception:
                    pass
            return out

        a = amp.to(torch.float32)
        power = a * a
        s1 = power.sum()
        s2 = (power * power).sum()
        s1_f = float(s1.detach().item())
        s2_f = float(s2.detach().item())
        if s1_f > 0.0 and s2_f > 0.0:
            pr = float((s1_f * s1_f) / s2_f)
            p = torch.clamp(power / s1, min=0.0)
            H = float((-(p * torch.log2(torch.clamp(p, min=1e-30))).sum()).detach().item())
        else:
            pr = 0.0
            H = 0.0

        out = {
            "mode_participation": pr,
            "mode_entropy": H,
        }
        if "psi_delta_rel" in state:
            try:
                out["psi_delta_rel"] = float(state["psi_delta_rel"])
            except Exception:
                pass
        return out

