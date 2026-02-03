"""Kernel-based rule-shift experiment (Metal/MPS).

Runs the *new* implementation:
- Particle/grid physics: `optimizer.manifold_physics.ManifoldPhysics` (Metal kernels)
- Spectral carrier memory: `optimizer.manifold_physics.SpectralCarrierPhysics` (Metal kernels)

Produces paper-ready artifacts:
- `paper/tables/rule_shift_summary.tex`
- `paper/figures/rule_shift.pdf`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from optimizer.manifold_physics import (
    ManifoldPhysics,
    ManifoldPhysicsConfig,
    SpectralCarrierPhysics,
    SpectralCarrierConfig,
)
from sensorium.core.tokenizer import UniversalTokenizer, UniversalTokenizerConfig


@dataclass(frozen=True, slots=True)
class KernelRuleShiftConfig:
    device: str = "mps"
    grid_size: tuple[int, int, int] = (32, 32, 32)
    dt: float = 0.02
    steps: int = 2000
    shift_at: int = 1000

    # Injection
    particles_per_token: int = 8
    injection_spread: float = 0.6
    energy_scale: float = 1.0

    # Carrier update cadence
    carrier_every: int = 5

    # Universal tokenizer
    hash_vocab_size: int = 4096

    # Omega mapping for token ids -> oscillator excitation frequency
    omega_range: float = 2.0
    omega_bins: int = 64


def _token_stream(tokenizer: UniversalTokenizer) -> Tuple[List[int], List[int]]:
    # A deterministic phrase; forward is bytes in-order, reverse is bytes reversed.
    phrase = "<bos> The cat sat on the mat <eos>"
    ids = tokenizer.encode_text(phrase, add_bos_eos=False).to(torch.long).tolist()
    if len(ids) < 2:
        ids = [tokenizer.bos_id, tokenizer.eos_id]
    return ids, list(reversed(ids))


def _omega_from_id(token_id: int, *, omega_range: float) -> float:
    # Map integer IDs into [0, omega_range) deterministically.
    m = 2048
    return float((int(token_id) % m) / float(m) * float(omega_range))


def _center_from_id(token_id: int, grid_size: tuple[int, int, int]) -> torch.Tensor:
    # 3D integer hash -> center coordinate inside the grid.
    gx, gy, gz = grid_size
    x = (int(token_id) * 73856093) % gx
    y = (int(token_id) * 19349663) % gy
    z = (int(token_id) * 83492791) % gz
    # Keep away from borders for stability.
    return torch.tensor([1 + (x % max(1, gx - 2)), 1 + (y % max(1, gy - 2)), 1 + (z % max(1, gz - 2))], dtype=torch.float32)


def _hist_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    num = float((a * b).sum().item())
    den = float(torch.sqrt((a * a).sum() + eps).item() * torch.sqrt((b * b).sum() + eps).item())
    return num / (den + eps)


def _expected_hist(
    token_ids: List[int],
    *,
    omega_bins: int,
    omega_range: float,
) -> torch.Tensor:
    # CPU histogram of expected omegas (uniform weights).
    h = torch.zeros(int(omega_bins), dtype=torch.float32)
    for tid in token_ids:
        w = _omega_from_id(int(tid), omega_range=omega_range)
        bi = int(min(omega_bins - 1, max(0, (w / omega_range) * omega_bins)))
        h[bi] += 1.0
    if float(h.sum().item()) > 0:
        h /= h.sum()
    return h


def _carrier_hist(
    omega: torch.Tensor,
    amp: torch.Tensor,
    *,
    omega_bins: int,
    omega_range: float,
) -> torch.Tensor:
    # CPU histogram of carriers, weighted by amplitude.
    om = omega.detach().to("cpu", dtype=torch.float32).clamp(0.0, float(omega_range))
    aa = amp.detach().to("cpu", dtype=torch.float32).clamp(min=0.0)
    h = torch.zeros(int(omega_bins), dtype=torch.float32)
    if int(om.numel()) == 0:
        return h
    for o, a in zip(om.tolist(), aa.tolist()):
        bi = int(min(omega_bins - 1, max(0, (float(o) / omega_range) * omega_bins)))
        h[bi] += float(a)
    if float(h.sum().item()) > 0:
        h /= h.sum()
    return h


def run_kernel_rule_shift(
    cfg: KernelRuleShiftConfig,
    *,
    spectral_cfg: Optional[SpectralCarrierConfig] = None,
    out_dir: Path = Path("./paper"),
) -> Dict[str, Any]:
    """Run rule shift and write paper artifacts."""
    device = cfg.device
    if device != "mps":
        raise RuntimeError("Kernel rule shift currently expects device='mps' (Metal).")

    tok = UniversalTokenizer(UniversalTokenizerConfig(hash_vocab_size=int(cfg.hash_vocab_size), num_labels=0))
    fwd, rev = _token_stream(tok)
    exp_fwd = _expected_hist(fwd, omega_bins=cfg.omega_bins, omega_range=cfg.omega_range)
    exp_rev = _expected_hist(rev, omega_bins=cfg.omega_bins, omega_range=cfg.omega_range)

    # Physics + carriers (Metal kernels)
    mp_cfg = ManifoldPhysicsConfig(grid_size=cfg.grid_size, dt=float(cfg.dt), poisson_iterations=25, device=device)
    physics = ManifoldPhysics(mp_cfg, device=device)

    scfg = spectral_cfg or SpectralCarrierConfig(
        max_carriers=64,
        coupling_scale=0.25,
        carrier_reg=0.15,
        temperature=0.01,
        conflict_threshold=0.35,
        offender_weight_floor=1e-3,
        ema_alpha=0.10,
        recenter_alpha=0.10,
        gate_width_init=0.35,
        gate_width_min=0.08,
        gate_width_max=1.25,
        # memory/topdown defaults are in dataclass
    )
    carriers = SpectralCarrierPhysics(config=scfg, grid_size=cfg.grid_size, dt=float(cfg.dt), device=device)

    # State tensors on MPS
    dev = torch.device(device)
    dtype = torch.float32
    positions = torch.empty((0, 3), device=dev, dtype=dtype)
    velocities = torch.empty((0, 3), device=dev, dtype=dtype)
    energies = torch.empty((0,), device=dev, dtype=dtype)
    heats = torch.empty((0,), device=dev, dtype=dtype)
    excitations = torch.empty((0,), device=dev, dtype=dtype)
    masses = torch.empty((0,), device=dev, dtype=dtype)
    osc_phase = torch.empty((0,), device=dev, dtype=dtype)

    # Time series (CPU)
    score_fwd: List[float] = []
    score_rev: List[float] = []
    total_energy: List[float] = []
    total_heat: List[float] = []
    num_carriers: List[int] = []
    num_crystallized: List[int] = []

    two_pi = float(2.0 * torch.pi)

    for t in range(int(cfg.steps)):
        seq = fwd if t < int(cfg.shift_at) else rev
        token_id = int(seq[t % len(seq)])

        # Inject a small burst for this token.
        n = int(cfg.particles_per_token)
        center = _center_from_id(token_id, cfg.grid_size).to(device=dev, dtype=dtype)
        pos = center.view(1, 3) + torch.randn(n, 3, device=dev, dtype=dtype) * float(cfg.injection_spread)
        pos = pos.clamp(0.5, float(min(cfg.grid_size) - 1.5))
        vel = torch.randn(n, 3, device=dev, dtype=dtype) * 0.05
        en = torch.full((n,), float(cfg.energy_scale), device=dev, dtype=dtype)
        ht = torch.zeros((n,), device=dev, dtype=dtype)
        om = float(_omega_from_id(token_id, omega_range=cfg.omega_range))
        ex = torch.full((n,), om, device=dev, dtype=dtype) + torch.randn(n, device=dev, dtype=dtype) * 0.01
        ms = en.clone()
        ph = torch.rand(n, device=dev, dtype=dtype) * two_pi

        positions = torch.cat([positions, pos], dim=0)
        velocities = torch.cat([velocities, vel], dim=0)
        energies = torch.cat([energies, en], dim=0)
        heats = torch.cat([heats, ht], dim=0)
        excitations = torch.cat([excitations, ex], dim=0)
        masses = torch.cat([masses, ms], dim=0)
        osc_phase = torch.cat([osc_phase, ph], dim=0)

        # Physics step (Metal kernels)
        positions, velocities, energies, heats, excitations = physics.step(
            positions, velocities, energies, heats, excitations, masses
        )

        # Carrier update cadence (Metal kernels)
        if (t % int(cfg.carrier_every)) == 0 and int(osc_phase.numel()) > 0:
            cst = carriers.step(osc_phase, excitations, energies)
            osc_phase = cst["osc_phase"]
            # alignment scores
            h = _carrier_hist(
                cst["frequencies"],
                cst["amplitudes"],
                omega_bins=cfg.omega_bins,
                omega_range=cfg.omega_range,
            )
            score_fwd.append(_hist_cosine(h, exp_fwd))
            score_rev.append(_hist_cosine(h, exp_rev))
            num_carriers.append(int(cst["frequencies"].numel()))
            cs = cst.get("carrier_state")
            if cs is None:
                num_crystallized.append(0)
            else:
                num_crystallized.append(int((cs.detach().to("cpu") == 2).sum().item()))
        else:
            # Repeat last value for plotting continuity.
            score_fwd.append(score_fwd[-1] if score_fwd else 0.0)
            score_rev.append(score_rev[-1] if score_rev else 0.0)
            num_carriers.append(num_carriers[-1] if num_carriers else 0)
            num_crystallized.append(num_crystallized[-1] if num_crystallized else 0)

        total_energy.append(float(energies.sum().detach().to("cpu").item()))
        total_heat.append(float(heats.sum().detach().to("cpu").item()))

    # Summaries
    s_at = int(cfg.shift_at)
    pre_win = 200
    post_win = 50
    pre_fwd = float(torch.tensor(score_fwd[max(0, s_at - pre_win) : s_at]).mean().item()) if s_at > 0 else float(score_fwd[-1])
    post_immediate = float(torch.tensor(score_rev[s_at : min(len(score_rev), s_at + post_win)]).mean().item()) if s_at < len(score_rev) else float(score_rev[-1])
    threshold = 0.8 * pre_fwd
    recovery_steps: Optional[int] = None
    for i in range(s_at, len(score_rev)):
        if float(score_rev[i]) >= threshold:
            recovery_steps = int(i - s_at)
            break

    metrics = {
        "steps": int(cfg.steps),
        "shift_at": int(cfg.shift_at),
        "pre_shift_alignment": float(pre_fwd),
        "post_shift_alignment_immediate": float(post_immediate),
        "recovery_steps": recovery_steps,
        "final_carriers": int(num_carriers[-1]) if num_carriers else 0,
        "final_crystallized": int(num_crystallized[-1]) if num_crystallized else 0,
        "score_fwd": score_fwd,
        "score_rev": score_rev,
        "energy": total_energy,
        "heat": total_heat,
        "num_carriers": num_carriers,
        "num_crystallized": num_crystallized,
    }

    # Write paper artifacts
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = r"""\begin{table}[t]
\centering
\caption{Kernel rule-shift adaptation. We measure alignment between the carrier spectrum and the expected regime spectrum before and after a reversal at step """ + str(int(cfg.shift_at)) + r""".}
\label{tab:rule_shift}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Pre-shift alignment & """ + f"{pre_fwd:.3f}" + r""" \\
Post-shift alignment (immediate) & """ + f"{post_immediate:.3f}" + r""" \\
Steps to 80\% recovery & """ + (str(recovery_steps) if recovery_steps is not None else "N/A") + r""" \\
Final carrier count & """ + f"{metrics['final_carriers']}" + r""" \\
Final crystallized carriers & """ + f"{metrics['final_crystallized']}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    (tables_dir / "rule_shift_summary.tex").write_text(table, encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = list(range(int(cfg.steps)))
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        ax1.plot(x, score_fwd, label="alignment to fwd", linewidth=1.2)
        ax1.plot(x, score_rev, label="alignment to rev", linewidth=1.2)
        ax1.axvline(int(cfg.shift_at), color="r", linestyle="--", alpha=0.7)
        ax1.set_ylabel("alignment")
        ax1.legend(loc="upper right", fontsize=8)

        ax2.plot(x, total_energy, label="total energy", alpha=0.8)
        ax2.plot(x, total_heat, label="total heat", alpha=0.8)
        ax2.axvline(int(cfg.shift_at), color="r", linestyle="--", alpha=0.7)
        ax2.set_ylabel("energy / heat")
        ax2.legend(loc="upper right", fontsize=8)

        ax3.plot(x, num_carriers, label="#carriers", alpha=0.9)
        ax3.plot(x, num_crystallized, label="#crystallized", alpha=0.9)
        ax3.axvline(int(cfg.shift_at), color="r", linestyle="--", alpha=0.7)
        ax3.set_ylabel("count")
        ax3.set_xlabel("step")
        ax3.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(figures_dir / "rule_shift.pdf", format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        # Best-effort: paper will show placeholder box if figure missing.
        pass

    return metrics

