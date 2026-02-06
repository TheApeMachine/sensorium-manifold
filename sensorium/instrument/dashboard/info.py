from __future__ import annotations

import numpy as np
from tensordict import TensorDict


class InfoPlot:
    """Info text block for the simulation dashboard.
    
    Enhanced to show OmegaWave-specific metrics:
    - Mode statistics (counts by state)
    - Kuramoto order parameter
    - Spatial coherence σ_x
    - Energy partition
    """

    def __init__(self, ax) -> None:
        self.ax = ax
        self._step = 0
        self.ax.axis('off')
        # Single text artist updated each frame.
        self._text = self.ax.text(
            0.05,
            0.98,
            "",
            transform=self.ax.transAxes,
            fontsize=7.5,
            va="top",
            family="monospace",
            linespacing=1.15,
        )

    def update(self, state: TensorDict) -> None:
        """Update the info text from state dict."""
        # Do not clear axes; just update text.
        
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            return v
        
        def get_scalar(key, default=0.0):
            v = state.get(key, default)
            if v is None:
                return default
            if hasattr(v, 'item'):
                return float(v.item())
            try:
                return float(v)
            except (TypeError, ValueError):
                return default
        
        energy = get("energies", np.array([]))  # oscillator / mode energy (not thermal)
        heat = get("heats", np.array([]))
        masses = get("masses")
        velocities = get("velocities", np.zeros((0, 3)))
        positions = get("positions", np.zeros((0, 3)))
        c_v = get_scalar("c_v", 0.0)
        
        # OmegaWave fields
        omega_bins = get("omega_lattice")
        linewidth = get("mode_linewidth")
        psi_amp = get("psi_amplitude")
        psi_real = get("psi_real")
        psi_imag = get("psi_imag")
        mode_state = get("mode_state")
        mode_conflict = get("mode_conflict")
        sigma_x = get_scalar("spatial_sigma", 0.0)
        
        # ----- Energy statistics -----
        total_emode = float(energy.sum()) if len(energy) > 0 else 0.0
        total_h = float(heat.sum()) if len(heat) > 0 else 0.0
        
        m_eff = masses if masses is not None else energy
        v2 = (velocities ** 2).sum(axis=1) if len(velocities) > 0 else np.array([])
        total_ekin = float(0.5 * np.sum(m_eff * v2)) if len(v2) > 0 else 0.0
        total_E = total_emode + total_h + total_ekin
        
        num_modes = len(omega_bins) if omega_bins is not None else 0
        num_particles = len(positions)
        
        # ----- Temperature -----
        T_mean = 0.0
        if c_v > 0.0 and masses is not None and len(heat) > 0:
            m = masses.astype(np.float64) if hasattr(masses, "astype") else np.asarray(masses, dtype=np.float64)
            h = heat.astype(np.float64)
            denom = m * float(c_v)
            valid = denom > 0
            if np.any(valid):
                T_mean = float(np.mean(h[valid] / denom[valid]))
        
        # ----- Kuramoto order parameter -----
        kuramoto_R = 0.0
        if psi_real is not None and psi_imag is not None:
            phases = np.arctan2(psi_imag.astype(np.float64), psi_real.astype(np.float64))
            order = np.mean(np.exp(1j * phases))
            kuramoto_R = float(np.abs(order))
        
        # ----- Mode state counts -----
        n_neutral = n_stable = n_crystal = 0
        if mode_state is not None and len(mode_state) > 0:
            ms = mode_state.astype(np.int32)
            n_neutral = int(np.sum(ms == 0))
            n_stable = int(np.sum(ms == 1))
            n_crystal = int(np.sum(ms == 2))
        
        # ----- Linewidth stats -----
        gamma_mean = gamma_std = 0.0
        if linewidth is not None and len(linewidth) > 0:
            gamma_mean = float(np.mean(linewidth))
            gamma_std = float(np.std(linewidth))
        
        # ----- Mean conflict -----
        mean_conflict = 0.0
        if mode_conflict is not None and len(mode_conflict) > 0:
            mean_conflict = float(np.mean(mode_conflict))
        
        # ----- Psi power -----
        psi_power = 0.0
        if psi_amp is not None and len(psi_amp) > 0:
            psi_power = float(np.sum(psi_amp ** 2))
        
        # Build display text
        step = state.get("step", 0)
        if hasattr(step, 'item'):
            step = step.item()
        
        lines = [
            f"Step: {step:,}",
            f"N: {num_particles} particles",
            f"M: {num_modes} modes",
            "",
            "─── Energy ───",
            f"E_mode:  {total_emode:>8.2f}",
            f"E_heat:  {total_h:>8.2f}",
            f"E_kin:   {total_ekin:>8.2f}",
            f"Total:   {total_E:>8.2f}",
        ]

        cfl_any_bad = get_scalar("cfl_any_bad_rate", 0.0)
        cfl_max_rate = get_scalar("cfl_max_rate", 0.0)
        dt = get_scalar("dt", 0.0)
        if dt > 0.0:
            lines.append(f"dt:      {dt:>10.3g}")
        if cfl_any_bad > 0.0:
            lines.append(f"CFL:      BAD rate (max={cfl_max_rate:.3g})")
        
        if T_mean > 0:
            lines.append(f"T̄:      {T_mean:>8.3g}")
        
        lines.extend([
            "",
            "─── ω-Field ───",
            f"Kuramoto R: {kuramoto_R:.3f}",
            f"σ_x:        {sigma_x:.4f}",
            f"|Ψ|² power: {psi_power:.2f}",
            f"⟨conflict⟩: {mean_conflict:.3f}",
        ])
        
        if num_modes > 0:
            lines.extend([
                "",
                "─── Modes ───",
                f"neutral:    {n_neutral:>3} ({100*n_neutral/num_modes:4.1f}%)",
                f"stable:     {n_stable:>3} ({100*n_stable/num_modes:4.1f}%)",
                f"crystal:    {n_crystal:>3} ({100*n_crystal/num_modes:4.1f}%)",
            ])
        
        if gamma_mean > 0:
            lines.extend([
                "",
                "─── Linewidth ───",
                f"γ mean: {gamma_mean:.4f}",
                f"γ std:  {gamma_std:.4f}",
            ])

        # ----- Kernel "log book" (GPU debug events) -----
        kl = state.get("kernel_log", "")
        if isinstance(kl, str) and kl.strip():
            lines.extend(["", "─── Kernel Log ───"])
            # Keep it short; the buffer can be large.
            lines.extend(kl.splitlines()[-24:])
        self._text.set_text('\n'.join(lines))
