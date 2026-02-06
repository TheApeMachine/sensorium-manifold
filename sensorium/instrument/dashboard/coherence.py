"""Global coherence metrics visualization for ω-field dynamics.

Shows order parameters and synchronization measures:
- Kuramoto order parameter R (phase synchronization)
- Amplitude variance (mode energy distribution)
- Phase velocity distribution
- Crystallization fraction over time
"""

from __future__ import annotations

from collections import deque

import numpy as np


class CoherencePlot:
    """Time series of global coherence metrics for the ω-field.
    
    This visualization tracks derived order parameters that reveal
    the collective behavior of the ω-field:
    - Kuramoto R: phase synchronization (0=incoherent, 1=locked)
    - σ_x dynamics: spatial coherence length evolution
    - Crystallization: fraction of modes in crystallized state
    - Entropy: information-theoretic measure of mode distribution
    """

    __slots__ = (
        'ax',
        '_history',
        '_step',
        '_line_R',
        '_line_order',
        '_line_conf',
        '_poly_crystal',
        '_poly_stable',
    )

    def __init__(self, ax) -> None:
        self.ax = ax
        self._step = 0
        self._history = {
            "step": deque(maxlen=500),
            "kuramoto_R": deque(maxlen=500),
            "sigma_x": deque(maxlen=500),
            "frac_crystal": deque(maxlen=500),
            "frac_stable": deque(maxlen=500),
            "entropy": deque(maxlen=500),
            "mean_conflict": deque(maxlen=500),
        }
        # Pre-create artists; update via set_data and polygon vertex updates.
        (self._line_R,) = ax.plot([], [], color='#9b59b6', lw=2, label='Kuramoto R', zorder=10)
        # "order (1-H)" line
        (self._line_order,) = ax.plot([], [], color='#2ecc71', lw=1.2, alpha=0.7, label='order (1-H)', ls='--')
        # conflict line
        (self._line_conf,) = ax.plot([], [], color='#e74c3c', lw=1, alpha=0.6, label='⟨conflict⟩', ls=':')
        self._poly_crystal = ax.fill_between([], [], [], color='#f39c12', alpha=0.3, label='crystallized')
        self._poly_stable = ax.fill_between([], [], [], color='#3498db', alpha=0.2, label='stable')
        ax.legend(fontsize=6, loc='upper left', framealpha=0.7, ncol=2)
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Order metrics', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 1.05)

    def update(self, state: dict) -> None:
        """Update coherence metrics from state dict."""
        self._step += 1
        
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            return v
        
        psi_real = get("psi_real")
        psi_imag = get("psi_imag")
        psi_amp = get("psi_amplitude")
        mode_state = get("mode_state")
        mode_conflict = get("mode_conflict")
        sigma_x = state.get("spatial_sigma")
        
        # Default values
        kuramoto_R = 0.0
        frac_crystal = 0.0
        frac_stable = 0.0
        entropy = 0.0
        mean_conflict = 0.0
        sigma_val = 0.0
        
        if psi_real is not None and psi_imag is not None:
            psi_r = psi_real.astype(np.float64)
            psi_i = psi_imag.astype(np.float64)
            
            # Kuramoto order parameter: R = |⟨e^{iθ}⟩|
            phases = np.arctan2(psi_i, psi_r)
            order_complex = np.mean(np.exp(1j * phases))
            kuramoto_R = float(np.abs(order_complex))
        
        if psi_amp is not None:
            amp = psi_amp.astype(np.float64)
            M = len(amp)
            if M > 0:
                # Normalized amplitude distribution for entropy
                amp_norm = amp / (np.sum(amp) + 1e-12)
                amp_norm = amp_norm[amp_norm > 1e-12]  # Remove zeros
                if len(amp_norm) > 0:
                    entropy = float(-np.sum(amp_norm * np.log(amp_norm)))
                    # Normalize by log(M) to get [0, 1] range
                    entropy = entropy / (np.log(M) + 1e-12)
        
        if mode_state is not None:
            ms = mode_state.astype(np.int32)
            M = len(ms)
            if M > 0:
                frac_crystal = float(np.sum(ms == 2)) / M
                frac_stable = float(np.sum(ms == 1)) / M
        
        if mode_conflict is not None:
            mean_conflict = float(np.mean(mode_conflict))
        
        if sigma_x is not None:
            try:
                sigma_val = float(sigma_x)
            except (TypeError, ValueError):
                if hasattr(sigma_x, 'item'):
                    sigma_val = float(sigma_x.item())
        
        # Store history
        self._history["step"].append(self._step)
        self._history["kuramoto_R"].append(kuramoto_R)
        self._history["sigma_x"].append(sigma_val)
        self._history["frac_crystal"].append(frac_crystal)
        self._history["frac_stable"].append(frac_stable)
        self._history["entropy"].append(entropy)
        self._history["mean_conflict"].append(mean_conflict)

        steps = list(self._history["step"])
        
        if len(steps) > 1:
            st = np.asarray(steps, dtype=np.float64)
            
            # Primary axis: Kuramoto R and crystallization
            R_arr = np.asarray(self._history["kuramoto_R"], dtype=np.float64)
            fc_arr = np.asarray(self._history["frac_crystal"], dtype=np.float64)
            fs_arr = np.asarray(self._history["frac_stable"], dtype=np.float64)
            ent_arr = np.asarray(self._history["entropy"], dtype=np.float64)
            conf_arr = np.asarray(self._history["mean_conflict"], dtype=np.float64)

            self._line_R.set_data(st, R_arr)
            self._line_order.set_data(st, 1.0 - ent_arr)
            self._line_conf.set_data(st, conf_arr)

            # Update fill_between polygons (single-path PolyCollections).
            def _set_band(poly, x, y1, y2):
                x = np.asarray(x, dtype=np.float64)
                y1 = np.asarray(y1, dtype=np.float64)
                y2 = np.asarray(y2, dtype=np.float64)
                if x.size == 0:
                    verts = np.zeros((0, 2), dtype=np.float64)
                else:
                    verts = np.concatenate(
                        [
                            np.column_stack([x, y1]),
                            np.column_stack([x[::-1], y2[::-1]]),
                            np.array([[x[0], y1[0]]], dtype=np.float64),
                        ],
                        axis=0,
                    )
                try:
                    poly.get_paths()[0].vertices = verts
                except Exception:
                    pass

            _set_band(self._poly_crystal, st, np.zeros_like(fc_arr), fc_arr)
            _set_band(self._poly_stable, st, fc_arr, fc_arr + fs_arr)
        else:
            self._line_R.set_data([], [])
            self._line_order.set_data([], [])
            self._line_conf.set_data([], [])
        
        # Current values in title
        self.ax.set_title(
            f'Coherence | R={kuramoto_R:.2f} | σ_x={sigma_val:.3f} | '
            f'H={entropy:.2f} | {100*frac_crystal:.0f}% crystal',
            fontsize=8, pad=4
        )
