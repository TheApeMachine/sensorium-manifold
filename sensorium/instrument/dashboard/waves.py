from __future__ import annotations

import numpy as np


class WavesPlot:
    """Wave visualization (VU meter) for ω-mode coupling.
    
    Enhanced to show:
    - Top modes by amplitude (left section: VU bars)
    - Phase coherence visualization (center section: wave interference)
    - Mode state indicators (right section: status badges)
    - Real-time Ψ phase reconstruction (combined wave)
    """

    def __init__(self, ax) -> None:
        self.ax = ax
        # Persistent artists; update via set_data()/set_text() and polygon vertices.
        self._t = np.linspace(0, 4 * np.pi, 400)
        self._t_mask = self._t > 2 * np.pi
        self._t_right = self._t[self._t_mask]

        # Containers for dynamic artists; created on first valid update.
        self._initialized = False
        self._fills = []          # PolyCollections for VU segments + combined fill
        self._lines = []          # Line2D for waves + overlays
        self._texts = []          # Text for badges
        self._legend_handles = [] # Proxy handles for legend
        self._legend = None

    def update(self, state: dict) -> None:
        """Update the wave visualization from state dict."""
        def get(key, default=None):
            v = state.get(key, default)
            if v is not None and hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            return v
        
        omega_bins = get("omega_lattice")
        if omega_bins is None or len(omega_bins) == 0:
            self.ax.set_title('VU Meter: No ω-modes', fontsize=9)
            # Clear existing artists if any
            if getattr(self, "_initialized", False):
                for ln in self._lines:
                    ln.set_data([], [])
                for tx in self._texts:
                    tx.set_text("")
            return
            
        gamma = get("mode_linewidth")
        if gamma is None:
            gamma = np.full(len(omega_bins), 0.35)
        psi_amp = get("psi_amplitude")
        if psi_amp is None:
            psi_amp = np.ones(len(omega_bins))
        psi_amp = np.abs(psi_amp)
        
        psi_phase = get("psi_phase")
        if psi_phase is None:
            psi_phase = np.zeros_like(psi_amp)
            
        mode_state = get("mode_state")
        if mode_state is None:
            mode_state = np.zeros(len(omega_bins), dtype=np.int32)
        else:
            mode_state = mode_state.astype(np.int32)
        
        exc = get("excitations")
        if exc is None:
            exc = np.array([])
        
        # Get particle phases for coherence visualization
        particle_phase = get("phase")
        
        # Sort ω-modes by |Ψ| amplitude, take top 4 for more detail
        top_idx = np.argsort(psi_amp)[-4:][::-1]
        
        t = self._t
        t_right = self._t_right
        mask = self._t_mask
        
        # Track global stats
        total_coupled = 0
        total_coherent = 0

        # Lazy-create a bounded set of artists so we never clear/recreate per frame.
        # We preallocate enough for:
        # - 4 modes
        #   - up to 12 VU segment fills each => 48 fills
        #   - 1 main wave line each => 4 lines
        #   - up to 6 coupled overlay lines each => 24 lines
        #   - optional combined overlay line each mode => 4 lines
        #   - 2 text badges each => 8 texts
        # - combined Ψ fill + line => 1 fill + 1 line
        if not self._initialized:
            # VU segment fills (as fill_between with empty data)
            for _ in range(4 * 12):
                poly = self.ax.fill_between([], [], [], color=(0.25, 0.72, 0.35), alpha=0.0)
                self._fills.append(poly)
            # Main mode wave lines
            for _ in range(4):
                (ln,) = self.ax.plot([], [], lw=1.5, alpha=0.0)
                self._lines.append(ln)
            # Coupled overlay lines (6 per mode)
            for _ in range(4 * 6):
                (ln,) = self.ax.plot([], [], alpha=0.0, lw=0.6, color='#5fbdbd')
                self._lines.append(ln)
            # Combined overlay line (per mode)
            for _ in range(4):
                (ln,) = self.ax.plot([], [], lw=1.2, alpha=0.0, ls='--', color='#2c3e50')
                self._lines.append(ln)
            # Status texts (2 per mode)
            for _ in range(4 * 2):
                tx = self.ax.text(0, 0, "", fontsize=6, va='center')
                self._texts.append(tx)
            # Combined Ψ fill + line
            self._fills.append(self.ax.fill_between([], [], [], color='#9b59b6', alpha=0.0))
            (ln_comb,) = self.ax.plot([], [], color='#9b59b6', lw=1.5, alpha=0.0)
            self._lines.append(ln_comb)

            self.ax.tick_params(labelsize=6)
            self.ax.set_xlabel('Phase / Time', fontsize=7)
            self.ax.set_ylabel('Mode', fontsize=7)
            self._initialized = True

        # Helpers to update PolyCollections produced by fill_between
        def _set_fill(poly, x, ylow, yhigh, *, color=None, alpha=None):
            x = np.asarray(x, dtype=np.float64)
            ylow = np.asarray(ylow, dtype=np.float64)
            yhigh = np.asarray(yhigh, dtype=np.float64)
            if x.size == 0:
                verts = np.zeros((0, 2), dtype=np.float64)
            else:
                verts = np.concatenate(
                    [
                        np.column_stack([x, ylow]),
                        np.column_stack([x[::-1], yhigh[::-1]]),
                        np.array([[x[0], ylow[0]]], dtype=np.float64),
                    ],
                    axis=0,
                )
            try:
                poly.get_paths()[0].vertices = verts
            except Exception:
                pass
            if color is not None:
                try:
                    poly.set_facecolor(color)
                except Exception:
                    pass
            if alpha is not None:
                try:
                    poly.set_alpha(alpha)
                except Exception:
                    pass

        # Reset all artists to invisible/empty; we’ll fill what we use.
        for poly in self._fills:
            try:
                poly.set_alpha(0.0)
            except Exception:
                pass
        for ln in self._lines:
            ln.set_data([], [])
            ln.set_alpha(0.0)
        for tx in self._texts:
            tx.set_text("")
        
        for i, ci in enumerate(top_idx):
            if ci >= len(omega_bins):
                continue
                
            omega_k = omega_bins[ci]
            gamma_k = gamma[ci]
            amp_k = psi_amp[ci]
            phase_k = psi_phase[ci]
            state_k = mode_state[ci]
            y_offset = i * 2.2  # Slightly tighter packing
            
            # Find coupled particles via the *Lorentzian* lineshape
            if len(exc) > 0:
                d = exc - omega_k
                g2 = max(float(gamma_k) * float(gamma_k), 1e-12)
                tuning = g2 / (d * d + g2)
                coupled_mask = tuning > 0.3
                coupled_indices = np.where(coupled_mask)[0]
                coupled_count = len(coupled_indices)
            else:
                coupled_count = 0
                coupled_indices = np.array([], dtype=np.int64)
                tuning = np.array([])
            
            total_coupled += coupled_count
            
            # Compute phase coherence among coupled particles
            coherence_R = 0.0
            if coupled_count > 0 and particle_phase is not None and len(particle_phase) > 0:
                try:
                    coupled_phases = particle_phase[coupled_indices]
                    # Kuramoto order parameter for this mode's coupled particles
                    order = np.mean(np.exp(1j * coupled_phases))
                    coherence_R = float(np.abs(order))
                    total_coherent += coupled_count * coherence_R
                except (IndexError, TypeError):
                    coherence_R = 0.0
            
            # State-based coloring
            state_colors = {
                0: ('#7f8c8d', 'neutral'),   # gray
                1: ('#3498db', 'stable'),     # blue
                2: ('#f39c12', 'crystal'),    # gold
            }
            base_color, state_label = state_colors.get(int(state_k), ('#7f8c8d', 'neutral'))
            
            # ----------------------------------------------------------------
            # Draw VU meter bars (left section: amplitude indicator)
            # ----------------------------------------------------------------
            amp_norm = amp_k / (psi_amp.max() + 1e-8)
            num_segments = 12
            segment_width = (2 * np.pi) / num_segments
            active_segments = max(1, int(amp_norm * num_segments))
            bar_height = 0.35 + 0.35 * amp_norm
            
            fill_base = i * 12
            for seg in range(active_segments):
                seg_start = seg * segment_width
                seg_end = seg_start + segment_width * 0.85
                seg_mask = (t >= seg_start) & (t < seg_end)
                
                # Gradient from green to yellow to red
                seg_progress = seg / max(num_segments - 1, 1)
                if seg_progress < 0.5:
                    sp = seg_progress / 0.5
                    seg_r, seg_g, seg_b = 0.25 + sp * 0.50, 0.72 - sp * 0.02, 0.35 - sp * 0.05
                elif seg_progress < 0.75:
                    sp = (seg_progress - 0.5) / 0.25
                    seg_r, seg_g, seg_b = 0.75 + sp * 0.13, 0.70 - sp * 0.20, 0.30 - sp * 0.02
                else:
                    sp = (seg_progress - 0.75) / 0.25
                    seg_r, seg_g, seg_b = 0.88, 0.50 - sp * 0.15, 0.28 + sp * 0.04
                
                seg_alpha = 0.5 + 0.4 * amp_norm
                # Update preallocated segment poly
                x_seg = t[seg_mask]
                y_low = np.full_like(x_seg, y_offset - bar_height / 2.0, dtype=np.float64)
                y_high = np.full_like(x_seg, y_offset + bar_height / 2.0, dtype=np.float64)
                _set_fill(self._fills[fill_base + seg], x_seg, y_low, y_high,
                          color=(seg_r, seg_g, seg_b), alpha=seg_alpha)
            
            # ----------------------------------------------------------------
            # Draw wave reconstruction (center section: Ψ_k contribution)
            # ----------------------------------------------------------------
            # Show the actual Ψ_k wave (amplitude and phase)
            wave_k = amp_norm * 0.6 * np.sin(omega_k * t + phase_k)
            # Fade wave based on coherence
            wave_alpha = 0.3 + 0.5 * coherence_R
            ln_main = self._lines[i]  # first 4 lines are main waves
            ln_main.set_data(t_right, wave_k[mask] + y_offset)
            ln_main.set_color(base_color)
            ln_main.set_alpha(wave_alpha)
            
            # Draw coupled particle contributions as thin overlaid waves
            if coupled_count > 0 and len(exc) > 0:
                combined_wave = np.zeros_like(t)
                overlay_base = 4 + i * 6  # after 4 main lines, next 24 are overlays
                for j, oi in enumerate(coupled_indices[:6]):
                    osc_omega = exc[oi]
                    osc_phase = particle_phase[oi] if (particle_phase is not None and oi < len(particle_phase)) else 0.0
                    osc_amp = tuning[oi] * 0.25 * amp_norm
                    wave = osc_amp * np.sin(osc_omega * t + osc_phase)
                    ln = self._lines[overlay_base + j]
                    ln.set_data(t_right, wave[mask] + y_offset)
                    ln.set_alpha(0.25)
                    combined_wave += wave
                
                # Interference pattern (sum of contributions)
                if coupled_count > 1:
                    combined_wave = combined_wave / coupled_count
                    ln_comb = self._lines[4 + 24 + i]  # after main+overlays, 4 combined lines
                    ln_comb.set_data(t_right, combined_wave[mask] + y_offset)
                    ln_comb.set_alpha(0.6)
            
            # ----------------------------------------------------------------
            # Draw status badges (right section)
            # ----------------------------------------------------------------
            x_badge = float(t[-1] + 0.3)
            
            # Mode state badge
            tx_state = self._texts[i * 2]
            tx_state.set_position((x_badge, y_offset + 0.15))
            tx_state.set_text(f'[{state_label}]')
            tx_state.set_color(base_color)
            tx_state.set_fontweight('bold')
            
            # Coherence indicator
            if coherence_R > 0.7:
                coh_color, coh_label = '#27ae60', 'R↑'
            elif coherence_R > 0.4:
                coh_color, coh_label = '#f39c12', 'R~'
            else:
                coh_color, coh_label = '#95a5a6', 'R↓'
            tx_coh = self._texts[i * 2 + 1]
            tx_coh.set_position((x_badge, y_offset - 0.15))
            tx_coh.set_text(f'{coupled_count}p {coh_label}{coherence_R:.2f}')
            tx_coh.set_color(coh_color)
        
        # ----------------------------------------------------------------
        # Combined Ψ wave (bottom overlay - shows global interference)
        # ----------------------------------------------------------------
        combined_psi = np.zeros_like(t)
        for ci in top_idx:
            if ci < len(omega_bins):
                combined_psi += psi_amp[ci] * np.sin(omega_bins[ci] * t + psi_phase[ci])
        
        if psi_amp.max() > 0:
            combined_psi = combined_psi / psi_amp.max() * 0.4
        y_combined = -0.8
        poly_comb = self._fills[-1]
        _set_fill(poly_comb, t, np.full_like(t, y_combined, dtype=np.float64), y_combined + combined_psi,
                  color='#9b59b6', alpha=0.3)
        ln_global = self._lines[-1]
        ln_global.set_data(t, y_combined + combined_psi)
        ln_global.set_color('#9b59b6')
        ln_global.set_alpha(0.8)
        
        # Axis setup
        self.ax.set_xlim(0, 4 * np.pi + 1.8)
        self.ax.set_ylim(-1.5, len(top_idx) * 2.2)
        
        # Summary stats in title
        n_crystal = int(np.sum(mode_state == 2))
        n_stable = int(np.sum(mode_state == 1))
        avg_coherence = (total_coherent / total_coupled) if total_coupled > 0 else 0.0
        self.ax.set_title(
            f'ω-mode VU meter | {n_crystal} crystal, {n_stable} stable | '
            f'{total_coupled} coupled, ⟨R⟩={avg_coherence:.2f}',
            fontsize=9, pad=4
        )
