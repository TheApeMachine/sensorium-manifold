"""Real-time dashboard for monitoring the thermo-manifold simulation.

Uses matplotlib FuncAnimation for proper smooth animation.
The simulation pushes state to a queue, and FuncAnimation pulls from it.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from collections import deque
from dataclasses import dataclass
import atexit
import threading
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .carriers import CarrierState


@dataclass
class FrameData:
    """Data for one animation frame."""
    step: int
    positions: np.ndarray          # Particle positions (N, 3)
    velocities: np.ndarray
    energies: np.ndarray
    heats: np.ndarray
    excitations: np.ndarray        # Oscillator frequencies (N,)
    step_time_ms: float
    masses: Optional[np.ndarray] = None
    # Carrier data (carriers have no inherent position - computed for viz)
    carrier_frequencies: Optional[np.ndarray] = None   # (M,)
    carrier_gate_widths: Optional[np.ndarray] = None   # (M,)
    carrier_amplitudes: Optional[np.ndarray] = None    # (M,)
    # Optional field slices for 3D overlays (CPU-side)
    gravity_slice: Optional[np.ndarray] = None         # (gx, gy) slice of gravity potential
    # Optional extra info for integrity/prediction overlays
    extra: Optional[Dict[str, Any]] = None


class SimulationDashboard:
    """Real-time dashboard using FuncAnimation for smooth animation."""
    
    def __init__(
        self,
        config: Any = None,
        *,
        grid_size: tuple[int, int, int] | None = None,
        device: str | None = None,
        dashboard_update_interval: int | None = None,
        update_every: int = 1,
        max_particles: Optional[int] = None,
        video_path: Path | None = None,
    ):
        """Create a dashboard.

        Backward compatible with the simulation runner, which passes a full
        `SimulationConfig`. For generator/byte experiments, you can pass only:
        `grid_size`, `device`, and `dashboard_update_interval`.
        """
        # Support both "full config object" and explicit params.
        if config is not None:
            if grid_size is None:
                grid_size = tuple(getattr(config, "grid_size"))
            if device is None:
                device = str(getattr(config, "device"))
            if dashboard_update_interval is None:
                dashboard_update_interval = int(getattr(config, "dashboard_update_interval", 10))

        if grid_size is None:
            grid_size = (32, 32, 32)
        if device is None:
            device = "mps"

        self.grid_size = tuple(int(x) for x in grid_size)
        self.device = str(device)
        self._update_every = update_every
        # If None, render all particles (no sampling)
        self._max_particles = max_particles
        
        # Animation state
        self._fig = None
        self._anim = None
        self._latest_frame: Optional[FrameData] = None
        self._frame_lock = threading.Lock()
        
        # Caching for expensive computations
        self._cached_carrier_positions: Optional[np.ndarray] = None
        self._cached_carrier_hash: Optional[int] = None  # Hash of carrier state to detect changes

        # Optional video recording (incremental writer; suitable for continuous runs)
        self._record_lock = threading.Lock()
        self._record_writer = None
        self._record_path: Optional[Path] = None
        self._recording: bool = False

        # Diagnostics
        self._last_nonfinite_warn_step: int = -10**18

        # Field overlays (viz-only; does not affect simulation)
        # You can control these by adding attributes to the config object:
        #   - dashboard_field_overlay: bool
        #   - dashboard_field_overlay_mode: "surface" | "wind" | "surface+wind"
        #   - dashboard_field_overlay_stride: int (downsampling for wind)
        self._field_overlay_enabled: bool = bool(getattr(config, "dashboard_field_overlay", True)) if config is not None else True
        self._field_overlay_mode: str = str(getattr(config, "dashboard_field_overlay_mode", "surface")).lower() if config is not None else "surface"
        self._field_overlay_stride: int = int(getattr(config, "dashboard_field_overlay_stride", 4)) if config is not None else 4
        
        # Axes
        self._ax3d = None
        self._ax_energy = None     # Energy metrics (top right left, top)
        self._ax_sigma = None      # Carrier gate width stats (top right left, bottom)
        self._ax_info = None       # Info text (top right right)
        self._ax_waves = None      # Wave visualization (bottom right)
        
        # 3D artists
        self._plot_particles = None
        self._plot_halos = None
        self._plot_carriers = None
        self._plot_arrows = []
        self._plot_links = []
        self._plot_field_surface = None
        self._plot_field_quiver = None
        
        # History for 2D plots
        self._history: Dict[str, deque] = {
            "step": deque(maxlen=500),
            # [CHOICE] energy accounting (dashboard)
            # [FORMULA] E_int = Σ energy_i (internal / oscillator store)
            #          E_heat = Σ heat_i (thermal store)
            #          E_kin  = Σ 0.5 m_i ||v_i||^2  (mechanical kinetic energy)
            # [REASON] prior dashboard omitted E_kin, which made runs look like
            #          "inputs inject heat not energy" when energy was entering as KE.
            # [NOTES] This is a visualization/integrity metric; it does not affect simulation.
            "energy_int_sum": deque(maxlen=500),
            "energy_heat_sum": deque(maxlen=500),
            "energy_kin_sum": deque(maxlen=500),
            "mean_excitation": deque(maxlen=500),
            "max_velocity": deque(maxlen=500),
            "step_time_ms": deque(maxlen=500),
            # Carrier gate width (σ_k) summary (NaN when no carriers)
            "sigma_mean": deque(maxlen=500),
            "sigma_p10": deque(maxlen=500),
            "sigma_p90": deque(maxlen=500),
            "sigma_min": deque(maxlen=500),
            "sigma_max": deque(maxlen=500),
        }
        self._injections: List[Dict[str, Any]] = []
    
    def _init_figure(self) -> None:
        """Initialize matplotlib figure and start animation."""
        plt.ion()
        
        self._fig = plt.figure(figsize=(16, 8))
        
        # Layout: 3D on left, top-right has combined chart + info, bottom-right has wave viz
        margin = 0.03
        left_w = 0.48
        right_x = left_w + 2*margin
        right_w = 1.0 - right_x - margin
        # [CHOICE] right-panel vertical split (metrics vs wave-space)
        # [FORMULA] top_h + bottom_h + margins = 1
        # [REASON] make the wave/pulse panel slightly shorter so top-right titles/labels
        #          never collide (especially after adding σ_k stats).
        top_h = 0.40
        bottom_h = 0.50
        
        # 3D view (left side, full height)
        self._ax3d = self._fig.add_axes([margin, margin, left_w, 1-2*margin], projection="3d")
        
        # Top right: combined metrics (left) + info text (right)
        # Split into two stacked panels: energy (top) + σ_k stats (bottom)
        combined_x = right_x
        combined_y = 1 - margin - top_h
        combined_w = right_w * 0.65
        combined_h = top_h
        # [CHOICE] intra-panel gap (energy vs σ_k)
        # [FORMULA] gap = 0.10 * combined_h
        # [REASON] prevent title overlap even on smaller screens / font scaling.
        gap = 0.10 * combined_h
        h_energy = 0.58 * combined_h
        h_sigma = combined_h - h_energy - gap
        self._ax_energy = self._fig.add_axes([combined_x, combined_y + h_sigma + gap, combined_w, h_energy])
        self._ax_sigma = self._fig.add_axes([combined_x, combined_y, combined_w, h_sigma])
        self._ax_info = self._fig.add_axes([right_x + right_w * 0.68, 1 - margin - top_h, right_w * 0.30, top_h])
        
        # Bottom right: wave visualization (full width)
        self._ax_waves = self._fig.add_axes([right_x, margin, right_w, bottom_h])
        
        # Setup 3D axis
        ax = self._ax3d
        ax.set_facecolor('white')
        ax.xaxis.pane.set_facecolor('#f0f0f0')
        ax.yaxis.pane.set_facecolor('#f0f0f0')
        ax.zaxis.pane.set_facecolor('#f0f0f0')
        gx, gy, gz = self.grid_size
        ax.set_xlim(0, gx); ax.set_ylim(0, gy); ax.set_zlim(0, gz)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.tick_params(labelsize=6)
        
        # Initialize 3D artists
        self._plot_particles, = ax.plot([], [], [], 'o',
            color=(0.5, 0.5, 0.5, 0.6), markersize=4, markeredgecolor='k',
            markeredgewidth=0.3, linestyle='None')
        
        self._plot_halos, = ax.plot([], [], [], 'o',
            color='none', markersize=6, markeredgecolor='orange',
            markeredgewidth=2, linestyle='None', fillstyle='none')
        
        self._plot_carriers, = ax.plot([], [], [], 's',
            color=(0.3, 0.6, 0.8, 0.7), markersize=6, markeredgecolor='k',
            markeredgewidth=0.4, linestyle='None')
        
        # Start animation - interval in ms
        self._anim = FuncAnimation(
            self._fig, 
            self._animate_frame,
            interval=25,  # Update every 50ms for smooth animation
            blit=False,   # 3D doesn't support blitting
            cache_frame_data=False
        )
        
        # Show the figure window (non-blocking) and force initial draw
        plt.show(block=False)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        
    
    def _compute_carrier_positions(
        self,
        carrier_frequencies: np.ndarray,   # (M,) carrier center frequencies
        carrier_gate_widths: np.ndarray,   # (M,) carrier gate widths (σ)
        particle_positions: np.ndarray,    # (N, 3) particle positions (already sampled in stream.py)
        oscillator_frequencies: np.ndarray,# (N,) oscillator frequencies (excitations)
    ) -> np.ndarray:
        """Compute carrier positions for visualization.
        
        NOTE: particle_positions is already sampled to ~5000 particles in stream.py,
        so we don't need additional sampling here. The main optimization is caching
        (see _animate_frame) to avoid recomputing when carriers haven't changed.
        
        Carriers have no inherent spatial position (they exist in frequency space).
        For visualization, we position each carrier at the weighted centroid of
        the oscillators it couples to, based on frequency alignment:
        
            tuning_ik = exp(-(ω_i - ω_k)² / σ_k²)
            position_k = Σ_i (tuning_ik * pos_i) / Σ_i tuning_ik
        """
        M = len(carrier_frequencies)
        if M == 0:
            return np.empty((0, 3))
        
        N = len(particle_positions)
        if N == 0:
            # No particles, put carriers at center
            gx, gy, gz = self.grid_size
            center = np.array([gx / 2.0, gy / 2.0, gz / 2.0])
            return np.tile(center, (M, 1))
        
        # Compute coupling matrix: (M, N)
        # tuning_ik = exp(-(ω_i - ω_k)² / σ_k²)
        omega_diff = carrier_frequencies[:, np.newaxis] - oscillator_frequencies[np.newaxis, :]  # (M, N)
        sigma_sq = (carrier_gate_widths ** 2)[:, np.newaxis]  # (M, 1)
        sigma_sq = np.maximum(sigma_sq, 1e-8)
        tuning = np.exp(-(omega_diff ** 2) / sigma_sq)  # (M, N)
        
        # Weighted centroid: (M, N) @ (N, 3) = (M, 3)
        weight_sum = tuning.sum(axis=1, keepdims=True)  # (M, 1)
        weight_sum = np.maximum(weight_sum, 1e-8)
        weighted_pos = tuning @ particle_positions  # (M, 3)
        centroid = weighted_pos / weight_sum  # (M, 3)
        
        # For single-coupled carriers, add small offset so visible near but not on particle
        significant_couplings = (tuning > 0.1).sum(axis=1)  # (M,)
        single_coupled = (significant_couplings == 1)
        if single_coupled.any():
            idx = np.arange(M)
            offset_dir = np.stack([
                np.sin(idx * 1.0),
                np.sin(idx * 1.7),
                np.sin(idx * 2.3)
            ], axis=1)  # (M, 3)
            centroid[single_coupled] = centroid[single_coupled] + offset_dir[single_coupled] * 0.5
        
        return centroid
    
    def _animate_frame(self, frame_num: int):
        """Animation function called by FuncAnimation."""
        with self._frame_lock:
            frame = self._latest_frame
        
        if frame is None:
            # Return artists even when no frame data yet (allows initial draw)
            if self._plot_particles is not None:
                return [self._plot_particles, self._plot_halos, self._plot_carriers]
            return []
        
        pos = frame.positions
        vel = frame.velocities
        heat = frame.heats
        exc = frame.excitations
        energy = frame.energies
        n = len(pos)

        # Filter non-finite positions (Matplotlib 3D will silently render nothing for NaNs).
        if n > 0:
            finite = np.isfinite(pos).all(axis=1)
            if not bool(finite.all()):
                bad = int((~finite).sum())
                if frame.step - self._last_nonfinite_warn_step >= 50:
                    print(f"[dashboard] non-finite positions detected at step={frame.step} (dropping {bad}/{n} for viz)")
                    self._last_nonfinite_warn_step = int(frame.step)
                pos = pos[finite]
                vel = vel[finite]
                heat = heat[finite]
                exc = exc[finite]
                energy = energy[finite]
                n = len(pos)
        
        # Safety check: if no particles, clear plots and return
        if n == 0:
            if self._plot_particles is not None:
                # For 3D Line artists, set_data(x, y) is more reliable than set_xdata/set_ydata.
                self._plot_particles.set_data([], [])
                self._plot_particles.set_3d_properties([])
                self._plot_halos.set_data([], [])
                self._plot_halos.set_3d_properties([])
                self._plot_carriers.set_data([], [])
                self._plot_carriers.set_3d_properties([])
            return [self._plot_particles, self._plot_halos, self._plot_carriers] if self._plot_particles is not None else []
        
        ax = self._ax3d
        
        # Update particles
        self._plot_particles.set_data(pos[:, 0], pos[:, 1])
        self._plot_particles.set_3d_properties(pos[:, 2])
        
        # Update halos
        heat_norm = heat / (heat.max() + 1e-8)
        hot_mask = heat_norm > 0.15
        if hot_mask.any():
            hot_pos = pos[hot_mask]
            self._plot_halos.set_data(hot_pos[:, 0], hot_pos[:, 1])
            self._plot_halos.set_3d_properties(hot_pos[:, 2])
            avg_heat = heat_norm[hot_mask].mean()
            self._plot_halos.set_markeredgecolor((1.0, 1.0 - avg_heat, 0.0, 0.5 + 0.4 * avg_heat))
            self._plot_halos.set_markeredgewidth(1 + 3 * avg_heat)
        else:
            self._plot_halos.set_data([], [])
            self._plot_halos.set_3d_properties([])
        
        # Update carriers - compute positions from frequency coupling (with caching)
        if frame.carrier_frequencies is not None and len(frame.carrier_frequencies) > 0:
            # Create hash of carrier state to detect changes
            carrier_hash = hash((
                tuple(frame.carrier_frequencies[:10]),  # First 10 frequencies as signature
                len(frame.carrier_frequencies),
                tuple(frame.carrier_gate_widths[:10]) if frame.carrier_gate_widths is not None else None,
            ))
            
            # Only recompute if carriers changed or cache is invalid
            # NOTE: pos/exc are already sampled to ~5000 particles in stream.py
            if carrier_hash != self._cached_carrier_hash or self._cached_carrier_positions is None:
                gate_widths = frame.carrier_gate_widths if frame.carrier_gate_widths is not None else np.full(len(frame.carrier_frequencies), 0.35)
                self._cached_carrier_positions = self._compute_carrier_positions(
                    frame.carrier_frequencies,
                    gate_widths,
                    pos,  # Already sampled in stream.py
                    exc,  # Already sampled in stream.py
                )
                self._cached_carrier_hash = carrier_hash
            
            cpos = self._cached_carrier_positions
            self._plot_carriers.set_data(cpos[:, 0], cpos[:, 1])
            self._plot_carriers.set_3d_properties(cpos[:, 2])
            num_carriers = len(cpos)
        else:
            cpos = None
            self._cached_carrier_positions = None
            self._cached_carrier_hash = None
            self._plot_carriers.set_data([], [])
            self._plot_carriers.set_3d_properties([])
            num_carriers = 0
        
        # Update arrows
        for arrow in self._plot_arrows:
            try:
                arrow.remove()
            except:
                pass
        self._plot_arrows = []
        
        vel_mags = np.linalg.norm(vel, axis=1)
        vel_max = vel_mags.max() + 1e-8
        if vel_max > 0.01:
            skip = max(1, n // 30)
            for i in range(0, n, skip):
                if vel_mags[i] > 0.01:
                    t = vel_mags[i] / vel_max
                    arrow, = ax.plot(
                        [pos[i, 0], pos[i, 0] + vel[i, 0] * 3],
                        [pos[i, 1], pos[i, 1] + vel[i, 1] * 3],
                        [pos[i, 2], pos[i, 2] + vel[i, 2] * 3],
                        color=(t, 0.2, 1-t, 0.7), linewidth=0.8
                    )
                    self._plot_arrows.append(arrow)
        
        for link in self._plot_links:
            try:
                link.remove()
            except:
                pass
        self._plot_links = []
        
        if cpos is not None and len(cpos) > 0:
            cfreq = frame.carrier_frequencies
            gate_widths = frame.carrier_gate_widths if frame.carrier_gate_widths is not None else np.full(len(cfreq), 0.35)
            camp = np.abs(frame.carrier_amplitudes) if frame.carrier_amplitudes is not None else np.ones(len(cfreq))
            camp_norm = camp / (camp.max() + 1e-8)
            
            # NOTE: pos/exc are already sampled to ~5000 particles in stream.py, so we can use them directly
            link_sample_idx = np.arange(n)
            link_pos = pos
            link_exc = exc
            
            for ci in range(min(len(cpos), 20)):
                # Compute tuning only for sampled particles
                d = link_exc - cfreq[ci]
                sigma_sq = max(gate_widths[ci] ** 2, 1e-8)
                tuning = np.exp(-(d * d) / sigma_sq)
                
                # Show top coupled oscillators (tuning > 0.3 means well-aligned)
                well_coupled = tuning > 0.3
                if well_coupled.any():
                    # Get indices of well-coupled oscillators, sorted by tuning
                    coupled_idx = np.where(well_coupled)[0]
                    # Limit to top 5 per carrier to avoid clutter
                    if len(coupled_idx) > 5:
                        top_tuning = tuning[coupled_idx]
                        top_order = np.argsort(top_tuning)[-5:]
                        coupled_idx = coupled_idx[top_order]
                    
                    for pi_idx in coupled_idx:
                        pi = link_sample_idx[pi_idx]  # Map back to original index
                        t = tuning[pi_idx]
                        s = t * camp_norm[ci]
                        link, = ax.plot(
                            [pos[pi, 0], cpos[ci, 0]],
                            [pos[pi, 1], cpos[ci, 1]],
                            [pos[pi, 2], cpos[ci, 2]],
                            # [CHOICE] bond line styling (3D)
                            # [FORMULA] medium gray + low alpha, scaled by tuning
                            # [REASON] keep structure visible without overpowering particles
                            # [NOTES] dashed to distinguish from velocity arrows.
                            linestyle="--",
                            color=(0.45, 0.45, 0.45, 1.0),
                            linewidth=0.25 + s * 1.2,
                            alpha=0.08 + 0.22 * t,
                        )
                        self._plot_links.append(link)

        # ---------------------------------------------------------------------
        # Optional field overlay: gravity potential slice (surface + "wind")
        # ---------------------------------------------------------------------
        if self._field_overlay_enabled and frame.gravity_slice is not None:
            gslice = frame.gravity_slice
            if gslice.size > 0:
                # Remove old artists to avoid accumulation.
                if self._plot_field_surface is not None:
                    try:
                        self._plot_field_surface.remove()
                    except Exception:
                        pass
                    self._plot_field_surface = None
                if self._plot_field_quiver is not None:
                    try:
                        self._plot_field_quiver.remove()
                    except Exception:
                        pass
                    self._plot_field_quiver = None

                gx, gy, gz = self.grid_size

                # [CHOICE] overlay plane + height scaling
                # [FORMULA] z = z0 + s * normalize(field_slice)
                # [REASON] show "peaks/valleys" intuition without obscuring particles
                # [NOTES] scale is derived from grid size (viz-only; not a simulation knob).
                z0 = 0.08 * float(gz)
                height_scale = 0.18 * float(min(gx, gy, gz))

                g = np.asarray(gslice, dtype=np.float64)
                g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                gmin = float(g.min())
                gmax = float(g.max())
                denom = (gmax - gmin) if (gmax > gmin) else 1.0
                gn = (g - gmin) / denom  # [0,1]

                X, Y = np.meshgrid(np.arange(gx), np.arange(gy), indexing="ij")
                Z = z0 + height_scale * gn

                # Surface mesh (more opaque for better visibility)
                cmap = plt.get_cmap("viridis")
                facecolors = cmap(gn)
                facecolors[..., 3] = 0.55  # alpha - increased from 0.16 for less transparency
                self._plot_field_surface = ax.plot_surface(
                    X,
                    Y,
                    Z,
                    rstride=1,
                    cstride=1,
                    facecolors=facecolors,
                    linewidth=0.0,
                    antialiased=False,
                    shade=False,
                    zorder=0,
                )

                # Optional "wind map": show -∇φ on the same plane (downsampled)
                if self._field_overlay_mode in ("wind", "surface+wind", "both"):
                    stride = max(1, int(self._field_overlay_stride))
                    dfdx, dfdy = np.gradient(g.astype(np.float64))
                    U = (-dfdx)[::stride, ::stride]
                    V = (-dfdy)[::stride, ::stride]
                    XX = X[::stride, ::stride]
                    YY = Y[::stride, ::stride]
                    ZZ = Z[::stride, ::stride]

                    # Normalize arrows for readability (viz-only)
                    mag = np.sqrt(U * U + V * V) + 1e-12
                    U = U / mag
                    V = V / mag
                    self._plot_field_quiver = ax.quiver(
                        XX,
                        YY,
                        ZZ,
                        U,
                        V,
                        0.0,
                        length=1.2,
                        normalize=False,
                        color=(0.2, 0.2, 0.2, 0.2),
                        linewidth=0.6,
                    )
        
        ax.set_title(f'Step {frame.step} | {n}p | {num_carriers}c', fontsize=9)
        
        # Update 2D plots
        self._update_2d_plots(frame)

        # Grab frame for recording
        with self._record_lock:
            if self._recording and self._record_writer is not None:
                try:
                    self._record_writer.grab_frame()
                except Exception as e:
                    print(f"[dashboard] recording failed ({self._record_path}): {e}")
                    try:
                        self._record_writer.finish()
                    except Exception:
                        pass
                    self._record_writer = None
                    self._recording = False
        
        return [self._plot_particles, self._plot_halos, self._plot_carriers]
    
    def _update_2d_plots(self, frame: FrameData) -> None:
        """Update the combined metrics and wave visualization."""
        energy = frame.energies
        heat = frame.heats
        exc = frame.excitations
        steps = list(self._history["step"])
        
        # Energy chart (energy, heat, conservation)
        ax = self._ax_energy
        ax.clear()
        if len(steps) > 1:
            eint = list(self._history["energy_int_sum"])
            eheat = list(self._history["energy_heat_sum"])
            ekin = list(self._history["energy_kin_sum"])
            ax.plot(steps, eint, color="#2980b9", lw=1.5, label="E_int")
            ax.plot(steps, eheat, color="#c0392b", lw=1.5, label="E_heat")
            ax.plot(steps, ekin, color="#7f8c8d", lw=1.5, label="E_kin")
            ax.plot(
                steps,
                [a + b + c for a, b, c in zip(eint, eheat, ekin)],
                color="#8e44ad",
                ls="--",
                lw=1.5,
                label="Total",
            )
            ax.legend(fontsize=7, loc='upper left')
        ax.set_title('Energy Conservation', fontsize=10, pad=6)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Step', fontsize=8)

        # σ_k (gate width) stats chart
        ax = self._ax_sigma
        ax.clear()
        if len(steps) > 1:
            s_mean = np.array(list(self._history["sigma_mean"]), dtype=np.float64)
            s_p10 = np.array(list(self._history["sigma_p10"]), dtype=np.float64)
            s_p90 = np.array(list(self._history["sigma_p90"]), dtype=np.float64)
            s_min = np.array(list(self._history["sigma_min"]), dtype=np.float64)
            s_max = np.array(list(self._history["sigma_max"]), dtype=np.float64)

            valid = np.isfinite(s_mean)
            if valid.any():
                st = np.array(steps, dtype=np.int64)
                ax.plot(st[valid], s_mean[valid], color="#16a085", lw=1.6, label="σ mean")
                ax.fill_between(
                    st[valid],
                    s_p10[valid],
                    s_p90[valid],
                    color="#16a085",
                    alpha=0.20,
                    label="σ p10–p90",
                )
                ax.plot(st[valid], s_min[valid], color="#16a085", lw=0.8, alpha=0.5, ls=":")
                ax.plot(st[valid], s_max[valid], color="#16a085", lw=0.8, alpha=0.5, ls=":")
                ax.legend(fontsize=7, loc="upper left")
        ax.set_title("Carrier gate width (σ_k): open/sharpen", fontsize=10, pad=6)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Step", fontsize=8)
        
        # Info text block
        ax = self._ax_info
        ax.clear()
        ax.axis('off')
        total_eint = float(energy.sum())
        total_h = float(heat.sum())
        # Kinetic energy from per-particle state (see update() for mass fallback logic).
        m_eff = frame.energies if frame.masses is None else frame.masses
        v2 = (frame.velocities ** 2).sum(axis=1)
        total_ekin = float(0.5 * np.sum(m_eff * v2))
        num_carriers = len(frame.carrier_frequencies) if frame.carrier_frequencies is not None else 0
        avg_ms = np.mean(list(self._history["step_time_ms"])[-50:]) if self._history["step_time_ms"] else 0
        fps = 1000/avg_ms if avg_ms > 0 else 0

        # Current σ summary (if available)
        sigma_line = ""
        if frame.carrier_gate_widths is not None and len(frame.carrier_gate_widths) > 0:
            s = frame.carrier_gate_widths
            sigma_line = (
                f"σ mean: {float(np.mean(s)):.3f}\n"
                f"σ p10/p90: {float(np.percentile(s, 10)):.3f} / {float(np.percentile(s, 90)):.3f}\n"
            )

        # Integrity check: scripted prediction vs observed response (if provided)
        pred_line = ""
        if frame.extra is not None:
            pred = frame.extra.get("prediction") if isinstance(frame.extra, dict) else None
            if isinstance(pred, dict):
                omega = pred.get("omega")
                k = pred.get("k")
                label = pred.get("label", "")
                if omega is not None and k is not None:
                    try:
                        omega_f = float(omega)
                        k_i = int(k)
                        k_i = max(1, min(k_i, int(len(frame.energies))))
                        # Observed: mean excitation among top-K energetic particles.
                        top_idx = np.argpartition(frame.energies, -k_i)[-k_i:]
                        mean_exc_hot = float(np.mean(frame.excitations[top_idx]))
                        delta = abs(mean_exc_hot - omega_f)

                        # Observed: best carrier frequency match (if carriers available)
                        carrier_delta = None
                        if frame.carrier_frequencies is not None and len(frame.carrier_frequencies) > 0:
                            carrier_delta = float(np.min(np.abs(frame.carrier_frequencies - omega_f)))

                        pred_line = (
                            f"token: {label} ω*={omega_f:.3f}\n"
                            f"topK⟨ω⟩={mean_exc_hot:.3f} |Δ|={delta:.3f}\n"
                            + (f"min|ω_k-ω*|={carrier_delta:.3f}\n" if carrier_delta is not None else "")
                        )
                    except Exception:
                        pred_line = ""
        
        ax.text(0.05, 0.95, f"""Step: {frame.step:,}
Particles: {len(frame.positions)}
Carriers: {num_carriers}

E_int: {total_eint:.1f}
E_heat: {total_h:.1f}
E_kin: {total_ekin:.1f}
Total: {total_eint+total_h+total_ekin:.1f}

{pred_line}\
{sigma_line}\
{frame.step_time_ms:.1f} ms/step
{fps:.0f} steps/sec""", transform=ax.transAxes, fontsize=9, va='top', family='monospace')
        
        # Wave visualization - show top carriers with their coupled oscillators
        ax = self._ax_waves
        ax.clear()
        
        if frame.carrier_frequencies is not None and len(frame.carrier_frequencies) > 0:
            cfreq = frame.carrier_frequencies
            cgate = frame.carrier_gate_widths if frame.carrier_gate_widths is not None else np.full(len(cfreq), 0.35)
            camp = np.abs(frame.carrier_amplitudes) if frame.carrier_amplitudes is not None else np.ones(len(cfreq))
            
            # Sort carriers by amplitude, take top 3
            top_idx = np.argsort(camp)[-3:][::-1]
            
            # Time axis for wave display (show ~2 pulse cycles)
            t = np.linspace(0, 4 * np.pi, 400)
            
            # Color palette - soft, muted tones
            # Coupled oscillator sines: soft cyan/teal for contrast
            coupled_sine_color = '#5fbdbd'  # Soft teal
            # Main combined wave: dark charcoal gray
            main_wave_color = '#3a3a3a'
            
            for i, ci in enumerate(top_idx):
                if ci >= len(cfreq):
                    continue
                    
                carrier_omega = cfreq[ci]
                carrier_gate = cgate[ci]
                carrier_amp = camp[ci]
                y_offset = i * 3.0  # Vertical offset for this carrier
                
                # NOTE: exc is already sampled to ~5000 particles in stream.py
                wave_exc = exc
                
                # Find oscillators coupled to this carrier (from sampled set)
                d = wave_exc - carrier_omega
                sigma_sq = max(carrier_gate * carrier_gate, 1e-8)
                tuning = np.exp(-(d * d) / sigma_sq)
                coupled_mask = tuning > 0.3
                
                # Calculate conflict level for VU meter coloring
                # Conflict is based on: spread of coupled oscillators (high spread = conflict)
                # and misalignment from carrier center frequency
                coupled_indices = np.where(coupled_mask)[0]
                coupled_count = len(coupled_indices)
                
                if coupled_count > 0:
                    coupled_freqs = wave_exc[coupled_indices]
                    # Conflict metric: variance of coupled oscillator frequencies + mean detuning
                    freq_variance = np.var(coupled_freqs) if coupled_count > 1 else 0.0
                    mean_detuning = np.mean(np.abs(coupled_freqs - carrier_omega))
                    # Normalize conflict to [0, 1] - higher = more conflict
                    conflict = np.clip(freq_variance / 0.5 + mean_detuning / 0.3, 0.0, 1.0) / 2.0
                else:
                    conflict = 0.0
                
                # Soft VU meter gradient: muted sage green -> warm amber -> soft coral red
                # Using HSL-inspired interpolation for smoother, more natural transitions
                if conflict < 0.5:
                    # Sage green to warm amber
                    t_c = conflict / 0.5
                    # Sage: (0.45, 0.65, 0.45) -> Amber: (0.85, 0.68, 0.35)
                    r = 0.45 + t_c * 0.40
                    g = 0.65 + t_c * 0.03
                    b = 0.45 - t_c * 0.10
                else:
                    # Warm amber to soft coral
                    t_c = (conflict - 0.5) / 0.5
                    # Amber: (0.85, 0.68, 0.35) -> Coral: (0.85, 0.42, 0.38)
                    r = 0.85
                    g = 0.68 - t_c * 0.26
                    b = 0.35 + t_c * 0.03
                
                vu_color = (r, g, b)
                
                # Draw VU meter blocks - height and alpha scaled by amplitude
                pulse_period = 1 * np.pi
                pulse_width = pulse_period * 0.5
                pulse_phase = (t % pulse_period)
                pulse_on = pulse_phase < pulse_width
                
                amp_norm = carrier_amp / (camp.max() + 1e-8)
                bar_height = 0.5 + 0.5 * amp_norm
                
                # VU meter discrete segments - each block gets its own gradient color
                num_segments = 10
                segment_width = (4 * np.pi) / num_segments
                active_segments = max(1, int(amp_norm * num_segments))
                
                for seg in range(active_segments):
                    seg_start = seg * segment_width
                    seg_end = seg_start + segment_width * 0.85  # 85% fill, 15% gap
                    seg_mask = (t >= seg_start) & (t < seg_end)
                    
                    # Gradient based on segment position (like real VU meter LEDs)
                    seg_progress = seg / max(num_segments - 1, 1)
                    
                    # Soft gradient: sage green -> warm amber -> soft coral
                    if seg_progress < 0.6:
                        # More green range (first 6 segments)
                        sp = seg_progress / 0.6
                        seg_r = 0.45 + sp * 0.35
                        seg_g = 0.70 - sp * 0.05
                        seg_b = 0.45 - sp * 0.08
                    elif seg_progress < 0.85:
                        # Amber/yellow range (segments 7-8)
                        sp = (seg_progress - 0.6) / 0.25
                        seg_r = 0.80 + sp * 0.08
                        seg_g = 0.65 - sp * 0.12
                        seg_b = 0.37 - sp * 0.02
                    else:
                        # Coral/red range (segments 9-10)
                        sp = (seg_progress - 0.85) / 0.15
                        seg_r = 0.88 - sp * 0.03
                        seg_g = 0.53 - sp * 0.13
                        seg_b = 0.35 + sp * 0.05
                    
                    # Alpha scales with segment position for gradient effect
                    seg_alpha = 0.55 + 0.35 * (seg / num_segments)
                    
                    seg_height = bar_height * 0.7
                    ax.fill_between(t, y_offset - seg_height/2, y_offset + seg_height/2,
                                   where=seg_mask & pulse_on, 
                                   color=(seg_r, seg_g, seg_b), alpha=seg_alpha)
                
                # Draw individual coupled sines - soft teal, faint
                combined_wave = np.zeros_like(t)
                coupled_indices_limited = coupled_indices[:8]
                for oi_sampled in coupled_indices_limited:
                    osc_omega = wave_exc[oi_sampled]
                    display_freq = max(osc_omega, 0.5) * 3
                    osc_amp = tuning[oi_sampled] * 0.4
                    wave = osc_amp * np.sin(display_freq * t)
                    wave_masked = np.where(pulse_on, wave, 0)
                    ax.plot(t, wave_masked + y_offset, color=coupled_sine_color, 
                           alpha=0.35, lw=0.7)
                    combined_wave += wave
                
                # Draw combined wave - dark gray, bold
                if coupled_count > 0:
                    combined_wave = combined_wave / max(coupled_count, 1)
                    combined_masked = np.where(pulse_on, combined_wave, 0)
                    ax.plot(t, combined_masked + y_offset, color=main_wave_color, 
                           lw=2.0, alpha=0.85)
                
                # Label with conflict status
                conflict_label = "OK" if conflict < 0.33 else ("WARN" if conflict < 0.66 else "HIGH")
                label_color = '#5a8a5a' if conflict < 0.33 else ('#b08840' if conflict < 0.66 else '#a85555')
                ax.text(t[-1] + 0.2, y_offset, f'{coupled_count} osc [{conflict_label}]', fontsize=7, 
                       va='center', color=label_color)
                
                # Legend entry
                ax.plot([], [], 's', color=vu_color, markersize=8,
                       label=f'C{ci}: ω={carrier_omega:.2f}, σ={carrier_gate:.3f}')
            
            ax.legend(fontsize=7, loc='upper left')
            ax.set_xlim(0, 4 * np.pi + 1)
            ax.set_ylim(-1.5, 9)
        
        ax.set_title('VU Meter: Carrier Conflict (green=aligned, amber=moderate, coral=misaligned)', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Phase', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
    
    def update(
        self,
        step: int,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        energies: torch.Tensor,
        heats: torch.Tensor,
        excitations: torch.Tensor,
        step_time_ms: float,
        masses: Optional[torch.Tensor] = None,
        extra: Optional[Dict[str, Any]] = None,
        gravity_field: Optional[torch.Tensor] = None,
        carriers: Optional[CarrierState] = None,
    ) -> None:
        """Push new frame data to the animation."""
        if self._fig is None:
            self._init_figure()
        
        # Convert to numpy
        pos = positions.detach().cpu().numpy()
        vel = velocities.detach().cpu().numpy()
        energy = energies.detach().cpu().numpy()
        heat = heats.detach().cpu().numpy()
        exc = excitations.detach().cpu().numpy()
        mass = None if masses is None else masses.detach().cpu().numpy()
        
        n = len(pos)
        
        # Sample if needed (disabled by default; set max_particles to an int to enable)
        if self._max_particles is not None and n > self._max_particles:
            idx = np.random.choice(n, self._max_particles, replace=False)
            pos, vel, energy, heat, exc = pos[idx], vel[idx], energy[idx], heat[idx], exc[idx]
            if mass is not None:
                mass = mass[idx]
        
        # Carrier data (carriers have no position - computed in viz from frequencies)
        cfreq, cgate, camp = None, None, None
        if carriers is not None and carriers.num_carriers > 0:
            cfreq = carriers.frequencies.cpu().numpy()
            cgate = carriers.gate_widths.cpu().numpy()
            camp = carriers.amplitudes.cpu().numpy()

        # Optional gravity slice for 3D overlay (viz-only).
        gslice = None
        if gravity_field is not None:
            try:
                g = gravity_field.detach().to("cpu").to(torch.float32).numpy()
                if g.ndim == 3:
                    # [CHOICE] mid-plane slice
                    # [FORMULA] slice = field[:, :, z_mid]
                    # [REASON] cheap, stable visualization that updates each frame
                    # [NOTES] can be extended to multiple slices or iso-surfaces later.
                    z_mid = int(g.shape[2] // 2)
                    gslice = g[:, :, z_mid]
            except Exception:
                gslice = None
        
        # Update history
        self._history["step"].append(step)
        self._history["energy_int_sum"].append(float(energy.sum()))
        self._history["energy_heat_sum"].append(float(heat.sum()))
        # [CHOICE] kinetic energy from particle state (viz/integrity)
        # [FORMULA] E_kin = Σ 0.5 m_i ||v_i||^2
        # [REASON] dashboards must include KE or runs look like "heat injection"
        # [NOTES] if masses are not provided, we fall back to m_i := energy_i (legacy sim invariant).
        m_eff = energy if mass is None else mass
        v2 = (vel ** 2).sum(axis=1)
        self._history["energy_kin_sum"].append(float(0.5 * np.sum(m_eff * v2)))
        self._history["mean_excitation"].append(float(exc.mean()))
        self._history["max_velocity"].append(float(np.linalg.norm(vel, axis=1).max()))
        self._history["step_time_ms"].append(step_time_ms)

        # σ_k stats (NaN when missing)
        if cgate is not None and len(cgate) > 0:
            s = np.asarray(cgate, dtype=np.float64)
            self._history["sigma_mean"].append(float(np.mean(s)))
            self._history["sigma_p10"].append(float(np.percentile(s, 10)))
            self._history["sigma_p90"].append(float(np.percentile(s, 90)))
            self._history["sigma_min"].append(float(np.min(s)))
            self._history["sigma_max"].append(float(np.max(s)))
        else:
            self._history["sigma_mean"].append(float("nan"))
            self._history["sigma_p10"].append(float("nan"))
            self._history["sigma_p90"].append(float("nan"))
            self._history["sigma_min"].append(float("nan"))
            self._history["sigma_max"].append(float("nan"))
        
        # Create frame data
        frame = FrameData(
            step=step,
            positions=pos,
            velocities=vel,
            energies=energy,
            heats=heat,
            excitations=exc,
            masses=mass,
            step_time_ms=step_time_ms,
            carrier_frequencies=cfreq,
            carrier_gate_widths=cgate,
            carrier_amplitudes=camp,
            gravity_slice=gslice,
            extra=extra,
        )
        
        # Update latest frame (thread-safe)
        with self._frame_lock:
            self._latest_frame = frame
        
        # Process events to let FuncAnimation render
        self._fig.canvas.flush_events()
    
    def record_injection(self, step: int, file_id: int, pattern: str, num_particles: int, total_energy: float) -> None:
        """Record injection event."""
        self._injections.append({
            "step": step, "file_id": file_id, "pattern": pattern,
            "num_particles": num_particles, "energy": total_energy, "time": time.time()
        })
        if len(self._injections) > 20:
            self._injections = self._injections[-20:]
    
    def save(self, path: Path) -> None:
        """Save figure."""
        if self._fig is not None:
            # Stop animation before saving
            if self._anim is not None:
                es = getattr(self._anim, "event_source", None)
                if es is not None:
                    try:
                        es.stop()
                    except Exception:
                        # It's fine if the event source is already shut down.
                        pass
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Dashboard saved to {path}")

    def start_recording(
        self,
        path: Path,
        *,
        fps: int = 30,
        dpi: int = 150,
        codec: str = "libx264",
        bitrate: Optional[int] = None,
    ) -> None:
        """Start recording the animated dashboard to a video file.

        This is incremental (grabs frames as the dashboard runs), so it works for
        `--continuous` runs. The file is finalized when `stop_recording()` or
        `close()` is called (also registered via `atexit`).
        """
        if self._fig is None:
            self._init_figure()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._record_lock:
            # Close any existing writer
            if self._record_writer is not None:
                try:
                    self._record_writer.finish()
                except Exception:
                    pass
                self._record_writer = None
                self._recording = False

            suffix = path.suffix.lower()
            if suffix == ".gif":
                # GIF fallback (no ffmpeg required)
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=fps)
            else:
                # MP4/MOV/etc via ffmpeg
                from matplotlib.animation import FFMpegWriter
                writer_kwargs: Dict[str, Any] = {"fps": fps, "codec": codec}
                if bitrate is not None:
                    writer_kwargs["bitrate"] = int(bitrate)
                writer = FFMpegWriter(**writer_kwargs)

            # Initialize writer and mark active
            writer.setup(self._fig, str(path), dpi=dpi)
            self._record_writer = writer
            self._record_path = path
            self._recording = True

        atexit.register(self.stop_recording)
        print(f"[dashboard] recording to {path} ({fps} fps)")

    def stop_recording(self) -> None:
        """Stop recording and finalize the video file (if active)."""
        with self._record_lock:
            if self._record_writer is None:
                self._recording = False
                return
            try:
                self._record_writer.finish()
            finally:
                self._record_writer = None
                self._recording = False
                self._record_path = None
                print("[dashboard] recording finalized")
    
    def close(self) -> None:
        """Close dashboard."""
        self.stop_recording()
        if self._anim is not None:
            es = getattr(self._anim, "event_source", None)
            if es is not None:
                try:
                    es.stop()
                except Exception:
                    pass
            self._anim = None
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
