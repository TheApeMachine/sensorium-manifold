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
from .config import SimulationConfig
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
    # Carrier data (carriers have no inherent position - computed for viz)
    carrier_frequencies: Optional[np.ndarray] = None   # (M,)
    carrier_gate_widths: Optional[np.ndarray] = None   # (M,)
    carrier_amplitudes: Optional[np.ndarray] = None    # (M,)


class SimulationDashboard:
    """Real-time dashboard using FuncAnimation for smooth animation."""
    
    def __init__(self, config: SimulationConfig, update_every: int = 1, max_particles: Optional[int] = None):
        self.config = config
        self._update_every = update_every
        # If None, render all particles (no sampling)
        self._max_particles = max_particles
        
        # Animation state
        self._fig = None
        self._anim = None
        self._latest_frame: Optional[FrameData] = None
        self._frame_lock = threading.Lock()

        # Optional video recording (incremental writer; suitable for continuous runs)
        self._record_lock = threading.Lock()
        self._record_writer = None
        self._record_path: Optional[Path] = None
        self._recording: bool = False
        
        # Axes
        self._ax3d = None
        self._ax_combined = None   # Combined metrics (top right left)
        self._ax_info = None       # Info text (top right right)
        self._ax_waves = None      # Wave visualization (bottom right)
        
        # 3D artists
        self._plot_particles = None
        self._plot_halos = None
        self._plot_carriers = None
        self._plot_arrows = []
        self._plot_links = []
        
        # History for 2D plots
        self._history: Dict[str, deque] = {
            "step": deque(maxlen=500),
            "total_energy": deque(maxlen=500),
            "total_heat": deque(maxlen=500),
            "mean_excitation": deque(maxlen=500),
            "max_velocity": deque(maxlen=500),
            "step_time_ms": deque(maxlen=500),
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
        top_h = 0.35
        bottom_h = 0.55
        
        # 3D view (left side, full height)
        self._ax3d = self._fig.add_axes([margin, margin, left_w, 1-2*margin], projection="3d")
        
        # Top right: combined metrics (left) + info text (right)
        self._ax_combined = self._fig.add_axes([right_x, 1 - margin - top_h, right_w * 0.65, top_h])
        self._ax_info = self._fig.add_axes([right_x + right_w * 0.68, 1 - margin - top_h, right_w * 0.30, top_h])
        
        # Bottom right: wave visualization (full width)
        self._ax_waves = self._fig.add_axes([right_x, margin, right_w, bottom_h])
        
        # Setup 3D axis
        ax = self._ax3d
        ax.set_facecolor('white')
        ax.xaxis.pane.set_facecolor('#f0f0f0')
        ax.yaxis.pane.set_facecolor('#f0f0f0')
        ax.zaxis.pane.set_facecolor('#f0f0f0')
        gx, gy, gz = self.config.grid_size
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
            blit=False,   # 3D doesn't support blitting
            # Avoid unbounded caching for an open-ended animation.
            cache_frame_data=False
        )
        
        plt.show(block=False)
    
    def _compute_carrier_positions(
        self,
        carrier_frequencies: np.ndarray,   # (M,) carrier center frequencies
        carrier_gate_widths: np.ndarray,   # (M,) carrier gate widths (σ)
        particle_positions: np.ndarray,    # (N, 3) particle positions
        oscillator_frequencies: np.ndarray,# (N,) oscillator frequencies (excitations)
    ) -> np.ndarray:
        """Compute carrier positions for visualization.
        
        Carriers have no inherent spatial position (they exist in frequency space).
        For visualization, we position each carrier at the weighted centroid of
        the oscillators it couples to, based on frequency alignment:
        
            tuning_ik = exp(-(ω_i - ω_k)² / σ_k²)
            position_k = Σ_i (tuning_ik * pos_i) / Σ_i tuning_ik
        
        - Single coupled oscillator: carrier near that oscillator
        - Multiple coupled: carrier at weighted centroid
        """
        M = len(carrier_frequencies)
        if M == 0:
            return np.empty((0, 3))
        
        N = len(particle_positions)
        if N == 0:
            # No particles, put carriers at center
            gx, gy, gz = self.config.grid_size
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
            return []
        
        pos = frame.positions
        vel = frame.velocities
        heat = frame.heats
        exc = frame.excitations
        energy = frame.energies
        n = len(pos)
        
        ax = self._ax3d
        
        # Update particles
        self._plot_particles.set_xdata(pos[:, 0])
        self._plot_particles.set_ydata(pos[:, 1])
        self._plot_particles.set_3d_properties(pos[:, 2])
        
        # Update halos
        heat_norm = heat / (heat.max() + 1e-8)
        hot_mask = heat_norm > 0.15
        if hot_mask.any():
            hot_pos = pos[hot_mask]
            self._plot_halos.set_xdata(hot_pos[:, 0])
            self._plot_halos.set_ydata(hot_pos[:, 1])
            self._plot_halos.set_3d_properties(hot_pos[:, 2])
            avg_heat = heat_norm[hot_mask].mean()
            self._plot_halos.set_markeredgecolor((1.0, 1.0 - avg_heat, 0.0, 0.5 + 0.4 * avg_heat))
            self._plot_halos.set_markeredgewidth(1 + 3 * avg_heat)
        else:
            self._plot_halos.set_xdata([])
            self._plot_halos.set_ydata([])
            self._plot_halos.set_3d_properties([])
        
        # Update carriers - compute positions from frequency coupling
        if frame.carrier_frequencies is not None and len(frame.carrier_frequencies) > 0:
            gate_widths = frame.carrier_gate_widths if frame.carrier_gate_widths is not None else np.full(len(frame.carrier_frequencies), 0.35)
            cpos = self._compute_carrier_positions(
                frame.carrier_frequencies,
                gate_widths,
                pos,
                exc,
            )
            self._plot_carriers.set_xdata(cpos[:, 0])
            self._plot_carriers.set_ydata(cpos[:, 1])
            self._plot_carriers.set_3d_properties(cpos[:, 2])
            num_carriers = len(cpos)
        else:
            cpos = None
            self._plot_carriers.set_xdata([])
            self._plot_carriers.set_ydata([])
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
        
        # Update links (less frequently)
        if frame.step % 5 == 0:
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
                
                for ci in range(min(len(cpos), 20)):
                    # Compute actual tuning: T = exp(-(ω_osc - ω_carrier)² / σ²)
                    d = exc - cfreq[ci]
                    sigma_sq = gate_widths[ci] ** 2
                    tuning = np.exp(-(d * d) / max(sigma_sq, 1e-8))
                    
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
                        
                        for pi in coupled_idx:
                            t = tuning[pi]
                            s = t * camp_norm[ci]
                            link, = ax.plot(
                                [pos[pi, 0], cpos[ci, 0]],
                                [pos[pi, 1], cpos[ci, 1]],
                                [pos[pi, 2], cpos[ci, 2]],
                                'k--', linewidth=0.3 + s * 1.5, alpha=0.2 + 0.5 * t
                            )
                            self._plot_links.append(link)
        
        ax.set_title(f'Step {frame.step} | {n}p | {num_carriers}c', fontsize=9)
        
        # Update 2D plots
        self._update_2d_plots(frame)

        # If recording, append this frame to the output video.
        # We do this after all artists/axes have been updated.
        with self._record_lock:
            if self._recording and self._record_writer is not None:
                try:
                    self._record_writer.grab_frame()
                except Exception as e:
                    # Disable recording on error to avoid breaking the live dashboard.
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
        
        # Combined metrics chart (energy, heat, conservation)
        ax = self._ax_combined
        ax.clear()
        if len(steps) > 1:
            e_hist = list(self._history["total_energy"])
            h_hist = list(self._history["total_heat"])
            ax.plot(steps, e_hist, 'b-', lw=1.5, label='Energy')
            ax.plot(steps, h_hist, 'r-', lw=1.5, label='Heat')
            ax.plot(steps, [e+h for e,h in zip(e_hist, h_hist)], 'purple', ls='--', lw=1.5, label='Total')
            ax.legend(fontsize=7, loc='upper left')
        ax.set_title('Energy Conservation', fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Step', fontsize=8)
        
        # Info text block
        ax = self._ax_info
        ax.clear()
        ax.axis('off')
        total_e = float(energy.sum())
        total_h = float(heat.sum())
        num_carriers = len(frame.carrier_frequencies) if frame.carrier_frequencies is not None else 0
        avg_ms = np.mean(list(self._history["step_time_ms"])[-50:]) if self._history["step_time_ms"] else 0
        fps = 1000/avg_ms if avg_ms > 0 else 0
        
        ax.text(0.05, 0.95, f"""Step: {frame.step:,}
Particles: {len(frame.positions)}
Carriers: {num_carriers}

Energy: {total_e:.1f}
Heat: {total_h:.1f}
Total: {total_e+total_h:.1f}

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
            
            colors = ['#2ecc71', '#3498db', '#e74c3c']
            
            for i, ci in enumerate(top_idx):
                if ci >= len(cfreq):
                    continue
                    
                carrier_omega = cfreq[ci]
                carrier_gate = cgate[ci]
                carrier_amp = camp[ci]
                y_offset = i * 3.0  # Vertical offset for this carrier
                
                # Find oscillators coupled to this carrier
                d = exc - carrier_omega
                sigma_sq = max(carrier_gate * carrier_gate, 1e-8)
                tuning = np.exp(-(d * d) / sigma_sq)
                coupled_mask = tuning > 0.3
                
                # Draw the BLOCK PULSE (gate open/closed)
                # Pulse width is inversely related to frequency (wavelength)
                # Higher frequency = narrower pulse, lower frequency = wider pulse
                pulse_period = 2 * np.pi  # One full cycle
                pulse_width = pulse_period * 0.5  # 50% duty cycle
                
                # Create block pulse pattern
                pulse_phase = (t % pulse_period)
                pulse_on = pulse_phase < pulse_width
                pulse_height = 0.8
                
                # Draw block pulse as filled rectangles
                ax.fill_between(t, y_offset - pulse_height/2, y_offset + pulse_height/2,
                               where=pulse_on, color=colors[i % len(colors)], alpha=0.3,
                               label=f'C{ci}: ω={carrier_omega:.2f}')
                
                # Draw individual coupled sines (faint, inside the pulse)
                combined_wave = np.zeros_like(t)
                coupled_count = 0
                for oi in np.where(coupled_mask)[0][:8]:  # Limit to 8 per carrier
                    osc_omega = exc[oi]
                    # Scale frequency to show visible oscillation
                    display_freq = max(osc_omega, 0.5) * 3  # Ensure visible waves
                    osc_amp = tuning[oi] * 0.4
                    wave = osc_amp * np.sin(display_freq * t)
                    # Only show wave where pulse is on
                    wave_masked = np.where(pulse_on, wave, 0)
                    ax.plot(t, wave_masked + y_offset, color=colors[i % len(colors)], 
                           alpha=0.2, lw=0.8)
                    combined_wave += wave
                    coupled_count += 1
                
                # Draw combined wave (bold) - the sum of all coupled oscillators
                if coupled_count > 0:
                    combined_wave = combined_wave / max(coupled_count, 1)
                    # Mask to pulse region
                    combined_masked = np.where(pulse_on, combined_wave, 0)
                    ax.plot(t, combined_masked + y_offset, color=colors[i % len(colors)], 
                           lw=2.5, alpha=0.9)
                
                # Add label showing coupled count
                ax.text(t[-1] + 0.2, y_offset, f'{coupled_count} osc', fontsize=7, 
                       va='center', color=colors[i % len(colors)])
            
            ax.legend(fontsize=7, loc='upper left')
            ax.set_xlim(0, 4 * np.pi + 1)
            ax.set_ylim(-1.5, 9)
        
        ax.set_title('Wave Space: Top Carriers (pulse envelope + coupled oscillators)', fontsize=10)
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
        
        n = len(pos)
        
        # Sample if needed (disabled by default; set max_particles to an int to enable)
        if self._max_particles is not None and n > self._max_particles:
            idx = np.random.choice(n, self._max_particles, replace=False)
            pos, vel, energy, heat, exc = pos[idx], vel[idx], energy[idx], heat[idx], exc[idx]
        
        # Carrier data (carriers have no position - computed in viz from frequencies)
        cfreq, cgate, camp = None, None, None
        if carriers is not None and carriers.num_carriers > 0:
            cfreq = carriers.frequencies.cpu().numpy()
            cgate = carriers.gate_widths.cpu().numpy()
            camp = carriers.amplitudes.cpu().numpy()
        
        # Update history
        self._history["step"].append(step)
        self._history["total_energy"].append(float(energy.sum()))
        self._history["total_heat"].append(float(heat.sum()))
        self._history["mean_excitation"].append(float(exc.mean()))
        self._history["max_velocity"].append(float(np.linalg.norm(vel, axis=1).max()))
        self._history["step_time_ms"].append(step_time_ms)
        
        # Create frame data
        frame = FrameData(
            step=step,
            positions=pos,
            velocities=vel,
            energies=energy,
            heats=heat,
            excitations=exc,
            step_time_ms=step_time_ms,
            carrier_frequencies=cfreq,
            carrier_gate_widths=cgate,
            carrier_amplitudes=camp,
        )
        
        # Update latest frame (thread-safe)
        with self._frame_lock:
            self._latest_frame = frame
        
        # Process events to keep animation responsive
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
