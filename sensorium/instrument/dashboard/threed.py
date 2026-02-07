from __future__ import annotations

from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ThreeD:
    """3D renderer for particle simulation state.

    Pure visualization - reads simulation state, renders it, computes nothing.
    If data isn't in the state, it doesn't get rendered.

    Visualization modes (auto-selected based on available data):
    - Temperature mode: particles colored by T = Q/(m·c_v), size by |v|
    - Phase mode: particles colored by oscillator phase θ, size by amplitude
    - Coupling mode: particles colored by mode coupling strength

    The surface shows ω-field coupling/support when available, falling back
    to gravity potential.
    """

    __slots__ = (
        "grid_size",
        "ax",
        "_particles",
        "_field_surface",
        "_field_z0",
        "_field_height_scale",
        "_field_X",
        "_field_Y",
        "_field_cmap",
        "_particle_cmap",
        "_phase_cmap",
        "_color_mode",
        "_last_color_mode",
    )

    def __init__(self, grid_size: tuple[int, int, int], ax: Any) -> None:
        self.grid_size = gx, gy, gz = tuple(int(x) for x in grid_size)
        self.ax: Any = ax

        # Axis styling -- dark theme
        ax.set_facecolor("#0e0e1a")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("#141424")
            pane.set_alpha(0.6)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color("#aaa")
            axis._axinfo["tick"]["color"] = "#666"
            axis._axinfo["grid"]["color"] = "#333"
        ax.set(
            xlim=(0, gx), ylim=(0, gy), zlim=(0, gz), xlabel="X", ylabel="Y", zlabel="Z"
        )
        ax.tick_params(labelsize=5, colors="#aaa", pad=0)

        # Particle artist (Path3DCollection) so we can color by heat/coupling.
        self._particles = ax.scatter(
            [],
            [],
            [],
            s=18,
            c=[],
            cmap="inferno",
            alpha=0.80,
            edgecolors="#333",
            linewidths=0.2,
            depthshade=True,
        )

        # Field surface (lazy init)
        self._field_surface: Optional[Poly3DCollection] = None
        self._field_z0 = 0.08 * gz
        self._field_height_scale = 0.18 * min(gx, gy, gz)
        self._field_X, self._field_Y = np.meshgrid(
            np.arange(gx), np.arange(gy), indexing="ij"
        )
        self._field_cmap = plt.get_cmap("viridis")
        self._particle_cmap = plt.get_cmap("inferno")
        self._phase_cmap = plt.get_cmap("twilight")  # Cyclic colormap for phase

        self._color_mode = "temperature"  # auto-selected each frame
        self._last_color_mode = "temperature"

    def update(self, state: dict) -> list[object]:
        """Render simulation state (visualization-only light computations).

        Auto-selects the most informative color mode based on available data:
        - Phase mode: when particle_phase is available (shows oscillator dynamics)
        - Temperature mode: when heats/c_v available (shows thermodynamics)
        - Fallback: neutral coloring
        """
        ax: Any = self.ax
        if ax is None:
            return []

        # Get positions - the only required field
        positions = state.get("positions")
        if positions is None:
            self._particles.set_data_3d([], [], [])
            return [self._particles]

        # Convert to numpy if needed
        pos = (
            positions.detach().cpu().numpy()
            if hasattr(positions, "detach")
            else np.asarray(positions)
        )
        n = int(len(pos))

        if n == 0:
            # Clear scatter
            self._particles._offsets3d = ([], [], [])
            self._particles.set_array(np.array([], dtype=np.float32))
        else:
            # The simulation evolves positions in *physical* coordinates with a domain length
            # L_d = grid_d * dx, where dx = 1/max(grid_dims). The dashboard axes are in
            # grid-index coordinates [0, grid_d). If we plot physical coords directly,
            # particles appear bunched in a corner. Detect and rescale when needed.
            gx, gy, gz = self.grid_size
            dx = 1.0 / float(max(gx, gy, gz))
            domain = np.array([gx * dx, gy * dx, gz * dx], dtype=np.float64)
            # Heuristic: if positions live within the physical domain, scale to index space.
            pos_max = float(np.nanmax(pos)) if pos.size else 0.0
            dom_max = float(np.max(domain))
            if dom_max > 0.0 and pos_max <= (1.01 * dom_max):
                pos_phys = pos
                pos_plot = pos / dx
            else:
                pos_phys = pos * dx
                pos_plot = pos

            # ----------------------------------------------------------------
            # Determine best color mode based on available data
            # Priority: phase (ω-wave dynamics) > temperature > fallback
            # ----------------------------------------------------------------
            particle_phase = state.get("phase")
            particle_energy = state.get("energy_osc")
            heat = state.get("heats", None)
            masses = state.get("masses", None)
            c_v = state.get("c_v", None)

            # Convert tensors to numpy
            def to_np(x):
                if x is None:
                    return None
                if hasattr(x, "detach"):
                    return x.detach().cpu().numpy()
                return np.asarray(x)

            particle_phase = to_np(particle_phase)
            particle_energy = to_np(particle_energy)
            heat = to_np(heat)
            masses = to_np(masses)

            if c_v is not None and hasattr(c_v, "item"):
                c_v = float(c_v.item())
            try:
                c_vf = float(c_v) if c_v is not None else None
            except Exception:
                c_vf = None

            col = None
            color_label = "neutral"
            color_uniform = False
            use_cyclic_cmap = False

            # Try phase mode first (OmegaWave dynamics)
            if particle_phase is not None and particle_phase.size == n:
                # Phase is cyclic [-π, π], normalize to [0, 1] for colormap
                phase = particle_phase.astype(np.float64)
                phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
                # Normalize phase to [0, 1] (cyclic)
                col = (phase + np.pi) / (2 * np.pi)
                col = np.clip(col, 0.0, 1.0).astype(np.float32)
                color_label = "phase θ"
                use_cyclic_cmap = True
                self._color_mode = "phase"

            # Fall back to temperature mode
            elif (
                heat is not None
                and heat.size == n
                and masses is not None
                and masses.size == n
                and (c_vf is not None and c_vf > 0.0)
            ):
                denom = masses.astype(np.float64) * float(c_vf)
                with np.errstate(divide="ignore", invalid="ignore"):
                    T = np.where(denom > 0, heat.astype(np.float64) / denom, 0.0)
                col = T
                color_label = "T(Q/(mc_v))"
                self._color_mode = "temperature"

            elif heat is not None and heat.size == n:
                col = heat.astype(np.float64)
                color_label = "heat Q"
                self._color_mode = "heat"

            if col is None:
                col = np.zeros((n,), dtype=np.float64)
                color_label = "neutral"
                self._color_mode = "neutral"

            # Normalize for non-cyclic colormaps
            if not use_cyclic_cmap:
                col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
                col_range = float(np.max(col) - np.min(col)) if col.size else 0.0
                if col_range <= 1e-12:
                    col = np.full((n,), 0.55, dtype=np.float32)
                    color_uniform = True
                else:
                    p10 = float(np.percentile(col, 10)) if col.size else 0.0
                    p90 = float(np.percentile(col, 90)) if col.size else 1.0
                    if not (p90 > p10):
                        p10, p90 = float(np.min(col)), float(np.max(col))
                    den = (p90 - p10) if (p90 > p10) else 1.0
                    col = (col - p10) / den
                    col = np.clip(col, 0.0, 1.0).astype(np.float32)

            # ----------------------------------------------------------------
            # Marker size: prefer oscillator amplitude (√E_osc), fall back to speed
            # ----------------------------------------------------------------
            size_label = "|v|"
            if particle_energy is not None and particle_energy.size == n:
                # Size by oscillator amplitude A = √E
                amp = np.sqrt(np.maximum(particle_energy.astype(np.float64), 0.0))
                amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
                if np.max(amp) > 0:
                    amp_norm = amp / np.max(amp)
                    sizes = 12.0 + 50.0 * amp_norm
                    self._particles.set_sizes(sizes.astype(np.float32))
                    size_label = "√E_osc"
                else:
                    self._particles.set_sizes(np.full((n,), 18.0, dtype=np.float32))
            else:
                vel_s = state.get("velocities", None)
                if vel_s is not None:
                    vel_s = (
                        vel_s.detach().cpu().numpy()
                        if hasattr(vel_s, "detach")
                        else np.asarray(vel_s)
                    )
                    if vel_s.ndim == 2 and vel_s.shape[0] == n and vel_s.shape[1] == 3:
                        sp = np.sqrt(np.sum(vel_s.astype(np.float64) ** 2, axis=1))
                        sp = np.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)
                        s10 = float(np.percentile(sp, 10)) if sp.size else 0.0
                        s90 = float(np.percentile(sp, 90)) if sp.size else 1.0
                        if not (s90 > s10):
                            s10, s90 = float(np.min(sp)), float(np.max(sp))
                        den = (s90 - s10) if (s90 > s10) else 1.0
                        spn = np.clip((sp - s10) / den, 0.0, 1.0).astype(np.float32)
                        self._particles.set_sizes(10.0 + 40.0 * spn)
                    else:
                        self._particles.set_sizes(np.full((n,), 18.0, dtype=np.float32))
                else:
                    self._particles.set_sizes(np.full((n,), 18.0, dtype=np.float32))

            self._particles._offsets3d = (
                pos_plot[:, 0],
                pos_plot[:, 1],
                pos_plot[:, 2],
            )
            self._particles.set_array(col)

            # Select colormap based on mode
            if use_cyclic_cmap:
                self._particles.set_cmap(self._phase_cmap)
            else:
                self._particles.set_cmap(self._particle_cmap)

        # Field surface: prefer ω-field coupling/support map (behavior), else show gravity.
        field2d, surface_label = self._coupling_or_support_field_xy(state)
        if field2d is None:
            gravity = state.get("gravity_potential")
            if gravity is not None:
                g = (
                    gravity.detach().cpu().numpy()
                    if hasattr(gravity, "detach")
                    else np.asarray(gravity)
                )
                if g.ndim == 3:
                    g = g[:, :, g.shape[2] // 2]
                if g.size > 0:
                    field2d = g
                    surface_label = "gravity φ(x,y)"
        if field2d is not None:
            self._render_field(field2d)

        step = state.get("step", 0)
        if hasattr(step, "item"):
            step = step.item()

        # Build title with current visualization mode
        color_label = ""
        color_uniform = False
        size_label = ""
        extra = []
        label = color_label + (" (uniform)" if color_uniform else "")
        if label:
            extra.append(f"color: {label}")
        if size_label:
            extra.append(f"size: {size_label}")
        if "surface_label" in locals() and surface_label:
            extra.append(f"surface: {surface_label}")
        ax.set_title(
            f"Step {step} | {n}p\n" + " | ".join(extra), fontsize=7, color="#ddd", pad=2
        )

        return [self._particles]

    def _render_field(self, g: np.ndarray) -> None:
        """Render a scalar field as a translucent surface."""
        if self._field_surface is not None:
            try:
                self._field_surface.remove()
            except Exception:
                pass

        g = np.nan_to_num(g.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        # Robust normalize (percentiles) so “flat-ish” fields still show structure.
        lo = float(np.percentile(g, 5))
        hi = float(np.percentile(g, 95))
        if not (hi > lo):
            lo = float(g.min())
            hi = float(g.max())
        den = (hi - lo) if (hi > lo) else 1.0
        gn = (g - lo) / den
        gn = np.clip(gn, 0.0, 1.0)
        Z = self._field_z0 + self._field_height_scale * gn

        facecolors = self._field_cmap(gn)
        facecolors[..., 3] = 0.55

        self._field_surface = self.ax.plot_surface(
            self._field_X,
            self._field_Y,
            Z,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            zorder=0,
        )

    def _coupling_or_support_field_xy(
        self, state: dict
    ) -> tuple[Optional[np.ndarray], str]:
        """Compute a qualitative ω-field map on the mid-z plane.

        Prefer “coupling” when the ω-field has non-trivial |Ψ| energy; otherwise show
        “support” from anchors so the surface still reflects *actual* simulation state
        without inventing heat/energy.

        This is intentionally a visualization proxy (not a solver input).
        """
        pos = state.get("positions")
        amp = state.get("psi_amplitude")
        aidx = state.get("mode_anchor_idx")
        aw = state.get("mode_anchor_weight")
        sigma_x = state.get("spatial_sigma", None)
        if pos is None or amp is None or aidx is None or aw is None or sigma_x is None:
            return None, ""

        # numpy conversion
        pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        amp = amp.detach().cpu().numpy() if hasattr(amp, "detach") else np.asarray(amp)
        aidx = (
            aidx.detach().cpu().numpy() if hasattr(aidx, "detach") else np.asarray(aidx)
        )
        aw = aw.detach().cpu().numpy() if hasattr(aw, "detach") else np.asarray(aw)
        try:
            sigma = float(sigma_x)
        except Exception:
            return None, ""
        if not (sigma > 0.0) or pos.size == 0 or amp.size == 0:
            return None, ""

        gx, gy, gz = self.grid_size
        dx = 1.0 / float(max(gx, gy, gz))
        domain = np.array([gx * dx, gy * dx, gz * dx], dtype=np.float64)

        # Determine whether positions are physical or index coordinates.
        pos_max = float(np.nanmax(pos)) if pos.size else 0.0
        dom_max = float(np.max(domain))
        pos_phys = (
            pos if (dom_max > 0.0 and pos_max <= (1.01 * dom_max)) else (pos * dx)
        )

        slots = int(len(aidx) // max(int(amp.shape[0]), 1))
        if slots <= 0:
            return None, ""

        # Decide whether we have a real coupling field yet.
        amp_abs = np.abs(amp.astype(np.float64))
        amp_energy = float(np.sum(amp_abs))
        use_coupling = amp_energy > 1e-6

        # choose top modes by |Ψ| if available, otherwise by anchor support
        K = int(min(6, amp.shape[0]))
        if use_coupling:
            top = np.argsort(amp_abs)[-K:][::-1]
            label = "ω-field coupling Σ|Ψ_k|·overlap"
        else:
            # support per mode = Σ_a |w_ka|
            w_abs = np.abs(aw.astype(np.float64))
            support = np.zeros((int(amp.shape[0]),), dtype=np.float64)
            for k in range(int(amp.shape[0])):
                base = k * slots
                support[k] = float(np.sum(w_abs[base : base + slots]))
            top = np.argsort(support)[-K:][::-1]
            label = "ω-mode support Σ_a|w_ka|·overlap"

        # grid points on mid-z plane in physical coordinates
        z0 = 0.5 * float(gz) * dx
        X = (self._field_X.astype(np.float64) * dx)[..., None]
        Y = (self._field_Y.astype(np.float64) * dx)[..., None]
        Z = np.full((gx, gy, 1), z0, dtype=np.float64)
        P = np.concatenate([X, Y, Z], axis=2)  # (gx,gy,3)

        inv_4s2 = 1.0 / (4.0 * sigma * sigma)
        C = np.zeros((gx, gy), dtype=np.float64)

        for k in top:
            ak = float(amp_abs[k]) if use_coupling else 1.0
            base = int(k) * slots
            for a in range(slots):
                idx = int(aidx[base + a])
                if idx < 0 or idx >= pos_phys.shape[0]:
                    continue
                w = float(np.abs(aw[base + a]))
                if not (w > 0.0):
                    continue
                pa = pos_phys[idx].astype(np.float64)
                d = P - pa[None, None, :]
                # minimum-image on torus
                d = d - domain[None, None, :] * np.rint(d / domain[None, None, :])
                r2 = (
                    d[..., 0] * d[..., 0]
                    + d[..., 1] * d[..., 1]
                    + d[..., 2] * d[..., 2]
                )
                C += (ak * w) * np.exp(-r2 * inv_4s2)

        return C, label
