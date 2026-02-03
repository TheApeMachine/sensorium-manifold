"""Kernel-based continuous simulation (finite run).

This is the kernel analogue of the interactive simulator, but runs for a fixed
number of steps so it can be used in an experiment pipeline.

Outputs (best-effort):
- `paper/figures/continuous_final.png`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from optimizer.manifold_physics import (
    ManifoldPhysics,
    ManifoldPhysicsConfig,
    SpectralCarrierPhysics,
    SpectralCarrierConfig,
    ParticleGenerator,
)
from sensorium.manifold.visualizer import SimulationDashboard
from sensorium.manifold.carriers import CarrierState


@dataclass(frozen=True, slots=True)
class KernelContinuousConfig:
    device: str = "mps"
    grid_size: tuple[int, int, int] = (32, 32, 32)
    dt: float = 0.02
    steps: int = 1500

    num_particles_init: int = 600
    inject_every: int = 50
    inject_min: int = 40
    inject_max: int = 80

    # Dashboard
    dashboard_enabled: bool = False
    dashboard_update_interval: int = 10
    dashboard_video_path: Optional[Path] = None


def run_kernel_continuous(cfg: KernelContinuousConfig, *, out_dir: Path = Path("./paper")) -> Dict[str, Any]:
    if cfg.device != "mps":
        raise RuntimeError("Kernel continuous experiment currently expects device='mps' (Metal).")

    dev = torch.device(cfg.device)
    dtype = torch.float32
    two_pi = float(2.0 * torch.pi)

    # Physics
    mp_cfg = ManifoldPhysicsConfig(
        grid_size=cfg.grid_size,
        dt=float(cfg.dt),
        poisson_iterations=25,
        device=cfg.device,
    )
    physics = ManifoldPhysics(mp_cfg, device=cfg.device)

    # Generator (Metal kernel)
    generator = ParticleGenerator(grid_size=cfg.grid_size, device=cfg.device)

    # Carriers
    scfg = SpectralCarrierConfig(max_carriers=64)
    carrier_physics = SpectralCarrierPhysics(config=scfg, grid_size=cfg.grid_size, dt=float(cfg.dt), device=cfg.device)
    carriers = CarrierState.empty(cfg.device, dtype)

    # Initial particles
    init = generator.generate_file(num_particles=int(cfg.num_particles_init), pattern="cluster", energy_scale=1.0)
    positions = init["positions"].to(dev, dtype=dtype)
    velocities = init["velocities"].to(dev, dtype=dtype)
    energies = init["energies"].to(dev, dtype=dtype)
    heats = init["heats"].to(dev, dtype=dtype)
    excitations = init["excitations"].to(dev, dtype=dtype)
    masses = init["masses"].to(dev, dtype=dtype)
    osc_phase = torch.rand(int(cfg.num_particles_init), device=dev, dtype=dtype) * two_pi

    # Dashboard (optional)
    dashboard = None
    if cfg.dashboard_enabled:
        # Build a minimal compatible config-like object for the dashboard.
        from sensorium.manifold.config import SimulationConfig

        sim_cfg = SimulationConfig(
            grid_size=cfg.grid_size,
            dt=float(cfg.dt),
            poisson_iterations=25,
            device=cfg.device,
            dashboard_enabled=True,
            dashboard_update_interval=int(cfg.dashboard_update_interval),
        )
        dashboard = SimulationDashboard(sim_cfg)
        if cfg.dashboard_video_path:
            dashboard.start_recording(Path(cfg.dashboard_video_path))

    # Run
    for step in range(int(cfg.steps)):
        if cfg.inject_every > 0 and (step % int(cfg.inject_every) == 0) and step > 0:
            n_new = int(cfg.inject_min + (step * 1103515245 + 12345) % max(1, (cfg.inject_max - cfg.inject_min + 1)))
            pat = ParticleGenerator.PATTERN_NAMES[(step // cfg.inject_every) % len(ParticleGenerator.PATTERN_NAMES)]
            new = generator.generate_file(num_particles=n_new, pattern=pat, energy_scale=1.0)
            positions = torch.cat([positions, new["positions"]])
            velocities = torch.cat([velocities, new["velocities"]])
            energies = torch.cat([energies, new["energies"]])
            heats = torch.cat([heats, new["heats"]])
            excitations = torch.cat([excitations, new["excitations"]])
            masses = torch.cat([masses, new["masses"]])
            osc_phase = torch.cat([osc_phase, torch.rand(n_new, device=dev, dtype=dtype) * two_pi])

            if dashboard:
                dashboard.record_injection(
                    step=step,
                    file_id=int(new["file_id"]),
                    pattern=str(new["pattern"]),
                    num_particles=int(n_new),
                    total_energy=float(new["energies"].sum().detach().to("cpu").item()),
                )

        positions, velocities, energies, heats, excitations = physics.step(
            positions, velocities, energies, heats, excitations, masses
        )

        if step % 10 == 0:
            cst = carrier_physics.step(osc_phase, excitations, energies)
            osc_phase = cst["osc_phase"]
            carriers = CarrierState(
                frequencies=cst["frequencies"],
                gate_widths=cst["gate_widths"],
                amplitudes=cst["amplitudes"],
                phases=cst["phases"],
            )

        if dashboard and (step % int(cfg.dashboard_update_interval) == 0):
            dashboard.update(
                step=step,
                positions=positions,
                velocities=velocities,
                energies=energies,
                heats=heats,
                excitations=excitations,
                step_time_ms=0.0,
                carriers=carriers,
                gravity_field=physics.gravity_potential,
            )

    # Save final snapshot (always)
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if dashboard is None:
        # Create a dashboard just for a final render.
        from sensorium.manifold.config import SimulationConfig

        sim_cfg = SimulationConfig(
            grid_size=cfg.grid_size,
            dt=float(cfg.dt),
            poisson_iterations=25,
            device=cfg.device,
            dashboard_enabled=True,
            dashboard_update_interval=1,
        )
        dashboard = SimulationDashboard(sim_cfg)
        dashboard.update(
            step=int(cfg.steps),
            positions=positions,
            velocities=velocities,
            energies=energies,
            heats=heats,
            excitations=excitations,
            step_time_ms=0.0,
            carriers=carriers,
            gravity_field=physics.gravity_potential,
        )

    dashboard.save(fig_dir / "continuous_final.png")
    try:
        dashboard.close()
    except Exception:
        pass

    return {
        "steps": int(cfg.steps),
        "final_particles": int(positions.shape[0]),
        "final_energy": float(energies.sum().detach().to("cpu").item()),
        "final_heat": float(heats.sum().detach().to("cpu").item()),
    }

