from typing import Dict, Any, Optional, List
import time
from pathlib import Path
from dataclasses import dataclass
import torch
import json
import random
import numpy as np
from .config import SimulationConfig
from .visualizer import SimulationDashboard
from .carriers import CarrierState
from .profiler import create_profiler
from optimizer.metal.manifold_physics import (
    ManifoldPhysics,
    ManifoldPhysicsConfig,
    SpectralCarrierPhysics,
    SpectralCarrierConfig,
    ParticleGenerator,
)


def run_simulation(config: SimulationConfig) -> Dict[str, Any]:
    """Run an indefinite simulation with injections.
    
    Press Ctrl+C to stop.
    """

    # ---------------------------------------------------------------------
    # Reproducibility for integrity checks
    # ---------------------------------------------------------------------
    seed = int(getattr(config, "seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------------------------------------------------------------------
    # Optional scripted injections (step-based, deterministic)
    # ---------------------------------------------------------------------
    @dataclass(frozen=True)
    class InjectionEvent:
        step: int
        num_particles: int
        pattern: str = "random"
        energy_scale: float = 1.0
        omega: Optional[float] = None  # if set, overrides excitation for injected particles
        label: str = ""
        seed: Optional[int] = None     # if set, reseeds before this injection (determinism)

    def _load_injection_script(path: Path) -> List[InjectionEvent]:
        raw = json.loads(path.read_text())
        if not isinstance(raw, list):
            raise ValueError("Injection script must be a JSON list")
        events: List[InjectionEvent] = []
        for i, e in enumerate(raw):
            if not isinstance(e, dict):
                raise ValueError(f"Injection event {i} must be an object")
            events.append(
                InjectionEvent(
                    step=int(e["step"]),
                    num_particles=int(e["num_particles"]),
                    pattern=str(e.get("pattern", "random")),
                    energy_scale=float(e.get("energy_scale", 1.0)),
                    omega=(None if e.get("omega", None) is None else float(e["omega"])),
                    label=str(e.get("label", "")),
                    seed=(None if e.get("seed", None) is None else int(e["seed"])),
                )
            )
        events.sort(key=lambda ev: ev.step)
        return events

    script_path = getattr(config, "injection_script_path", None)
    script_events: List[InjectionEvent] = []
    script_loop = bool(getattr(config, "injection_script_loop", False))
    if script_path is not None:
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"injection_script_path not found: {script_path}")
        script_events = _load_injection_script(script_path)
        if len(script_events) == 0:
            raise ValueError(f"injection_script_path is empty: {script_path}")
    
    print("=" * 60)
    print("THERMO-MANIFOLD SIMULATION")
    print("=" * 60)
    print(f"Device:        {config.device}")
    print(f"Grid:          {config.grid_size}")
    print(f"Initial:       {config.num_particles} particles")
    if script_events:
        print(f"Inject mode:   SCRIPT ({len(script_events)} events){' loop' if script_loop else ''}")
        print(f"Script:        {str(script_path)}")
    else:
        print(f"Inject mode:   RANDOM")
        print(f"Inject every:  {config.inject_interval_min}-{config.inject_interval_max}s")
        print(f"Inject size:   {config.inject_particles_min}-{config.inject_particles_max} particles")
    print(f"Dashboard:     {'ON' if config.dashboard_enabled else 'OFF'}")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    
    # Initialize physics engine
    physics_config = ManifoldPhysicsConfig(
        grid_size=config.grid_size,
        grid_spacing=config.grid_spacing,
        dt=config.dt,
        unit_system=config.unit_system,
        # Material properties
        particle_radius=config.particle_radius,
        thermal_conductivity=config.thermal_conductivity,
        specific_heat=config.specific_heat,
        dynamic_viscosity=config.dynamic_viscosity,
        emissivity=config.emissivity,
        restitution=config.restitution,
        young_modulus=config.young_modulus,
    )
    physics = ManifoldPhysics(physics_config, device=str(config.device))
    print("[INFO] Using kernel-accelerated physics")
    
    # Initialize particle state
    state = ParticleState.random(config.num_particles, config)
    
    # Initialize Metal-accelerated data generator
    generator = ParticleGenerator(
        grid_size=config.grid_size,
        device=str(config.device),
    )
    print("[INFO] Using kernel-accelerated particle generation")
    
    # Initialize dashboard
    dashboard = SimulationDashboard(config) if config.dashboard_enabled else None
    if dashboard is not None and getattr(config, "dashboard_video_path", None):
        dashboard.start_recording(
            Path(config.dashboard_video_path),
            fps=int(getattr(config, "dashboard_video_fps", 30)),
            dpi=int(getattr(config, "dashboard_video_dpi", 150)),
        )
    
    # Initialize Metal-accelerated spectral carrier physics (entanglement layer)
    # Keep spectral config minimal here; the dynamics should self-regulate.
    spectral_cfg = SpectralCarrierConfig(max_carriers=64)
    carrier_physics = SpectralCarrierPhysics(
        config=spectral_cfg,
        grid_size=config.grid_size,
        dt=config.dt,
        device=str(config.device),
    )
    print("[INFO] Using kernel-accelerated spectral carriers (resonance potential)")
    
    # Initialize empty carriers for dashboard (will be populated on first carrier update)
    carriers = CarrierState.empty(config.device, config.dtype)

    # Random injection scheduling (time-based)
    def next_injection_time() -> float:
        min_t = config.inject_interval_min
        max_t = config.inject_interval_max
        step_s = config.inject_interval_step
        num_steps = int((max_t - min_t) / step_s) + 1
        interval = min_t + random.randint(0, num_steps - 1) * step_s
        return time.time() + interval

    next_inject = next_injection_time()
    script_idx = 0
    script_epoch_start_step = 0
    script_period = (script_events[-1].step + 1) if script_events else 0
    last_prediction: Optional[Dict[str, Any]] = None
    
    print("Starting simulation...")
    start_time = time.time()
    step = 0
    
    try:
        while True:
            step_start = time.perf_counter()
            
            # Check if it's time to inject a new file
            current_time = time.time()
            do_inject = False
            inject_event: Optional[InjectionEvent] = None

            if script_events:
                # Step-based scripted injection
                # If looping, events repeat every `script_period` steps.
                target_step = step - script_epoch_start_step
                while script_idx < len(script_events) and target_step >= script_events[script_idx].step:
                    if target_step == script_events[script_idx].step:
                        do_inject = True
                        inject_event = script_events[script_idx]
                    script_idx += 1

                if script_loop and script_idx >= len(script_events):
                    # Start next epoch
                    script_idx = 0
                    script_epoch_start_step += script_period
            else:
                # Time-based random injections
                do_inject = current_time >= next_inject

            if do_inject:
                # Generate and inject new file (Metal-accelerated)
                if inject_event is None:
                    num_new = random.randint(config.inject_particles_min, config.inject_particles_max)
                    pattern = None
                    energy_scale = 1.0
                    omega = None
                    label = ""
                    ev_seed = None
                else:
                    num_new = int(inject_event.num_particles)
                    pattern = str(inject_event.pattern)
                    energy_scale = float(inject_event.energy_scale)
                    omega = inject_event.omega
                    label = str(inject_event.label)
                    ev_seed = inject_event.seed

                if ev_seed is not None:
                    # Per-event deterministic reseed (integrity scripts)
                    random.seed(int(ev_seed))
                    np.random.seed(int(ev_seed))
                    torch.manual_seed(int(ev_seed))

                file_data = generator.generate_file(num_particles=num_new, pattern=pattern, energy_scale=float(energy_scale))
                
                # Concatenate new particles with existing (tensors already on MPS)
                state.positions = torch.cat([state.positions, file_data["positions"]])
                state.velocities = torch.cat([state.velocities, file_data["velocities"]])
                state.energies = torch.cat([state.energies, file_data["energies"]])
                state.heats = torch.cat([state.heats, file_data["heats"]])
                state.excitations = torch.cat([state.excitations, file_data["excitations"]])
                state.masses = torch.cat([state.masses, file_data["masses"]])
                # Oscillator phase for new particles (wave space)
                two_pi = float(2.0 * torch.pi)
                new_phase = torch.rand(num_new, device=config.device, dtype=config.dtype) * two_pi
                state.osc_phase = torch.cat([state.osc_phase, new_phase])

                # Token control: optionally override excitation (ω) for the injected particles.
                if omega is not None:
                    state.excitations[-num_new:] = float(omega)
                
                total_new_energy = file_data["energies"].sum().item()
                
                print(f"[{time.strftime('%H:%M:%S')}] INJECT #{file_data['file_id']}: "
                      f"+{num_new} particles ({file_data['pattern']}) "
                      f"E={total_new_energy:.1f} → Total: {len(state.positions)} particles")

                # Prediction payload for dashboard integrity check.
                # Prediction is intentionally simple and audit-friendly:
                # we expect the injected ω* to dominate top-K energetic particles.
                if omega is not None:
                    last_prediction = {
                        "omega": float(omega),
                        "k": int(num_new),
                        "label": (label if label else file_data.get("pattern", "")),
                        "step_injected": int(step),
                    }
                
                if dashboard:
                    dashboard.record_injection(
                        step=step,
                        file_id=file_data["file_id"],
                        pattern=file_data["pattern"],
                        num_particles=num_new,
                        total_energy=total_new_energy,
                    )
                
                # Schedule next injection
                if not script_events:
                    next_inject = next_injection_time()
                    secs_until = next_inject - time.time()
                    print(f"     Next injection in {secs_until:.0f}s")
            
            # Physics step
            state.positions, state.velocities, state.energies, state.heats, state.excitations = \
                physics.step(
                    state.positions,
                    state.velocities,
                    state.energies,
                    state.heats,
                    state.excitations,
                    state.masses,
                )
            
            # Update carriers (entanglement layer - every 10 steps, Metal-accelerated)
            if step % 10 == 0:
                carrier_state = carrier_physics.step(
                    state.osc_phase,
                    state.excitations,
                    state.energies,
                )
                # Oscillator phases are updated in-place; keep reference explicit.
                state.osc_phase = carrier_state["osc_phase"]
                # Convert to CarrierState for dashboard compatibility
                carriers = CarrierState(
                    frequencies=carrier_state["frequencies"],
                    gate_widths=carrier_state["gate_widths"],
                    amplitudes=carrier_state["amplitudes"],
                    phases=carrier_state["phases"],
                )
            
            step_time_ms = (time.perf_counter() - step_start) * 1000
            
            # Update dashboard
            if dashboard and step % config.dashboard_update_interval == 0:
                dashboard.update(
                    step=step,
                    positions=state.positions,
                    velocities=state.velocities,
                    energies=state.energies,
                    heats=state.heats,
                    excitations=state.excitations,
                    masses=state.masses,
                    step_time_ms=step_time_ms,
                    extra={"prediction": last_prediction} if last_prediction is not None else None,
                    carriers=carriers,
                    gravity_field=physics.gravity_potential,
                )
            
            # Progress indicator (every 500 steps)
            if step % 500 == 0:
                elapsed = time.time() - start_time
                total_energy = state.energies.sum().item()
                total_heat = state.heats.sum().item()
                print(f"Step {step:6d} | {elapsed/60:.1f}m | {len(state.positions):5d}p | "
                      f"E={total_energy:.1f} H={total_heat:.1f} | {step_time_ms:.1f}ms")
            
            step += 1
    
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")
    
    finally:
        total_time = time.time() - start_time
        if dashboard is not None:
            # Ensure we finalize any active video writer.
            try:
                dashboard.stop_recording()
            except Exception:
                pass
        
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Total time:     {total_time/60:.1f} minutes")
        print(f"Total steps:    {step:,}")
        print(f"Files injected: {generator.file_count}")
        print(f"Final particles: {len(state.positions):,}")
        print(f"Final energy:   {state.energies.sum().item():.2f}")
        print(f"Final heat:     {state.heats.sum().item():.2f}")
        
        if dashboard:
            dashboard.save(Path("artifacts") / "continuous_final.png")
            dashboard.close()
    
    return {
        "steps": step,
        "total_time_s": total_time,
        "files_injected": generator.file_count,
        "final_particles": len(state.positions),
        "final_energy": float(state.energies.sum().item()),
        "final_heat": float(state.heats.sum().item()),
    }


@dataclass
class ParticleState:
    """Holds all particle state tensors."""
    positions: torch.Tensor   # (N, 3)
    velocities: torch.Tensor  # (N, 3)
    energies: torch.Tensor    # (N,)
    heats: torch.Tensor       # (N,)
    excitations: torch.Tensor # (N,)
    masses: torch.Tensor      # (N,)
    osc_phase: torch.Tensor   # (N,) oscillator phase (wave space)
    
    @classmethod
    def random(cls, n: int, config: SimulationConfig) -> "ParticleState":
        """Create random initial state."""
        device = config.device
        dtype = config.dtype
        grid = config.grid_size
        
        # Positions: random in grid space
        positions = torch.rand(n, 3, device=device, dtype=dtype) * torch.tensor(
            [grid[0], grid[1], grid[2]], device=device, dtype=dtype
        ) * 0.5 + torch.tensor([grid[0]/4, grid[1]/4, grid[2]/4], device=device, dtype=dtype)
        
        # Small random velocities
        velocities = torch.randn(n, 3, device=device, dtype=dtype) * 0.1
        
        # Energy: random, uniform
        energies = torch.rand(n, device=device, dtype=dtype) * 0.5 + 0.5
        
        # Heat: starts at zero
        heats = torch.zeros(n, device=device, dtype=dtype)
        
        # Excitation (= oscillator frequency): diverse initial values
        # Spread across [0, 2] to ensure spectral diversity for carrier coupling
        excitations = torch.rand(n, device=device, dtype=dtype) * 2.0
        
        # Masses: proportional to energy
        masses = energies.clone()

        # Oscillator phase: uniform [0, 2π)
        two_pi = float(2.0 * torch.pi)
        osc_phase = torch.rand(n, device=device, dtype=dtype) * two_pi
        
        return cls(
            positions=positions,
            velocities=velocities,
            energies=energies,
            heats=heats,
            excitations=excitations,
            masses=masses,
            osc_phase=osc_phase,
        )
