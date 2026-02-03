from typing import Dict, Any
import time
from pathlib import Path
from dataclasses import dataclass
import torch
from .config import SimulationConfig
from .visualizer import SimulationDashboard
from .carriers import CarrierState
from .profiler import create_profiler
from optimizer.manifold_physics import (
    ManifoldPhysics,
    ManifoldPhysicsConfig,
    SpectralCarrierPhysics,
    SpectralCarrierConfig,
    ParticleGenerator,
)


def run_simulation(config: SimulationConfig) -> Dict[str, Any]:
    """Run an indefinite simulation with random file injections.
    
    Press Ctrl+C to stop.
    """
    import random
    
    print("=" * 60)
    print("THERMO-MANIFOLD CONTINUOUS SIMULATION")
    print("=" * 60)
    print(f"Device:        {config.device}")
    print(f"Grid:          {config.grid_size}")
    print(f"Initial:       {config.num_particles} particles")
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
        poisson_iterations=config.poisson_iterations,
        # Fundamental constants
        G=config.G,
        k_B=config.k_B,
        sigma_SB=config.sigma_SB,
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
    spectral_cfg = SpectralCarrierConfig(
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
        seed_carriers=3,
    )
    carrier_physics = SpectralCarrierPhysics(
        config=spectral_cfg,
        grid_size=config.grid_size,
        dt=config.dt,
        device=str(config.device),
    )
    print("[INFO] Using kernel-accelerated spectral carriers (resonance potential)")
    
    # Initialize empty carriers for dashboard (will be populated on first carrier update)
    carriers = CarrierState.empty(config.device, config.dtype)
    
    # Schedule first injection
    def next_injection_time() -> float:
        """Get next injection time with quantized intervals."""
        min_t = config.inject_interval_min
        max_t = config.inject_interval_max
        step = config.inject_interval_step
        
        # Quantize to step intervals
        num_steps = int((max_t - min_t) / step) + 1
        interval = min_t + random.randint(0, num_steps - 1) * step
        return time.time() + interval
    
    next_inject = next_injection_time()
    
    print("Starting simulation...")
    start_time = time.time()
    step = 0
    
    try:
        while True:
            step_start = time.perf_counter()
            
            # Check if it's time to inject a new file
            current_time = time.time()
            if current_time >= next_inject:
                # Generate and inject new file (Metal-accelerated)
                num_new = random.randint(config.inject_particles_min, config.inject_particles_max)
                file_data = generator.generate_file(num_particles=num_new)
                
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
                
                total_new_energy = file_data["energies"].sum().item()
                
                print(f"[{time.strftime('%H:%M:%S')}] INJECT #{file_data['file_id']}: "
                      f"+{num_new} particles ({file_data['pattern']}) "
                      f"E={total_new_energy:.1f} → Total: {len(state.positions)} particles")
                
                if dashboard:
                    dashboard.record_injection(
                        step=step,
                        file_id=file_data["file_id"],
                        pattern=file_data["pattern"],
                        num_particles=num_new,
                        total_energy=total_new_energy,
                    )
                
                # Schedule next injection
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
                    step_time_ms=step_time_ms,
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
