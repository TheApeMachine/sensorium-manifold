#!/usr/bin/env python3
"""Thermo-Manifold Simulation Entrypoint

Instrumented simulation runner with:
- GPU profiling for performance bottleneck identification
- Real-time dashboard for understanding dynamics
- Clean separation between physics engine and observations

Usage:
    python run.py                    # Run with defaults
    python run.py --profile          # Run with GPU profiling
    python run.py --no-dashboard     # Run headless (for benchmarks)
    python run.py --steps 1000       # Custom step count
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from sensorium.manifold.config import SimulationConfig
from sensorium.manifold.simulator import run_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Thermo-Manifold Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps (ignored in continuous mode)")
    parser.add_argument("--particles", type=int, default=1000, help="Initial number of particles")
    parser.add_argument("--grid", type=int, default=32, help="Grid size (cubic)")
    parser.add_argument("--profile", action="store_true", help="Enable GPU profiling")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable real-time dashboard")
    parser.add_argument("--dashboard-video", type=str, default=None, help="Record dashboard to video file (e.g. artifacts/dashboard.mp4 or .gif)")
    parser.add_argument("--dashboard-fps", type=int, default=30, help="Dashboard video FPS (default: 30)")
    parser.add_argument("--device", type=str, default=None, help="Device (mps, cuda, cpu)")
    
    # Continuous mode options
    parser.add_argument("--continuous", "-c", action="store_true", 
                        help="Run indefinitely with random file injections")
    parser.add_argument("--inject-min", type=float, default=10.0,
                        help="Minimum seconds between file injections (default: 10)")
    parser.add_argument("--inject-max", type=float, default=60.0,
                        help="Maximum seconds between file injections (default: 60)")
    parser.add_argument("--inject-particles-min", type=int, default=30,
                        help="Minimum particles per injection (default: 30)")
    parser.add_argument("--inject-particles-max", type=int, default=100,
                        help="Maximum particles per injection (default: 100)")
    
    args = parser.parse_args()
    
    # Build config
    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    config = SimulationConfig(
        grid_size=(args.grid, args.grid, args.grid),
        num_particles=args.particles,
        num_steps=args.steps,
        device=device,
        profile_enabled=args.profile,
        dashboard_enabled=not args.no_dashboard,
        dashboard_video_path=(None if args.dashboard_video is None else Path(args.dashboard_video)),
        dashboard_video_fps=int(args.dashboard_fps),
        continuous=args.continuous,
        inject_interval_min=args.inject_min,
        inject_interval_max=args.inject_max,
        inject_particles_min=args.inject_particles_min,
        inject_particles_max=args.inject_particles_max,
    )
    
    result = run_simulation(config)
    print("\nFinal results:")
    print(f"  Energy: {result['final_energy']:.4f}")
    print(f"  Heat:   {result['final_heat']:.4f}")
    print(f"  Total:  {result['final_energy'] + result['final_heat']:.4f}")


if __name__ == "__main__":
    main()
