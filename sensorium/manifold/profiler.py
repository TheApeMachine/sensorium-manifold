from typing import Optional, Any
from pathlib import Path
import time

from .config import SimulationConfig


def create_profiler(config: "SimulationConfig") -> Optional[Any]:
    """Create a torch profiler if profiling is enabled."""
    if not config.profile_enabled:
        return None
    
    # Use torch.profiler for GPU profiling
    from torch.profiler import profile, ProfilerActivity, schedule
    
    activities = [ProfilerActivity.CPU]
    if config.device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    # Note: MPS profiling is limited, but CPU profiling still works
    
    return profile(
        activities=activities,
        schedule=schedule(
            wait=config.profile_warmup_steps,
            warmup=2,
            active=config.num_steps - config.profile_warmup_steps - 2,
            repeat=1,
        ),
        on_trace_ready=lambda p: save_profiler_trace(p, config.profile_output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def save_profiler_trace(profiler, output_dir: Path) -> None:
    """Save profiler trace and print summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Chrome trace for visualization
    trace_path = output_dir / f"trace_{int(time.time())}.json"
    profiler.export_chrome_trace(str(trace_path))
    print(f"\nProfiler trace saved to {trace_path}")
    print("View in Chrome: chrome://tracing")
    
    # Print summary table
    print("\n" + profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
