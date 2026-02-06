"""Profiler instrument for the Sensorium Manifold.

Follows the Profiler protocol so it can be used to instrument the Manifold.
"""

from typing import Any
import time

from torch.profiler import profile, ProfilerActivity, schedule

class ProfilerInstrument:
    def __init__(self) -> None:
        self.profiler = profile(
            activities=[
                ProfilerActivity.CPU, 
                ProfilerActivity.CUDA
            ], 
            record_shapes=True, 
            profile_memory=True, 
            with_stack=True
        )
        
        self.profiler.start()

    def __enter__(self) -> None:
        self.profiler.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.profiler.stop()
        self.profiler.export_chrome_trace(f"trace_{int(time.time())}.json")
        return self.profiler