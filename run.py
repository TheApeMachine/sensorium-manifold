#!/usr/bin/env python3
"""Thermo-Manifold Simulation."""

import matplotlib.pyplot as plt

from sensorium.manifold import Manifold
from sensorium.tokenizer.universal import UniversalTokenizer, UniversalTokenizerConfig
from sensorium.dataset.synthetic import SyntheticDataset, SyntheticConfig, SyntheticPattern
from sensorium.instrument.dashboard.canvas import Canvas
from sensorium.instrument.dashboard.animation import Animation

tokenizer = UniversalTokenizer(config=UniversalTokenizerConfig(
    datasets=[SyntheticDataset(config=SyntheticConfig(
        pattern=SyntheticPattern.COLLISION,
        num_units=100,
        unit_length=64,
    ))]
))

manifold = Manifold(
    tokenizer=tokenizer,
)

canvas = Canvas(grid_size=(64, 64, 64), datafn=manifold.step)

# Add canvas to manifold instrumentation after both are created
manifold.instrumentation.append(canvas)

manifold.load()

# Create animation using the Animation wrapper
anim = Animation(fig=canvas.fig, animate_frame=canvas.animate_frame)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    anim.stop()
    # Explicitly delete to avoid __del__ issues during shutdown
    del anim

