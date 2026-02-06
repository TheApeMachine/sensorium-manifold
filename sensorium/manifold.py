"""Manifold is a wrapper around the physics simulation.

This is the main entry point for the Sensorium Manifold.
It is a wrapper around the physics simulation that provides a simple interface for
tokenizing data and running the physics simulation.
"""

from __future__ import annotations

import os
from typing import Sequence
from sensorium.console import console
from sensorium.tokenizer.prototype import Tokenizer
from sensorium.kernels.runtime import get_device
from sensorium.instrument.protocol import InstrumentProtocol
from sensorium.tokenizer.loader import Loader


class Manifold:
    """A wrapper around the physics simulation."""

    def __init__(
        self, 
        tokenizer: Tokenizer,
        grid_size: tuple[int, int, int] = (64, 64, 64),
        instrumentation: Sequence[InstrumentProtocol] = (),
        max_steps: int = 2000,
    ) -> None:
        # Keep a direct reference: some callers use `manifold.tokenizer`.
        self.tokenizer = tokenizer
        self.grid_size = grid_size
        self.instrumentation = instrumentation or []
        self.max_steps = int(max_steps)
        self.state = None
        self._step = 0
        self.device_name = get_device()
        self.thermo = None
        self.wave = None
        self.reset(grid_size=grid_size)

    def reset(self, *, grid_size: tuple[int, int, int] = (64, 64, 64)) -> None:
        """Reset the manifold to its initial state."""
        self.state = None
        self._step = 0
        console.header("Sensorium Manifold", Device=self.device_name, Grid=f"{self.grid_size}")
        with console.spinner(f"Initializing physics on {self.device_name}"):
            match self.device_name:
                case "mps":                
                    from sensorium.kernels.metal.manifold_physics import ThermodynamicsDomain, OmegaWaveDomain
                    self.thermo = ThermodynamicsDomain(grid_size=grid_size or self.grid_size)
                    self.wave = OmegaWaveDomain(grid_size=grid_size or self.grid_size)
                case "cuda":
                    # CUDA backend currently implements the thermodynamic gas domain only.
                    from sensorium.kernels.triton.manifold_physics import ThermodynamicsDomain
                    self.thermo = ThermodynamicsDomain(grid_size=grid_size or self.grid_size)
                    raise NotImplementedError("CUDA backend: OmegaWaveDomain not implemented")
                case _:
                    raise RuntimeError(f"Unsupported device: {self.device_name}")

    def load(self) -> None:
        """Load the first batch from the tokenizer as initial state."""
        try:
            with console.spinner("Loading data"):
                for token in self.tokenizer.stream():
                    self.step(token)

            console.success("Data loaded")
        except Exception as err:
            console.error(f"Error in load: {err}")
            raise err

    def step(self, token: tuple[int, int] | None = None) -> None:
        """Step the manifold."""
        try:
            if token is not None:
                self.state["particle"]["byte_values"].append(token[0])


            self._step += 1
            self.state = self.wave.step(**self.thermo.step(**self.state))
            self.state["step"] = self._step

            for instrument in self.instrumentation:
                instrument.update(self.state)
        except Exception as err:
            console.error(f"Error in step: {err}")
            os._exit(1)

    def run(self) -> dict:
        """Advance the manifold until completion.

        IMPORTANT: completion must be deterministic. We therefore always enforce a
        hard step budget (`self.max_steps`) and optionally stop earlier if the
        domain declares itself "done thinking".
        """
        if self.state is None:
            # Use the loader to construct a fully-formed initial state
            # (positions/velocities + tokenizer-derived fields).
            self.load()

        try:
            with console.spinner("Running manifold"):
                step_count = 0
                reason = "budget"
                while True:
                    self.step()
                    step_count += 1
                    if self.thermo.done_thinking(**self.state):
                        reason = "quiet"
                        break
                    if self.max_steps > 0 and step_count >= self.max_steps:
                        reason = "budget"
                        break

            if reason == "quiet":
                console.success("Completed", detail=f"{step_count} steps (quiet window)")
            else:
                console.success("Completed", detail=f"{step_count} steps (budget)")
            return self.state
        except Exception as err:
            console.error(f"Error in run: {err}")
            raise err