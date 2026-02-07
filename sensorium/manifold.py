"""Manifold is a wrapper around the physics simulation.

This is the main entry point for the Sensorium Manifold.
It is a wrapper around the physics simulation that provides a simple interface for
tokenizing data and running the physics simulation.
"""

from __future__ import annotations

from typing import Any, Sequence

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
        omega_num_modes: int | None = None,
    ) -> None:
        # Keep a direct reference: some callers use `manifold.tokenizer`.
        self.tokenizer = tokenizer
        self.grid_size = grid_size
        self.instrumentation = instrumentation or []
        self.max_steps = int(max_steps)
        self.omega_num_modes = (
            int(omega_num_modes)
            if isinstance(omega_num_modes, int) and omega_num_modes > 0
            else None
        )
        self.state: dict[str, Any] | None = None
        self._step = 0
        self.device_name = get_device()
        self.thermo: Any = None
        self.wave: Any = None
        self.reset(grid_size=grid_size)

    def reset(self, *, grid_size: tuple[int, int, int] = (64, 64, 64)) -> None:
        """Reset the manifold to its initial state."""
        self.state = None
        self._step = 0
        console.header(
            "Sensorium Manifold", Device=self.device_name, Grid=f"{self.grid_size}"
        )
        with console.spinner(f"Initializing physics on {self.device_name}"):
            match self.device_name:
                case "mps":
                    from sensorium.kernels.metal.manifold_physics import (
                        ThermodynamicsDomain,
                        OmegaWaveDomain,
                    )

                    self.thermo = ThermodynamicsDomain(
                        grid_size=grid_size or self.grid_size
                    )
                    self.wave = OmegaWaveDomain(
                        grid_size=grid_size or self.grid_size,
                        num_modes=self.omega_num_modes,
                    )
                case "cuda":
                    # CUDA backend currently implements the thermodynamic gas domain only.
                    from sensorium.kernels.triton.manifold_physics import (
                        ThermodynamicsDomain,
                        ThermodynamicsDomainConfig,
                    )

                    cfg = ThermodynamicsDomainConfig(
                        grid_size=grid_size or self.grid_size,
                        dt_max=0.01,
                    )
                    self.thermo = ThermodynamicsDomain(cfg, device="cuda")
                    self.wave = OmegaWaveDomain(
                        grid_size=grid_size or self.grid_size,
                        num_modes=self.omega_num_modes,
                    )
                    raise NotImplementedError(
                        "CUDA backend: OmegaWaveDomain not implemented"
                    )
                case _:
                    raise RuntimeError(f"Unsupported device: {self.device_name}")

    def load(self) -> None:
        """Load the first batch from the tokenizer as initial state."""
        try:
            with console.spinner("Loading data"):
                loader = Loader(self.tokenizer, grid_size=self.grid_size)
                for state in loader.stream():
                    self.step(state)

            console.success("Data loaded")
        except Exception as err:
            console.error(f"Error in load: {err}")
            raise err

    def step(
        self, state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Step the manifold."""
        try:
            assert self.thermo is not None
            assert self.wave is not None

            self.state = self.wave.step(**self.thermo.step(state or self.state))
            for instrument in self.instrumentation:
                instrument.update(self.state)
            return self.state
        except Exception as err:
            console.error(f"Error in step: {err}")
            raise

    def run(self) -> dict[str, Any]:
        """Advance the manifold until completion.

        This is the standard method for running the manifold, where it decides 
        for itself when to stop (done_thinking).
        In almost all cases, this should be the preferred method for running on
        real-world data and tasks.
        Manual stepping is only recommended for debugging or testing, and should
        NOT be used for experiments, because experiments need to show real-world
        operational performance. Do not think that the Manifold should be deterministic,
        if it isn't naturally deterministic (in its outcome), it isn't working correctly.
        The whole point is that you DO NOT have control over the system, you let things
        play out, and all you can do is observe the results. Observations give you a
        lot of flexibility and freedom to explore the system and its dynamics.
        """
        if self.state is None:
            self.load()

        try:
            with console.spinner("Running manifold"):
                while not self.thermo.done_thinking(**self.state):
                    self.step()

            console.success("Done thinking")
            return self.state
        except Exception as err:
            console.error(f"Error in run: {err}")
            raise err
