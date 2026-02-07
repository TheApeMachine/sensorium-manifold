"""Manifold is a wrapper around the physics simulation.

This is the main entry point for the Sensorium Manifold.
It is a wrapper around the physics simulation that provides a simple interface for
tokenizing data and running the physics simulation.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch

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
                    self.wave = OmegaWaveDomain(grid_size=grid_size or self.grid_size)
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
                    raise NotImplementedError(
                        "CUDA backend: OmegaWaveDomain not implemented"
                    )
                case _:
                    raise RuntimeError(f"Unsupported device: {self.device_name}")

    def _empty_state(self) -> dict[str, Any]:
        device = torch.device(self.device_name)
        gx, gy, gz = self.grid_size
        dx = float(1.0 / float(max(gx, gy, gz)))
        zeros_i64 = torch.zeros((0,), device=device, dtype=torch.int64)
        zeros_f32 = torch.zeros((0,), device=device, dtype=torch.float32)
        return {
            "positions": torch.zeros((0, 3), device=device, dtype=torch.float32),
            "velocities": torch.zeros((0, 3), device=device, dtype=torch.float32),
            "masses": zeros_f32,
            "heats": zeros_f32,
            "energies": zeros_f32,
            "energy_osc": zeros_f32,
            "excitations": zeros_f32,
            "omega": zeros_f32,
            "phase": zeros_f32,
            "token_ids": zeros_i64,
            "sequence_indices": zeros_i64,
            "dx": dx,
            "grid_size": tuple(self.grid_size),
        }

    def load(self) -> None:
        """Load the first batch from the tokenizer as initial state."""
        try:
            with console.spinner("Loading data"):
                loader = Loader(self.tokenizer, grid_size=self.grid_size)
                self.state = None
                for state in loader.stream():
                    self.state = state
                    break
                if self.state is None:
                    self.state = self._empty_state()

            console.success("Data loaded")
        except Exception as err:
            console.error(f"Error in load: {err}")
            raise err

    def step(
        self, token_or_state: tuple[int, int] | dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Step the manifold."""
        try:
            if isinstance(token_or_state, dict):
                self.state = token_or_state

            if self.state is None:
                self.load()
            assert isinstance(self.state, dict)

            self._step += 1
            assert self.thermo is not None
            assert self.wave is not None

            thermo_state = self.thermo.step(self.state)
            if not isinstance(thermo_state, dict):
                raise TypeError(
                    f"thermo.step must return dict, got {type(thermo_state)!r}"
                )
            next_state = self.wave.step(**thermo_state)
            if not isinstance(next_state, dict):
                raise TypeError(f"wave.step must return dict, got {type(next_state)!r}")
            next_state["step"] = self._step
            self.state = next_state

            for instrument in self.instrumentation:
                instrument.update(next_state)
            return next_state
        except Exception as err:
            console.error(f"Error in step: {err}")
            raise

    def run(self) -> dict[str, Any]:
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
                    assert self.thermo is not None
                    assert isinstance(self.state, dict)
                    if self.thermo.done_thinking(**self.state):
                        reason = "quiet"
                        break
                    if self.max_steps > 0 and step_count >= self.max_steps:
                        reason = "budget"
                        break

            if reason == "quiet":
                console.success(
                    "Completed", detail=f"{step_count} steps (quiet window)"
                )
            else:
                console.success("Completed", detail=f"{step_count} steps (budget)")
            assert isinstance(self.state, dict)
            return self.state
        except Exception as err:
            console.error(f"Error in run: {err}")
            raise err
