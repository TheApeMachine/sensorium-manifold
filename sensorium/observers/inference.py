"""Inference observer for the manifold.

The InferenceObserver is the central data store for experiment observations.
It accumulates results across multiple observe() calls and provides a
query interface for projectors to extract data by field name.

Design:
- Observers compute metrics and return dicts
- InferenceObserver accumulates these into a results list
- Projectors query the InferenceObserver for specific fields
- No manual data extraction needed in experiments

Example:
    # Setup
    inference = InferenceObserver([
        SpatialClustering(),
        CompressionRatio(),
    ])

    # Run multiple times (e.g., different collision rates)
    for dataset in datasets:
        manifold.add_dataset(dataset.generate)
        state = manifold.run()
        inference.observe(state, collision_rate=rate)  # Add metadata

    # Projector queries directly
    projector.project(inference)  # Gets data via inference.get("field_name")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Callable, Any, Sequence, List, Dict, Optional, Union, cast

import torch

from sensorium.observers.types import ObservationResult, ObserverProtocol
from sensorium.observers.dark.injector import DarkParticleInjector, DarkParticleConfig


@dataclass
class InferenceConfig:
    """Configuration for inference observation.

    Attributes:
        steps: Number of simulation steps to run after injection
        dark_config: Configuration for dark particle behavior
        cleanup_mode: How to handle dark particles after observation
            - "decay": Let dark particles decay naturally (default)
            - "immediate": Remove dark particles immediately after observation
            - "keep": Keep dark particles (caller responsible for cleanup)
        observer_query_cleanup_mode: Cleanup mode for observer-resolved dark queries
            - "immediate": Remove inferred query particles after observation (default)
            - "decay": Let inferred query particles decay
            - "keep": Keep inferred query particles
    """

    steps: int = 10
    dark_config: DarkParticleConfig = field(default_factory=DarkParticleConfig)
    cleanup_mode: str = "decay"
    observer_query_cleanup_mode: str = "immediate"


class InferenceObserver:
    """Central observer that accumulates results and provides query interface.

    The InferenceObserver:
    1. Runs a pipeline of metric observers on each state
    2. Accumulates results across multiple observe() calls
    3. Provides get()/query() interface for projectors

    This is the "data store" that projectors query to build outputs.

    Example:
        # Create with metric observers
        inference = InferenceObserver([
            SpatialClustering(),
            CompressionRatio(),
        ])

        # Accumulate results
        inference.observe(state1, label="run_1")
        inference.observe(state2, label="run_2")

        # Query accumulated data
        clustering_values = inference.get("spatial_clustering")  # List of values
        all_results = inference.results  # List of dicts

        # Projector uses this directly
        projector.project(inference)
    """

    def __init__(
        self,
        observers: Sequence[ObserverProtocol | Callable] | None = None,
        config: InferenceConfig | None = None,
        *,
        query: bytes | list[int] | Iterator[tuple[int, int]] | None = None,
        steps: int | None = None,
    ):
        """Initialize the inference observer.

        Args:
            observers: Pipeline of observers to apply
            config: Inference configuration
            query: Optional query data to inject as dark particles
            steps: Shortcut for config.steps (convenience)
        """
        self.observers = list(observers) if observers else []
        self.config = config or InferenceConfig()
        # Payload to inject as dark particles. Name avoids clashing with the
        # projector-facing `InferenceObserver.query(fields=...)` method.
        self.query_payload = query

        if steps is not None:
            self.config.steps = steps

        # Create dark particle injector
        self.injector = DarkParticleInjector(self.config.dark_config)

        # ACCUMULATED RESULTS - this is the key data store
        self._results: List[Dict[str, Any]] = []

        # Track last observation for backwards compatibility
        self._last_result: ObservationResult | None = None

    # =========================================================================
    # Query Interface - for projectors
    # =========================================================================

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Get all accumulated results as list of dicts."""
        return self._results

    def get(self, field: str, default: Any = None) -> List[Any]:
        """Get a specific field across all results.

        Args:
            field: Field name to extract
            default: Default value if field missing

        Returns:
            List of values for that field across all results
        """
        return [r.get(field, default) for r in self._results]

    def get_last(self, field: str, default: Any = None) -> Any:
        """Get a field from the last result only."""
        if not self._results:
            return default
        return self._results[-1].get(field, default)

    def query(self, fields: List[str]) -> List[Dict[str, Any]]:
        """Query multiple fields across all results.

        Args:
            fields: List of field names to extract

        Returns:
            List of dicts with only the requested fields
        """
        return [{f: r.get(f) for f in fields if f in r} for r in self._results]

    def as_dict(self) -> Dict[str, List[Any]]:
        """Convert results to column-oriented dict.

        Returns:
            Dict where keys are field names and values are lists
        """
        if not self._results:
            return {}

        # Get all field names
        all_fields = set()
        for r in self._results:
            all_fields.update(r.keys())

        return {field: self.get(field) for field in all_fields}

    def clear(self):
        """Clear accumulated results."""
        self._results = []
        self._last_result = None

    def __len__(self) -> int:
        """Number of accumulated results."""
        return len(self._results)

    def __iter__(self):
        """Iterate over results."""
        return iter(self._results)

    # =========================================================================
    # Observation Interface
    # =========================================================================

    @property
    def last_result(self) -> ObservationResult | None:
        """Get the result of the last observation."""
        return self._last_result

    def observe(
        self,
        state: dict | ObservationResult | None = None,
        *,
        manifold: Any | None = None,
        physics: Any | None = None,
        **metadata,
    ) -> Dict[str, Any]:
        """Run observers and accumulate results.

        Args:
            state: Simulation state dict
            manifold: Optional manifold reference for simulation steps
            physics: Optional physics engine for simulation steps
            **metadata: Additional fields to include in result (e.g., label, rate)

        Returns:
            Dict of observation results (also accumulated internally)
        """
        # Ensure we have a dict state
        if isinstance(state, ObservationResult):
            state_dict = state.data
        elif isinstance(state, dict):
            state_dict = state
        else:
            state_dict = {}

        # Determine device and dtype from state
        device = self._get_device(state_dict)
        dtype = self._get_dtype(state_dict)

        # Step 1: Resolve and inject dark query particles.
        effective_query = self.query_payload
        dark_query_source: str | None = None
        if effective_query is not None:
            dark_query_source = "explicit"
        else:
            effective_query = self._resolve_observer_query(state_dict, metadata)
            if effective_query is not None:
                dark_query_source = "observer"

        injected_indices: list[int] = []
        if effective_query is not None:
            injected_indices = self._inject_query(
                state_dict, effective_query, device, dtype
            )

        # Step 2: Run simulation steps (if manifold/physics provided)
        if self.config.steps > 0 and (manifold is not None or physics is not None):
            self._run_simulation(state_dict, manifold, physics, self.config.steps)

        # Step 3: Run observer pipeline, collecting all results
        result_dict: Dict[str, Any] = {}

        for observer in self.observers:
            obs_fn = getattr(observer, "observe", None)
            if callable(obs_fn):
                # Pass metadata through so metric observers can be keyed per-run without
                # forcing experiments to mutate state. Observers should ignore what they
                # don't need via **kwargs.
                obs_result = obs_fn(state_dict, manifold=manifold, **metadata)
            elif callable(observer):
                # Backward-compatible: callables may or may not accept metadata.
                try:
                    obs_result = observer(state_dict, manifold=manifold, **metadata)
                except TypeError:
                    obs_result = observer(state_dict)
            else:
                continue

            # Merge results
            if isinstance(obs_result, ObservationResult):
                result_dict.update(obs_result.data)
            elif isinstance(obs_result, dict):
                result_dict.update(obs_result)
            elif isinstance(obs_result, (int, float)):
                # Single value - use observer class name as key
                key = observer.__class__.__name__.lower()
                result_dict[key] = obs_result

        # Step 4: Add metadata
        result_dict.update(metadata)
        if dark_query_source is not None:
            result_dict["dark_query_source"] = dark_query_source
            result_dict["n_dark_injected"] = int(len(injected_indices))

        # Step 5: Cleanup dark particles
        cleanup_mode = self.config.cleanup_mode
        if dark_query_source == "observer" and cleanup_mode == "decay":
            cleanup_mode = self.config.observer_query_cleanup_mode

        if cleanup_mode == "immediate":
            self.injector.clear(state_dict)
        elif cleanup_mode == "decay":
            self.injector.decay_and_remove(state_dict)

        # Accumulate result
        self._results.append(result_dict)
        self._last_result = ObservationResult(data=result_dict, source="inference")

        return result_dict

    def _inject_query(
        self,
        state: dict,
        query: bytes | list[int] | Iterator[tuple[int, int]] | list[tuple[int, int]],
        device: str,
        dtype: torch.dtype,
    ) -> list[int]:
        """Inject query data as dark particles."""
        if isinstance(query, bytes):
            return self.injector.inject_bytes(state, query, device, dtype)

        if isinstance(query, list):
            if len(query) == 0:
                return []
            first = query[0]
            if isinstance(first, tuple) and len(first) == 2:
                pairs = cast(list[tuple[int, int]], query)
                return self.injector.inject(state, pairs, device, dtype)
            # Treat as a list[int] of bytes.
            bytes_list = cast(list[int], query)
            return self.injector.inject_bytes(state, bytes_list, device, dtype)

        return self.injector.inject(state, query, device, dtype)

    def _resolve_observer_query(
        self,
        state: dict,
        metadata: dict[str, Any],
    ) -> bytes | list[int] | Iterator[tuple[int, int]] | list[tuple[int, int]] | None:
        """Allow observers to request dark-query injection transparently.

        Observer contract (optional):
            def resolve_dark_query(self, state: dict, metadata: dict) -> query | None
        """
        for observer in self.observers:
            resolver = getattr(observer, "resolve_dark_query", None)
            if not callable(resolver):
                continue
            try:
                query = resolver(state, metadata=metadata)
            except TypeError:
                query = resolver(state, metadata)
            if query is not None:
                if isinstance(query, bytes):
                    return query
                if isinstance(query, list):
                    if len(query) == 0:
                        return []
                    first = query[0]
                    if isinstance(first, tuple) and len(first) == 2:
                        return cast(list[tuple[int, int]], query)
                    return cast(list[int], query)
                if hasattr(query, "__iter__") and hasattr(query, "__next__"):
                    return cast(Iterator[tuple[int, int]], query)
        return None

    def _run_simulation(
        self,
        state: dict,
        manifold: Any | None,
        physics: Any | None,
        steps: int,
    ):
        """Run simulation steps."""
        if manifold is not None and hasattr(manifold, "step"):
            for _ in range(steps):
                manifold.step(state)
        elif physics is not None and hasattr(physics, "step"):
            for _ in range(steps):
                physics.step(state)

    def _get_device(self, state: dict) -> str:
        """Get device from state tensors."""
        for key in ["positions", "energies", "excitations"]:
            tensor = state.get(key)
            if tensor is not None and hasattr(tensor, "device"):
                return str(tensor.device)
        return "mps"

    def _get_dtype(self, state: dict) -> torch.dtype:
        """Get dtype from state tensors."""
        for key in ["positions", "energies", "excitations"]:
            tensor = state.get(key)
            if tensor is not None and hasattr(tensor, "dtype"):
                return tensor.dtype
        return torch.float32

    def clear_dark_particles(self, state: dict) -> int:
        """Manually clear all dark particles."""
        return self.injector.clear(state)

    def decay_dark_particles(self, state: dict) -> list[int]:
        """Manually decay and remove depleted dark particles."""
        return self.injector.decay_and_remove(state)


# =============================================================================
# Convenience factory functions
# =============================================================================


def infer(
    query: bytes | list[int] | Iterator[tuple[int, int]],
    *observers: ObserverProtocol,
    steps: int = 10,
) -> InferenceObserver:
    """Create an inference observer with the given query and observers."""
    return InferenceObserver(
        query=query,
        observers=list(observers),
        config=InferenceConfig(steps=steps),
    )


def observe_reaction(
    *observers: ObserverProtocol,
) -> InferenceObserver:
    """Create an observer without dark particle injection."""
    return InferenceObserver(
        query=None,
        observers=list(observers),
        config=InferenceConfig(steps=0),
    )
