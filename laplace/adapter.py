"""Engine boundary for the standalone Laplace core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ModelLoadError(RuntimeError):
    """The runtime could not make the requested model ready for inference."""


@dataclass
class LoadedModel:
    """A loaded model instance reported by an inference runtime.

    ``status``, ``last_used_ms``, and ``queued`` form the external-tenant
    activity signal the Reaper uses for safe out-of-band eviction decisions.
    """

    id: str
    base_id: str
    size_bytes: int | None
    status: str | None = None
    last_used_ms: float | None = None
    queued: int | None = None
    context_length: int | None = None


@dataclass(frozen=True)
class LoadCost:
    """Peak memory a model needs at the context and parallel it loads with.

    ``estimate`` lists every input that was not derived from observed data
    (catalog size, architecture metadata, configured parallel). Empty means
    every term came from an observation.
    """

    weights_mb: int
    kv_mb: int
    overhead_mb: int
    context_length: int | None
    parallel: int
    estimate: tuple[str, ...] = ()

    @property
    def total_mb(self) -> int:
        return self.weights_mb + self.kv_mb + self.overhead_mb


class InferenceAdapter(Protocol):
    # Optional: ``async def load_cost(model_id, context_length) -> LoadCost | None``.
    # The broker uses it for live-memory admission when the adapter provides it.


    async def list_loaded(self) -> list[LoadedModel]:
        """Return the currently resident model instances."""

    async def footprint_mb(self, model_id: str) -> int | None:
        """Return the model footprint in MB, or ``None`` when unknown."""

    async def ensure_loaded(self, model_id: str, context_length: int | None) -> None:
        """Ensure ``model_id`` is resident at the requested context length."""

    def guard_window_remaining_s(self) -> float | None:
        """Return the active uncertain-load guard duration, if any."""

    async def force_unload(self, model_id: str, *, reason: str = "direct") -> bool:
        """Unload ``model_id`` and verify it is gone."""

    def base_id(self, model_id: str) -> str:
        """Collapse runtime-specific instance ids onto a stable base id."""
