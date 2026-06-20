from __future__ import annotations

from laplace.adapter import LoadedModel


class FakeAdapter:
    def __init__(
        self,
        *,
        loaded: list[LoadedModel] | None = None,
        footprints: dict[str, int | None] | None = None,
    ):
        self.loaded: dict[str, LoadedModel] = {entry.id: entry for entry in (loaded or [])}
        self.footprints = dict(footprints or {})
        self.ensure_loaded_calls: list[tuple[str, int | None]] = []
        self.force_unload_calls: list[str] = []
        self.unload_outcomes: dict[str, list[bool]] = {}

    def base_id(self, model_id: str) -> str:
        if not model_id:
            return model_id
        head, sep, tail = model_id.rpartition(":")
        return head if (sep and tail.isdigit()) else model_id

    async def list_loaded(self) -> list[LoadedModel]:
        return [self.loaded[key] for key in sorted(self.loaded)]

    async def footprint_mb(self, model_id: str) -> int | None:
        base = self.base_id(model_id)
        for key, value in self.footprints.items():
            if self.base_id(key) == base:
                return value
        entry = self.loaded.get(model_id)
        if entry and entry.size_bytes is not None:
            return int(entry.size_bytes * 1.3 / (1024 * 1024))
        return None

    async def ensure_loaded(self, model_id: str, context_length: int | None) -> None:
        self.ensure_loaded_calls.append((model_id, context_length))
        current = self.loaded.get(model_id)
        size_bytes = current.size_bytes if current else None
        status = current.status if current else "idle"
        last_used_ms = current.last_used_ms if current else None
        queued = current.queued if current else 0
        self.loaded[model_id] = LoadedModel(
            id=model_id,
            base_id=self.base_id(model_id),
            size_bytes=size_bytes,
            status=status,
            last_used_ms=last_used_ms,
            queued=queued,
            context_length=context_length,
        )

    async def force_unload(self, model_id: str) -> bool:
        self.force_unload_calls.append(model_id)
        outcomes = self.unload_outcomes.get(model_id)
        if outcomes:
            ok = outcomes.pop(0)
            if not ok:
                return False
        base = self.base_id(model_id)
        removed = False
        for current_id in list(self.loaded):
            if current_id == model_id or (model_id == base and self.base_id(current_id) == base):
                self.loaded.pop(current_id, None)
                removed = True
        return removed or not any(self.base_id(current_id) == base for current_id in self.loaded)


def loaded_model(
    model_id: str,
    *,
    size_bytes: int | None = None,
    status: str | None = None,
    last_used_ms: float | None = None,
    queued: int | None = None,
    context_length: int | None = None,
) -> LoadedModel:
    adapter = FakeAdapter()
    return LoadedModel(
        id=model_id,
        base_id=adapter.base_id(model_id),
        size_bytes=size_bytes,
        status=status,
        last_used_ms=last_used_ms,
        queued=queued,
        context_length=context_length,
    )
