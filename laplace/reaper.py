"""Model-eviction policy owner for the standalone Laplace core."""

from __future__ import annotations

import asyncio
import logging
import time

from laplace.adapter import InferenceAdapter, LoadedModel

log = logging.getLogger(__name__)

_GIB = 1024 ** 3
_LARGE_MODEL_BYTES = 30 * _GIB
_MEDIUM_MODEL_BYTES = 15 * _GIB
_EXTERNAL_IDLE_TTL = 300.0
_EXTERNAL_IDLE_SWEEPS = 2


class Reaper:
    """Owns eviction policy for a shared inference pool."""

    def __init__(
        self,
        adapter: InferenceAdapter,
        *,
        delay_seconds: float = 300.0,
        keep_loaded: set[str] | None = None,
    ):
        self.adapter = adapter
        self._delay = delay_seconds
        self._keep_loaded = frozenset(keep_loaded or set())
        self._last_used: dict[str, float] = {}
        self._pending: dict[str, asyncio.Task] = {}
        self._in_flight: dict[str, int] = {}
        self._loading: dict[str, int] = {}
        self._unloading: dict[str, int] = {}
        self._unload_done: dict[str, asyncio.Event] = {}
        self._base_index: dict[str, set[str]] = {}
        self._managed: set[str] = set()
        self._loaded_size_bytes: dict[str, int] = {}
        self._reap_tasks: set[asyncio.Task] = set()
        self._sweep_task: asyncio.Task | None = None
        self._last_successful_sweep: float | None = None
        self._broker = None
        self._ext_idle_sweeps: dict[str, int] = {}
        self._ext_last_used_seen: dict[str, float] = {}

    def set_broker(self, broker) -> None:
        self._broker = broker

    def touch(self, model_id: str) -> None:
        if not model_id or self._is_keep_loaded(model_id):
            return
        self._last_used[model_id] = time.monotonic()
        task = self._pending.pop(model_id, None)
        if task and not task.done():
            task.cancel()

    def acquire(self, model_id: str) -> None:
        if not model_id:
            return
        resolved = self._resolve_acquire_model_id(model_id)
        self._track_base_instance(resolved)
        self._mark_managed(resolved)
        self._mark_managed(model_id)
        self._in_flight[resolved] = self._in_flight.get(resolved, 0) + 1
        self._last_used[resolved] = time.monotonic()
        task = self._pending.pop(resolved, None)
        if task and not task.done():
            task.cancel()

    def start_loading(self, model_id: str) -> None:
        """Pin and touch a model while an admission is making it resident."""
        if not model_id:
            return
        base_id = self.adapter.base_id(model_id)
        self.touch(model_id)
        self._loading[base_id] = self._loading.get(base_id, 0) + 1

    def finish_loading(self, model_id: str) -> None:
        """Release a load pin and protect the just-loaded model for a full TTL."""
        if not model_id:
            return
        base_id = self.adapter.base_id(model_id)
        remaining = self._loading.get(base_id, 0) - 1
        if remaining > 0:
            self._loading[base_id] = remaining
        else:
            self._loading.pop(base_id, None)
        self.touch(model_id)

    def release(self, model_id: str) -> None:
        if not model_id:
            return
        resolved = self._resolve_release_model_id(model_id)
        remaining = self._in_flight.get(resolved, 0) - 1
        if remaining > 0:
            self._in_flight[resolved] = remaining
        else:
            self._in_flight.pop(resolved, None)
            self._untrack_base_instance(resolved)
        self._last_used[resolved] = time.monotonic()
        self._spawn_reap()

    def schedule_unload(self, model_id: str) -> None:
        if not model_id or self._is_keep_loaded(model_id):
            return
        self._last_used[model_id] = time.monotonic()
        old_task = self._pending.pop(model_id, None)
        if old_task and not old_task.done():
            old_task.cancel()
        self._pending[model_id] = asyncio.create_task(self._delayed_unload(model_id))

    async def _delayed_unload(self, model_id: str) -> None:
        try:
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            return

        last_use = self._last_used.get(model_id, 0.0)
        elapsed = time.monotonic() - last_use
        if elapsed < self._delay:
            self._pending[model_id] = asyncio.create_task(self._delayed_unload(model_id))
            return
        if self._is_loading(model_id):
            self._pending[model_id] = asyncio.create_task(self._delayed_unload(model_id))
            return
        if self._in_flight.get(model_id, 0) > 0 or self._in_flight.get(self.adapter.base_id(model_id), 0) > 0:
            self._pending[model_id] = asyncio.create_task(self._delayed_unload(model_id))
            return
        self._pending.pop(model_id, None)
        await self._unload(model_id, reason="reaper:idle-delay")

    def is_unloading(self, model_id: str) -> bool:
        """Whether a reaper unload of this model's base is in progress."""
        if not model_id:
            return False
        return self._unloading.get(self.adapter.base_id(model_id), 0) > 0

    async def wait_unloaded(self, model_id: str) -> None:
        """Return once no reaper unload of this model's base is in progress."""
        if not model_id:
            return
        base_id = self.adapter.base_id(model_id)
        while self._unloading.get(base_id, 0) > 0:
            await self._unload_done[base_id].wait()

    async def _unload(self, model_id: str, *, reason: str) -> bool:
        """Unload ``model_id`` unless an admission has claimed it since selection.

        Callers pick victims from a list built before earlier awaits, so the
        claim check runs again here. The check and the unloading mark have no
        await between them, and govern() pins with start_loading() before it
        checks is_unloading(). So either this skips the model, or the admission
        waits until the unload, including its retry, has returned.
        """
        base_id = self.adapter.base_id(model_id)
        if self._is_loading(model_id) or self._in_flight.get(model_id, 0) > 0:
            log.info(
                "laplace.reaper: skip unload of %s (%s): claimed by an admission",
                model_id,
                reason,
            )
            return False
        self._unloading[base_id] = self._unloading.get(base_id, 0) + 1
        done = self._unload_done.setdefault(base_id, asyncio.Event())
        try:
            return await self.adapter.force_unload(model_id, reason=reason)
        finally:
            remaining = self._unloading.get(base_id, 0) - 1
            if remaining > 0:
                self._unloading[base_id] = remaining
            else:
                self._unloading.pop(base_id, None)
                self._unload_done.pop(base_id, None)
                done.set()
                wake = getattr(self._broker, "wake_waiter", None)
                if wake is not None:
                    # Wake an admission the broker queued behind this unload.
                    wake()

    @property
    def pending_unloads(self) -> list[str]:
        return [model_id for model_id, task in self._pending.items() if not task.done()]

    def in_flight(self, model_id: str) -> int:
        return self._in_flight.get(model_id, 0)

    def mark_managed(self, model_id: str) -> None:
        """Claim a model as daemon-managed before acquire.

        Marking is a claim of imminent use: callers mark a resident model up
        front so the idle-TTL sweep does not treat it as an external tenant
        during the load/acquire window. The mark also records a touch so an
        unset _last_used is not read as infinitely idle and reaped before
        acquire records its first touch.
        """
        self._mark_managed(model_id)
        self._last_used[model_id] = time.monotonic()

    def is_managed(self, model_id: str) -> bool:
        if not model_id:
            return False
        return model_id in self._managed or self.adapter.base_id(model_id) in self._managed

    def unmark_managed(self, model_id: str) -> None:
        """Release a managed claim (first-touch rollback on a failed load).

        Safe only under the caller's first-touch rule: unmark when this caller
        took first touch, never acquired, and no other call holds the model.
        A no-op while the model is in flight, so a caller race cannot flip an
        actively-used model to external mid-call.
        """
        if not model_id:
            return
        base_id = self.adapter.base_id(model_id)
        if self._in_flight.get(model_id, 0) > 0 or self._in_flight.get(base_id, 0) > 0:
            return
        self._managed.discard(model_id)
        self._managed.discard(base_id)

    def evictable_models(self, resident: list[str]) -> list[str]:
        self._reconcile_base_index(resident)
        candidates = [
            model_id
            for model_id in resident
            if model_id
            and self._in_flight.get(model_id, 0) == 0
            and self._in_flight.get(self.adapter.base_id(model_id), 0) == 0
            and not self._is_loading(model_id)
            and not self._is_keep_loaded(model_id)
            and not self._is_external(model_id)
        ]
        return sorted(candidates, key=self._last_used_at)

    async def reap_duplicates(self, entries: list[LoadedModel] | None = None) -> list[str]:
        if entries is None:
            entries = await self.adapter.list_loaded()
        groups: dict[str, list[str]] = {}
        for entry in entries:
            if not entry.id:
                continue
            groups.setdefault(entry.base_id, []).append(entry.id)

        reaped: list[str] = []
        for base_id, instances in groups.items():
            if len(instances) < 2:
                continue
            if self._is_keep_loaded(base_id):
                continue
            if self._is_loading(base_id):
                continue
            in_flight_instances = [mid for mid in instances if self._in_flight.get(mid, 0) > 0]
            if in_flight_instances:
                keep = max(in_flight_instances, key=self._last_used_at)
            else:
                keep = max(instances, key=self._last_used_at)
            for model_id in instances:
                if model_id == keep:
                    continue
                if model_id == base_id:
                    # A plain-base request names this identifier, and in-flight
                    # attribution between siblings is a guess
                    # (_resolve_acquire_model_id), so never reap it as a duplicate.
                    continue
                if self._in_flight.get(model_id, 0) > 0:
                    continue
                unloaded = await self._unload(model_id, reason="reaper:duplicate")
                if unloaded:
                    reaped.append(model_id)
        return reaped

    def _model_ttl(self, model_id: str) -> float:
        size = self._loaded_size_bytes.get(model_id)
        if size is None:
            size = self._loaded_size_bytes.get(self.adapter.base_id(model_id))
        if size is None:
            return self._delay
        if size > _LARGE_MODEL_BYTES:
            return 120.0
        if size >= _MEDIUM_MODEL_BYTES:
            return 180.0
        return self._delay

    @staticmethod
    def reported_busy(entry: LoadedModel) -> bool:
        """Runtime-reported activity: queued work, or any status but idle.

        Unknown status counts as busy: an external tenant may be generating.
        """
        if entry.queued:
            return True
        return Reaper._entry_status(entry) != "idle"

    @staticmethod
    def _entry_status(entry: LoadedModel) -> str | None:
        return entry.status.lower() if isinstance(entry.status, str) and entry.status else None

    @staticmethod
    def _entry_last_used_ms(entry: LoadedModel) -> float | None:
        return entry.last_used_ms if entry.last_used_ms and entry.last_used_ms > 0 else None

    @staticmethod
    def _entry_queued(entry: LoadedModel) -> int | None:
        return entry.queued if isinstance(entry.queued, int) and entry.queued >= 0 else None

    def _eval_external_idle(
        self,
        model_id: str,
        entry: LoadedModel,
        now_wall_ms: float,
    ) -> tuple[bool, float | None, str]:
        status = self._entry_status(entry)
        queued = self._entry_queued(entry)
        last_used_ms = self._entry_last_used_ms(entry)

        prev_seen = self._ext_last_used_seen.get(model_id)
        if last_used_ms is not None and prev_seen is not None and last_used_ms > prev_seen:
            self._ext_idle_sweeps[model_id] = 0
        if last_used_ms is not None:
            self._ext_last_used_seen[model_id] = last_used_ms

        if status != "idle":
            self._ext_idle_sweeps[model_id] = 0
            return False, None, f"status={status or 'unknown'}"
        if queued is None or queued > 0:
            self._ext_idle_sweeps[model_id] = 0
            return False, None, f"queued={queued if queued is not None else 'unknown'}"
        if last_used_ms is None:
            self._ext_idle_sweeps[model_id] = 0
            return False, None, "lastUsedTime unknown"
        idle_secs = (now_wall_ms - last_used_ms) / 1000.0
        if idle_secs <= _EXTERNAL_IDLE_TTL:
            self._ext_idle_sweeps[model_id] = 0
            return False, idle_secs, f"idle {idle_secs:.0f}s < ttl {_EXTERNAL_IDLE_TTL:.0f}s"

        streak = self._ext_idle_sweeps.get(model_id, 0) + 1
        self._ext_idle_sweeps[model_id] = streak
        if streak < _EXTERNAL_IDLE_SWEEPS:
            return False, idle_secs, f"sustained-idle {streak}/{_EXTERNAL_IDLE_SWEEPS}"
        return True, idle_secs, "sustained-idle met"

    def _prune_external_counters(self, live_ids: set[str]) -> None:
        for tracker in (self._ext_idle_sweeps, self._ext_last_used_seen):
            for stale in [key for key in tracker if key not in live_ids]:
                tracker.pop(stale, None)

    async def sweep_loaded(self) -> None:
        entries = await self.adapter.list_loaded()
        reaped = await self.reap_duplicates(entries)

        live_entries = [entry for entry in entries if entry.id not in reaped]
        loaded = [entry.id for entry in live_entries if entry.id]
        entry_by_id = {entry.id: entry for entry in live_entries if entry.id}
        self._cache_loaded_entries(live_entries)
        self._reconcile_base_index(loaded)
        self._prune_external_counters(set(loaded))

        now = time.monotonic()
        now_wall_ms = time.time() * 1000.0
        bot_to_unload: list[str] = []
        external_to_unload: list[tuple[str, float | None]] = []

        for model_id in loaded:
            if not model_id or self._is_keep_loaded(model_id):
                continue
            if self._is_external(model_id):
                reap, idle_secs, _reason = self._eval_external_idle(
                    model_id,
                    entry_by_id[model_id],
                    now_wall_ms,
                )
                if reap:
                    external_to_unload.append((model_id, idle_secs))
                continue
            base_id = self.adapter.base_id(model_id)
            if self._is_loading(base_id):
                continue
            if self._in_flight.get(base_id, 0) > 0 and model_id in self._base_index.get(base_id, {model_id}):
                continue
            if self._in_flight.get(model_id, 0) > 0:
                continue
            last = self._last_used_at(model_id)
            ttl = self._model_ttl(model_id)
            if last and (now - last) < ttl:
                continue
            bot_to_unload.append(model_id)

        freed_mb = 0
        unloaded_count = 0

        for model_id in bot_to_unload:
            size_bytes = self._loaded_size_bytes.get(model_id) or self._loaded_size_bytes.get(
                self.adapter.base_id(model_id)
            )
            unloaded = await self._unload(model_id, reason="reaper:managed-idle-sweep")
            if not unloaded:
                continue
            unloaded_count += 1
            if size_bytes:
                freed_mb += int(size_bytes / (1024 * 1024))

        fresh_by_id: dict[str, LoadedModel] = {}
        if external_to_unload:
            fresh_entries = await self.adapter.list_loaded()
            fresh_by_id = {entry.id: entry for entry in fresh_entries if entry.id}

        for model_id, idle_secs in external_to_unload:
            fresh = fresh_by_id.get(model_id)
            abort_reason = None
            if fresh is None:
                abort_reason = "no longer resident"
            else:
                fresh_status = self._entry_status(fresh)
                fresh_queued = self._entry_queued(fresh)
                fresh_last_used = self._entry_last_used_ms(fresh)
                prev_seen = self._ext_last_used_seen.get(model_id)
                if fresh_status != "idle":
                    abort_reason = f"status={fresh_status or 'unknown'}"
                elif fresh_queued is None or fresh_queued > 0:
                    abort_reason = (
                        f"queued={fresh_queued if fresh_queued is not None else 'unknown'}"
                    )
                elif fresh_last_used is not None and prev_seen is not None and fresh_last_used > prev_seen:
                    abort_reason = "lastUsedTime advanced"
            if abort_reason is not None:
                self._ext_idle_sweeps[model_id] = 0
                continue
            size_bytes = self._loaded_size_bytes.get(model_id) or self._loaded_size_bytes.get(
                self.adapter.base_id(model_id)
            )
            unloaded = await self._unload(model_id, reason="reaper:external-idle-sweep")
            self._ext_idle_sweeps.pop(model_id, None)
            self._ext_last_used_seen.pop(model_id, None)
            if not unloaded:
                continue
            unloaded_count += 1
            if size_bytes:
                freed_mb += int(size_bytes / (1024 * 1024))

        if unloaded_count and self._broker is not None:
            await self._broker.notify_freed(freed_mb)
        self._last_successful_sweep = time.monotonic()

    def sweep_health(self, max_age_s: float) -> str:
        """Return the current autonomous-sweep state for readiness checks."""
        if self._sweep_task is None or self._sweep_task.done():
            return "stopped"
        if self._last_successful_sweep is None:
            return "starting"
        if time.monotonic() - self._last_successful_sweep > max_age_s:
            return "stale"
        return "healthy"

    async def request_free(self, needed_mb: int, priority=None, *, headroom_mb=None) -> bool:
        """Evict idle models, oldest first, until ``needed_mb`` is freed.

        With ``headroom_mb`` (an async live-headroom probe), freed memory is the
        measured headroom gain since eviction began; a victim's file size is
        credited only when the probe cannot be read.
        """
        if needed_mb <= 0:
            return True
        entries = await self.adapter.list_loaded()
        if not entries:
            return False
        self._cache_loaded_entries(entries)
        resident = [entry.id for entry in entries if entry.id]
        baseline = await headroom_mb() if headroom_mb is not None else None
        # Live path: never evict a model the runtime reports as active.
        busy = {
            entry.id for entry in entries
            if headroom_mb is not None and self.reported_busy(entry)
        }

        freed_mb = 0
        for victim in self.evictable_models(resident):
            if freed_mb >= needed_mb:
                break
            if self.in_flight(victim) > 0 or self.in_flight(self.adapter.base_id(victim)) > 0:
                continue
            if victim in busy:
                continue
            size_bytes = self._loaded_size_bytes.get(victim) or self._loaded_size_bytes.get(
                self.adapter.base_id(victim)
            )
            unloaded = await self._unload(victim, reason="reaper:request-free")
            if not unloaded:
                continue
            now = await headroom_mb() if baseline is not None else None
            if now is not None:
                freed_mb = max(freed_mb, now - baseline)
                log.info(
                    "laplace.reaper: request-free unloaded %s; measured freed %s MB of %s",
                    victim,
                    freed_mb,
                    needed_mb,
                )
            elif size_bytes:
                freed_mb += int(size_bytes / (1024 * 1024))
            else:
                freed_mb += needed_mb
        return freed_mb >= needed_mb

    async def run_sweep_loop(self, interval: float = 60.0) -> None:
        task = asyncio.current_task()
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.sweep_loaded()
                except Exception:
                    log.exception("laplace.reaper: sweep loop tick failed")
        finally:
            if self._sweep_task is task:
                self._sweep_task = None

    def start_sweep_loop(self, interval: float = 60.0) -> asyncio.Task:
        if self._sweep_task and not self._sweep_task.done():
            return self._sweep_task
        self._sweep_task = asyncio.create_task(
            self.run_sweep_loop(interval),
            name="laplace:reaper-sweep-loop",
        )
        return self._sweep_task

    def stop_sweep_loop(self) -> None:
        task = self._sweep_task
        self._sweep_task = None
        if task and not task.done():
            task.cancel()

    def _spawn_reap(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.reap_duplicates())
        self._reap_tasks.add(task)
        task.add_done_callback(self._reap_tasks.discard)

    def _track_base_instance(self, model_id: str) -> None:
        if not model_id:
            return
        base_id = self.adapter.base_id(model_id)
        self._base_index.setdefault(base_id, set()).add(model_id)

    def _untrack_base_instance(self, model_id: str) -> None:
        if not model_id:
            return
        base_id = self.adapter.base_id(model_id)
        instances = self._base_index.get(base_id)
        if not instances:
            return
        instances.discard(model_id)
        if not instances:
            self._base_index.pop(base_id, None)

    def _cache_loaded_entries(self, entries: list[LoadedModel]) -> None:
        sizes: dict[str, int] = {}
        for entry in entries:
            if not entry.id or entry.size_bytes is None:
                continue
            sizes[entry.id] = entry.size_bytes
            sizes.setdefault(entry.base_id, entry.size_bytes)
        self._loaded_size_bytes = sizes

    def _last_used_at(self, model_id: str) -> float:
        base_id = self.adapter.base_id(model_id)
        return self._last_used.get(model_id, self._last_used.get(base_id, 0.0))

    def _reconcile_base_index(self, model_ids: list[str]) -> None:
        live_groups: dict[str, set[str]] = {}
        for model_id in model_ids:
            if not model_id:
                continue
            live_groups.setdefault(self.adapter.base_id(model_id), set()).add(model_id)

        tracked_bases = {
            self.adapter.base_id(model_id)
            for model_id in (*self._in_flight, *self._last_used, *self._pending)
            if model_id
        }
        for base_id, instances in live_groups.items():
            self._base_index[base_id] = instances
        for base_id in list(self._base_index):
            if base_id in live_groups or base_id in tracked_bases:
                continue
            self._base_index.pop(base_id, None)

    def _mark_managed(self, model_id: str) -> None:
        if not model_id:
            return
        self._managed.add(model_id)
        self._managed.add(self.adapter.base_id(model_id))

    def _is_external(self, model_id: str) -> bool:
        if not model_id:
            return False
        base_id = self.adapter.base_id(model_id)
        if model_id in self._managed or base_id in self._managed:
            return False
        if self._in_flight.get(model_id, 0) > 0 or self._in_flight.get(base_id, 0) > 0:
            return False
        return True

    def _is_loading(self, model_id: str) -> bool:
        return self._loading.get(self.adapter.base_id(model_id), 0) > 0

    def _resolve_acquire_model_id(self, model_id: str) -> str:
        base_id = self.adapter.base_id(model_id)
        if base_id != model_id:
            return model_id
        instances = self._base_index.get(base_id, set())
        if not instances:
            return model_id
        if len(instances) == 1:
            return next(iter(instances))
        return max(instances, key=self._last_used_at)

    def _resolve_release_model_id(self, model_id: str) -> str:
        base_id = self.adapter.base_id(model_id)
        if base_id != model_id:
            return model_id
        if self._in_flight.get(model_id, 0) > 0:
            return model_id
        instances = self._base_index.get(base_id, set())
        if not instances:
            return model_id
        if len(instances) == 1:
            return next(iter(instances))
        in_flight = [instance for instance in instances if self._in_flight.get(instance, 0) > 0]
        if len(in_flight) == 1:
            return in_flight[0]
        if in_flight:
            return max(in_flight, key=self._last_used_at)
        return max(instances, key=self._last_used_at)

    def _is_keep_loaded(self, model_id: str) -> bool:
        return any(marker in model_id for marker in self._keep_loaded)
