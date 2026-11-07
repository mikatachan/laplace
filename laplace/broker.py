"""Priority-aware admission gate for a shared inference pool."""

from __future__ import annotations

import asyncio
import heapq
import logging
from typing import Awaitable, Callable, Optional

from laplace.adapter import InferenceAdapter, LoadCost, LoadedModel
from laplace.priority import PriorityTier, from_legacy_string
from laplace.reaper import Reaper

log = logging.getLogger(__name__)

FLAT_MARGIN_MB = 3072
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_CONCURRENT_LMS = 2


class ContentionBroker:
    """Admission gate for a single local inference pool."""

    def __init__(
        self,
        adapter: InferenceAdapter,
        reaper: Reaper,
        *,
        budget_mb: int | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_concurrent_lms: int = DEFAULT_MAX_CONCURRENT_LMS,
        headroom_mb: Optional[Callable[[], Awaitable[Optional[int]]]] = None,
    ):
        """``headroom_mb`` enables live-memory admission: a cold load must fit
        in the live headroom it returns, after reserving KV growth for busy
        resident models and the full cost of admitted loads still pending.
        Without it, admission uses the static footprint budget."""
        self._adapter = adapter
        self._reaper = reaper
        self._budget_override = budget_mb
        self._timeout = timeout_s
        self._max_concurrent = max(1, int(max_concurrent_lms))
        self._lock = asyncio.Lock()
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._counter = 0
        self._reserved: dict[str, int] = {}
        self._budget_cache: int | None = None
        self._footprint_cache: dict[str, int | None] = {}
        self._headroom = headroom_mb
        self._reserved_ctx: dict[str, int | None] = {}
        self._reaper.set_broker(self)

    async def admit(
        self,
        model_id: str,
        context_length: int | None = None,
        priority: str = "scheduled",
    ) -> str:
        if not model_id:
            return "admit"
        try:
            tier = from_legacy_string(priority)
        except ValueError:
            log.warning(
                "laplace.broker: unknown priority %r for %s; falling back to SCHEDULED",
                priority,
                model_id,
            )
            tier = PriorityTier.SCHEDULED

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        evicted_this_cycle = False

        while True:
            async with self._lock:
                decision = await self._try_admit(
                    model_id,
                    tier,
                    allow_evict=not evicted_this_cycle,
                    context_length=context_length,
                )
                if decision == "admit":
                    self._wake()
                    return "admit"
                if isinstance(decision, tuple) and decision[0] == "evict":
                    _, needed_mb, evict_prio = decision
                else:
                    needed_mb = None
                    evict_prio = None
            if needed_mb is not None:
                try:
                    if self._headroom is not None:
                        await self._reaper.request_free(
                            needed_mb, evict_prio, headroom_mb=self._headroom
                        )
                    else:
                        await self._reaper.request_free(needed_mb, evict_prio)
                finally:
                    evicted_this_cycle = True
                continue

            async with self._lock:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return "backpressure"
                fut = loop.create_future()
                self._counter += 1
                entry = (int(tier), self._counter, fut)
                heapq.heappush(self._waiters, entry)
                if self._waiters and self._waiters[0] is not entry:
                    self._wake()

            try:
                await asyncio.wait_for(fut, timeout=remaining)
            except asyncio.TimeoutError:
                async with self._lock:
                    self._remove_waiter(entry)
                return "backpressure"
            except asyncio.CancelledError:
                # The admit() task was cancelled while parked on the heap. Without
                # this cleanup the entry lingers in self._waiters until a later
                # _wake() happens to pop it: heap bloat under a cancel-heavy
                # workload. Remove it, then re-raise to preserve cancellation.
                # Acquiring the lock is itself a cancellation point, so retry the
                # guarded removal across re-cancellation; the lock is held only
                # for the O(n) heap remove, never across an await, so this
                # terminates for one-shot and bounded cancellation.
                while True:
                    try:
                        async with self._lock:
                            self._remove_waiter(entry)
                        break
                    except asyncio.CancelledError:
                        continue
                raise
            evicted_this_cycle = False

    async def done(self, model_id: str) -> None:
        if not model_id:
            return
        # Cancellation safety: acquiring the lock is a cancellation point, so a
        # second cancel thrown into the acquire could abort this before the
        # refcount is decremented and leak the reservation, leaving
        # _active_calls() elevated against phantom load. Retry the guarded
        # release across re-cancellation; the decrement runs exactly once
        # (guarded by `released`), so retrying never double-releases.
        released = False
        while True:
            try:
                async with self._lock:
                    if not released:
                        count = self._reserved.get(model_id, 0) - 1
                        if count > 0:
                            self._reserved[model_id] = count
                        else:
                            self._reserved.pop(model_id, None)
                            self._reserved_ctx.pop(model_id, None)
                        released = True
                    self._wake()
                return
            except asyncio.CancelledError:
                if released:
                    raise
                continue

    async def notify_freed(self, freed_mb: int | None = None) -> None:
        if freed_mb is not None:
            log.info("laplace.broker: notify_freed(%s MB)", freed_mb)
        async with self._lock:
            self._wake()

    def _wake(self) -> None:
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)
                return

    def _remove_waiter(self, entry: tuple[int, int, asyncio.Future]) -> None:
        try:
            self._waiters.remove(entry)
        except ValueError:
            return
        heapq.heapify(self._waiters)

    async def _try_admit(
        self,
        model_id: str,
        priority: PriorityTier,
        *,
        allow_evict: bool = True,
        context_length: int | None = None,
    ):
        if self._headroom is not None and hasattr(self._adapter, "load_cost"):
            return await self._try_admit_live(model_id, priority, allow_evict, context_length)
        footprint = await self._footprint_mb(model_id)
        if footprint is None:
            self._reserve(model_id)
            return "admit"
        if self._active_calls() >= self._max_concurrent:
            return None

        budget = await self._budget_mb()
        resident = await self._resident()

        if self._is_committed(model_id, resident):
            self._reserve(model_id)
            return "admit"

        used = await self._committed_mb(resident)
        if used + footprint <= budget:
            self._reserve(model_id)
            return "admit"

        evictable = [
            victim for victim in self._reaper.evictable_models(resident) if not self._is_reserved(victim)
        ]
        non_evictable = [model for model in resident if model not in evictable]
        blocked_mb = await self._committed_mb(non_evictable)
        if blocked_mb + footprint > budget:
            return None
        if not allow_evict:
            return None

        used = await self._committed_mb(resident)
        needed_mb = used + footprint - budget
        return ("evict", needed_mb, priority)

    async def _try_admit_live(
        self,
        model_id: str,
        priority: PriorityTier,
        allow_evict: bool,
        context_length: int | None,
    ):
        if self._active_calls() >= self._max_concurrent:
            return None
        entries = [entry for entry in await self._adapter.list_loaded() if entry.id]
        resident = [entry.id for entry in entries]
        if self._is_committed(model_id, resident):
            # No new load: memory for this model is already resident or reserved.
            self._reserve(model_id, context_length)
            return "admit"

        cost = await self._adapter.load_cost(model_id, context_length)
        if cost is None:
            log.warning(
                "laplace.broker: load decision model=%s ctx=%s decision=queue "
                "reason=cost-unknown (no catalog size or footprint table entry)",
                model_id,
                context_length,
            )
            return None
        headroom = await self._headroom()
        if headroom is None:
            log.warning(
                "laplace.broker: load decision model=%s cost_mb=%s decision=queue "
                "reason=headroom-unreadable",
                model_id,
                cost.total_mb,
            )
            return None

        inflight_mb, inflight_estimate = await self._inflight_reserve_mb(entries, context_length)
        pending_mb, pending_estimate = await self._pending_load_mb(resident)
        available = headroom - inflight_mb - pending_mb
        estimate = (*cost.estimate, *inflight_estimate, *pending_estimate)

        def note(decision: str, **extra) -> None:
            log.info(
                "laplace.broker: load decision model=%s ctx=%s parallel=%s cost_mb=%s "
                "(weights=%s kv=%s overhead=%s) headroom_mb=%s inflight_reserve_mb=%s "
                "pending_mb=%s available_mb=%s basis=%s decision=%s%s",
                model_id,
                cost.context_length,
                cost.parallel,
                cost.total_mb,
                cost.weights_mb,
                cost.kv_mb,
                cost.overhead_mb,
                headroom,
                inflight_mb,
                pending_mb,
                available,
                f"estimate({','.join(estimate)})" if estimate else "derived",
                decision,
                "".join(f" {key}={value}" for key, value in extra.items()),
            )

        if cost.total_mb <= available:
            note("admit")
            self._reserve(model_id, context_length)
            return "admit"

        shortfall = cost.total_mb - available
        evictable_ids = set(self._reaper.evictable_models(resident))
        evictable = [
            entry for entry in entries
            if entry.id in evictable_ids and not self._is_reserved(entry.id)
        ]
        # Credit idle victims with weights + overhead only: KV they no longer
        # hold is not counted, so eviction is chosen only when it surely helps.
        credit = 0
        for entry in evictable:
            victim = await self._adapter.load_cost(entry.id, entry.context_length)
            if victim is not None:
                credit += victim.weights_mb + victim.overhead_mb
        if credit < shortfall or not allow_evict:
            note("queue", shortfall_mb=shortfall, evictable_credit_mb=credit)
            return None
        note("evict", shortfall_mb=shortfall, evictable_credit_mb=credit)
        return ("evict", shortfall, priority)

    async def _inflight_reserve_mb(
        self, entries: list[LoadedModel], requested_ctx: int | None
    ) -> tuple[int, tuple[str, ...]]:
        """KV a busy resident model may still grow to at its loaded context.

        Live headroom already excludes KV a busy model holds now, so reserving
        its full KV double-counts that part; the error is on the safe side.
        """
        total = 0
        estimate: list[str] = []
        for entry in entries:
            if not self._is_busy(entry):
                continue
            ctx = entry.context_length
            if ctx is None:
                ctx = requested_ctx
                estimate.append(f"inflight-ctx-unknown:{entry.id}")
            cost = await self._adapter.load_cost(entry.id, ctx)
            if cost is None:
                estimate.append(f"inflight-cost-unknown:{entry.id}")
                continue
            total += cost.kv_mb
            estimate.extend(f"inflight:{entry.id}:{item}" for item in cost.estimate)
        return total, tuple(estimate)

    async def _pending_load_mb(self, resident: list[str]) -> tuple[int, tuple[str, ...]]:
        """Full cost of admitted loads that are not resident yet."""
        total = 0
        estimate: list[str] = []
        for model_id in self._reserved:
            if any(self._same_model(model_id, current) for current in resident):
                continue
            cost: LoadCost | None = await self._adapter.load_cost(
                model_id, self._reserved_ctx.get(model_id)
            )
            if cost is None:
                estimate.append(f"pending-cost-unknown:{model_id}")
                continue
            total += cost.total_mb
            estimate.extend(f"pending:{model_id}:{item}" for item in cost.estimate)
        return total, tuple(estimate)

    def _is_busy(self, entry: LoadedModel) -> bool:
        if self._reaper.in_flight(entry.id) > 0:
            return True
        if self._reaper.in_flight(self._adapter.base_id(entry.id)) > 0:
            return True
        if self._is_reserved(entry.id):
            return True
        if entry.queued:
            return True
        # Unknown status is treated as busy: an external tenant may be generating.
        return entry.status != "idle"

    def _reserve(self, model_id: str, context_length: int | None = None) -> None:
        self._reserved[model_id] = self._reserved.get(model_id, 0) + 1
        if context_length is not None:
            self._reserved_ctx[model_id] = max(
                context_length, self._reserved_ctx.get(model_id) or 0
            )

    def _active_calls(self) -> int:
        return sum(self._reserved.values())

    def _is_reserved(self, model_id: str) -> bool:
        return any(self._same_model(model_id, reserved) for reserved in self._reserved)

    def _is_committed(self, model_id: str, resident: list[str]) -> bool:
        if any(self._same_model(model_id, current) for current in resident):
            return True
        return self._is_reserved(model_id)

    async def _committed_mb(self, resident: list[str]) -> int:
        total = 0
        for model_id in resident:
            footprint = await self._footprint_mb(model_id)
            if footprint is None:
                total += await self._budget_mb()
            else:
                total += footprint
        for reserved in self._reserved:
            if not any(self._same_model(reserved, current) for current in resident):
                total += (await self._footprint_mb(reserved)) or 0
        return total

    def _same_model(self, a: str, b: str) -> bool:
        return self._adapter.base_id(a) == self._adapter.base_id(b)

    async def _budget_mb(self) -> int:
        if self._budget_override is not None:
            return self._budget_override
        if self._budget_cache is None:
            wired_limit = await self._wired_limit_mb()
            self._budget_cache = max(0, wired_limit - FLAT_MARGIN_MB)
        return self._budget_cache

    async def _wired_limit_mb(self) -> int:
        try:
            proc = await asyncio.create_subprocess_exec(
                "sysctl",
                "-n",
                "iogpu.wired_limit_mb",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            text = (out or b"").decode().strip()
            if proc.returncode == 0 and text.isdigit():
                value = int(text)
                if value > 0:
                    return value
                # sysctl reports 0 when the wired limit is system-managed, not an
                # actual 0 MB budget. Fall through to the default so admission is
                # not starved to total backpressure.
                log.info(
                    "laplace.broker: iogpu.wired_limit_mb=0 (system-managed); "
                    "using default budget",
                )
            else:
                log.warning(
                    "laplace.broker: sysctl iogpu.wired_limit_mb failed (rc=%s): %s",
                    proc.returncode,
                    ((err or out) or b"").decode().strip()[:160],
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("laplace.broker: sysctl read failed: %s", exc)
        return 90112

    async def _resident(self) -> list[str]:
        return [entry.id for entry in await self._adapter.list_loaded() if entry.id]

    async def _footprint_mb(self, model_id: str) -> int | None:
        if model_id in self._footprint_cache:
            return self._footprint_cache[model_id]
        footprint = await self._adapter.footprint_mb(model_id)
        self._footprint_cache[model_id] = footprint
        return footprint
