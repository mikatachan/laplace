"""Priority-aware admission gate for a shared inference pool."""

from __future__ import annotations

import asyncio
import heapq
import logging

from laplace.adapter import InferenceAdapter
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
    ):
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
            evicted_this_cycle = False

    async def done(self, model_id: str) -> None:
        if not model_id:
            return
        async with self._lock:
            count = self._reserved.get(model_id, 0) - 1
            if count > 0:
                self._reserved[model_id] = count
            else:
                self._reserved.pop(model_id, None)
            self._wake()

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
    ):
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

    def _reserve(self, model_id: str) -> None:
        self._reserved[model_id] = self._reserved.get(model_id, 0) + 1

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
                return int(text)
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
