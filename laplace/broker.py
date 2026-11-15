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
# How often a parked admission checks ``abandoned`` (same cadence as the proxy's
# disconnect watch, laplace_serve/proxy.py DISCONNECT_POLL_S).
ABANDON_POLL_S = 1.0
# best_effort replace: unload an idle smaller copy only when the context the
# capacity cap predicts after the unload costs at least this much more than the
# copy. Covers the headroom drift measured between turns (largest drop 1738 MB
# in the GLM canary, 2026-11-15 03:32) so a replace never reloads smaller, and
# skips ~18 s reloads (3 s unload + 15 s load) for small gains.
REPLACE_MIN_GAIN_MB = 2048


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
        fit_context: bool = False,
        measured_eviction: bool = False,
        best_effort: bool = False,
    ):
        """``headroom_mb`` enables live-memory admission: a cold load must fit
        in the live headroom it returns, after reserving KV growth for busy
        resident models and the full cost of admitted loads still pending.
        Without it, admission uses the static footprint budget.

        Both flags below apply only to live-memory admission.
        ``fit_context``: when the configured context does not fit, load at the
        largest context that does, if that still covers the job's need (see
        ``admit_sized``). ``measured_eviction``: once per admission, evict
        idle victims even when their estimated credit is below the shortfall,
        then decide again on the measured headroom. ``best_effort`` (needs
        ``fit_context``): when the need costs more than could ever be freed,
        cap it at the largest achievable context; and replace an idle resident
        copy smaller than the need at once instead of waiting for the reaper."""
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
        self._fit_context = fit_context
        self._measured_eviction = measured_eviction
        self._best_effort = best_effort
        self._reaper.set_broker(self)

    async def admit(
        self,
        model_id: str,
        context_length: int | None = None,
        priority: str = "scheduled",
        *,
        abandoned: Callable[[], bool] | None = None,
    ) -> str:
        decision, _ = await self.admit_sized(
            model_id, context_length, priority, abandoned=abandoned
        )
        return decision

    async def admit_sized(
        self,
        model_id: str,
        context_length: int | None = None,
        priority: str = "scheduled",
        *,
        need_context: int | None = None,
        abandoned: Callable[[], bool] | None = None,
    ) -> tuple[str, int | None]:
        """Admit like ``admit`` and also return the context to load or reuse at.

        ``abandoned`` (optional): returns True once the caller no longer wants
        the result, e.g. its client disconnected. It is checked before each
        evaluation, before any eviction or replace unload, and every
        ABANDON_POLL_S while parked; when True the wait ends with
        ``("abandoned", None)``, no reservation is held, and no eviction or
        replace is started. An eviction or replace already running is never
        interrupted.

        ``context_length`` is the configured context. ``need_context`` is the
        job's own estimate (prompt plus output allowance). With ``fit_context``
        on and a need given, a load that does not fit at the configured
        context is admitted at the largest context that fits, if that covers
        the need. Without them the returned context is ``context_length``.
        """
        if not model_id:
            return "admit", context_length
        if not (self._fit_context and need_context and context_length):
            need_context = None
        elif need_context > context_length:
            # More than configured: today's behaviour, the configured context.
            need_context = context_length
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
        # After a measured (credit < shortfall) eviction this admission never
        # evicts again, by either branch, so it cannot evict, fail to fit, and
        # evict again on a later wake. Never set when measured_eviction is off.
        measured_eviction_used = False
        # best_effort: replace a smaller resident copy at most once per
        # admission, so a failed unload cannot loop.
        replace_used = False

        while True:
            replace_id = None
            if abandoned is not None and abandoned():
                return self._abandon(model_id, "before-evaluation")
            async with self._lock:
                decision = await self._try_admit(
                    model_id,
                    tier,
                    allow_evict=not (evicted_this_cycle or measured_eviction_used),
                    context_length=context_length,
                    need_context=need_context,
                    allow_measured_eviction=(
                        self._measured_eviction and not measured_eviction_used
                    ),
                    allow_replace=self._best_effort and not replace_used,
                )
                queued_for_unload = decision is None and self._reaper.is_unloading(model_id)
                if decision == "admit":
                    self._wake()
                    return "admit", context_length
                if isinstance(decision, tuple) and decision[0] == "admit":
                    self._wake()
                    return "admit", decision[1]
                if isinstance(decision, tuple) and decision[0] == "evict":
                    _, needed_mb, evict_prio = decision[:3]
                    if len(decision) > 3 and decision[3]:
                        measured_eviction_used = True
                else:
                    needed_mb = None
                    evict_prio = None
                if isinstance(decision, tuple) and decision[0] == "replace":
                    replace_id = decision[1]
                    replace_used = True
            if replace_id is not None:
                if abandoned is not None and abandoned():
                    return self._abandon(model_id, "before-replace")
                await self._reaper.unload_for_replace(replace_id)
                continue
            if needed_mb is not None:
                if abandoned is not None and abandoned():
                    return self._abandon(model_id, "before-eviction")
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
                    return "backpressure", None
                if queued_for_unload and not self._reaper.is_unloading(model_id):
                    # The unload ended before we parked, so its wake found no
                    # waiter. Re-evaluate now instead of waiting for another.
                    continue
                fut = loop.create_future()
                self._counter += 1
                entry = (int(tier), self._counter, fut)
                heapq.heappush(self._waiters, entry)
                if self._waiters and self._waiters[0] is not entry:
                    self._wake()

            try:
                if abandoned is None:
                    await asyncio.wait_for(fut, timeout=remaining)
                elif await self._park_or_abandon(fut, remaining, abandoned):
                    async with self._lock:
                        self._remove_waiter(entry)
                    return self._abandon(model_id, "while-queued")
            except asyncio.TimeoutError:
                async with self._lock:
                    self._remove_waiter(entry)
                return "backpressure", None
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
                            if abandoned is not None:
                                # asyncio.wait does not cancel fut: drop it, and
                                # pass on a wake that already reached it.
                                if fut.done():
                                    self._wake()
                                else:
                                    fut.cancel()
                        break
                    except asyncio.CancelledError:
                        continue
                raise
            evicted_this_cycle = False

    @staticmethod
    async def _park_or_abandon(
        fut: asyncio.Future, remaining: float, abandoned: Callable[[], bool]
    ) -> bool:
        """Wait for a wake like ``wait_for(fut, remaining)``, polling ``abandoned``.

        True when abandoned first; raises TimeoutError when ``remaining`` runs out.
        """
        loop = asyncio.get_running_loop()
        end = loop.time() + remaining
        while not fut.done():
            if abandoned():
                return True
            left = end - loop.time()
            if left <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait({fut}, timeout=min(ABANDON_POLL_S, left))
        return False

    def _abandon(self, model_id: str, stage: str) -> tuple[str, None]:
        log.info(
            "laplace.broker: load decision model=%s decision=abandoned "
            "reason=caller-gone stage=%s",
            model_id,
            stage,
        )
        # This admission may hold a wake it will not use: pass one on. A spare
        # wake only makes the head waiter evaluate once more.
        self._wake()
        return "abandoned", None

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

    def wake_waiter(self) -> None:
        """Wake the head waiter from synchronous code (the reaper's unload exit).

        The heap is only mutated in synchronous stretches, so this needs no lock.
        """
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
        need_context: int | None = None,
        allow_measured_eviction: bool = False,
        allow_replace: bool = False,
    ):
        if self._headroom is not None and hasattr(self._adapter, "load_cost"):
            return await self._try_admit_live(
                model_id,
                priority,
                allow_evict,
                context_length,
                need_context,
                allow_measured_eviction,
                allow_replace,
            )
        footprint = await self._footprint_mb(model_id)
        if self._unload_in_progress(model_id, context_length):
            return None
        if footprint is None:
            self._reserve(model_id)
            return "admit"
        if self._active_calls() >= self._max_concurrent:
            return None

        budget = await self._budget_mb()
        resident = await self._resident()
        if self._unload_in_progress(model_id, context_length):
            return None

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
        need_context: int | None = None,
        allow_measured_eviction: bool = False,
        allow_replace: bool = False,
    ):
        if self._active_calls() >= self._max_concurrent:
            return None
        entries = [entry for entry in await self._adapter.list_loaded() if entry.id]
        resident = [entry.id for entry in entries]
        if self._unload_in_progress(model_id, context_length):
            return None
        # With fit-sized context an instance only has to cover the job's need.
        required_ctx = need_context or context_length
        covered = self._covered_by_existing(model_id, required_ctx, entries)
        capped_from = None
        # One live reading per pass: the cap is sized to fill this snapshot
        # exactly, so a second vm_stat a few ms later can read 1 MB less and
        # refuse the capped load with nothing resident (canary 2026-11-15).
        snapshot = None
        if covered != "admit" and self._best_effort and need_context:
            snapshot = await self._live_available(entries, resident, context_length)
            cap = await self._capacity_cap(model_id, need_context, entries, snapshot)
            if cap is not None:
                log.info(
                    "laplace.broker: load decision model=%s need_ctx=%s capped_ctx=%s "
                    "reason=need-exceeds-capacity",
                    model_id,
                    need_context,
                    cap,
                )
                capped_from, need_context = need_context, cap
                required_ctx = cap
                covered = self._covered_by_existing(model_id, required_ctx, entries)
        if covered == "admit":
            # No new load: an instance at this context is resident or pending.
            if need_context and not any(
                self._same_model(model_id, entry.id) for entry in entries
            ):
                # Riding on a pending load: whichever admitted request reaches
                # the loader first must load at the pending load's context, or
                # the larger request finds a too-small instance.
                required_ctx = max(
                    self._reserved_ctx.get(reserved) or 0
                    for reserved in self._reserved
                    if self._same_model(model_id, reserved)
                )
            self._reserve(model_id, required_ctx)
            if not need_context:
                return "admit"
            log.info(
                "laplace.broker: load decision model=%s need_ctx=%s configured_ctx=%s "
                "decision=admit reason=%s",
                model_id,
                capped_from or need_context,
                context_length,
                "need-capped-to-capacity" if capped_from else "covered-by-existing-instance",
            )
            return ("admit", required_ctx)
        if covered == "wait":
            log.info(
                "laplace.broker: load decision model=%s ctx=%s decision=queue "
                "reason=pending-load-at-smaller-context",
                model_id,
                required_ctx,
            )
            return None
        if need_context and allow_replace:
            victim = self._idle_smaller_copy(model_id, entries)
            if victim is not None and not await self._replace_gains(
                model_id, need_context, victim
            ):
                # Reloading would gain less than the headroom drift between
                # turns, or could come back smaller: keep the copy.
                self._reserve(model_id, victim.context_length)
                log.info(
                    "laplace.broker: load decision model=%s need_ctx=%s configured_ctx=%s "
                    "decision=admit reason=replace-gain-below-threshold resident_ctx=%s",
                    model_id,
                    capped_from or need_context,
                    context_length,
                    victim.context_length,
                )
                return ("admit", victim.context_length)
            if victim is not None:
                log.info(
                    "laplace.broker: load decision model=%s need_ctx=%s decision=replace "
                    "reason=idle-resident-at-smaller-context resident=%s resident_ctx=%s",
                    model_id,
                    need_context,
                    victim.id,
                    victim.context_length,
                )
                return ("replace", victim.id)
        if need_context and any(self._same_model(model_id, entry.id) for entry in entries):
            # A resident instance below the need cannot serve this job, and a
            # second instance would be refused as a duplicate load. Wait for
            # the reaper's idle unload; it wakes this queue.
            log.info(
                "laplace.broker: load decision model=%s need_ctx=%s decision=queue "
                "reason=resident-at-smaller-context resident_ctx=%s",
                model_id,
                need_context,
                [entry.context_length for entry in entries if self._same_model(model_id, entry.id)],
            )
            return None

        cost = await self._adapter.load_cost(model_id, context_length)
        if cost is None:
            log.warning(
                "laplace.broker: load decision model=%s ctx=%s decision=queue "
                "reason=cost-unknown (no catalog size or footprint table entry)",
                model_id,
                context_length,
            )
            return None
        if snapshot is None:
            snapshot = await self._live_available(entries, resident, context_length)
        if snapshot is None:
            log.warning(
                "laplace.broker: load decision model=%s cost_mb=%s decision=queue "
                "reason=headroom-unreadable",
                model_id,
                cost.total_mb,
            )
            return None
        headroom, inflight_mb, inflight_estimate, pending_mb, pending_estimate = snapshot
        available = headroom - inflight_mb - pending_mb
        estimate = (*cost.estimate, *inflight_estimate, *pending_estimate)

        def note(decision: str, **extra) -> None:
            if need_context:
                extra = {"configured_ctx": context_length, "need_ctx": need_context, **extra}
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
            return ("admit", context_length) if need_context else "admit"

        if need_context:
            fit_ctx = await self._fit_context_length(
                model_id, available, need_context, context_length
            )
            if fit_ctx is not None:
                cost = await self._adapter.load_cost(model_id, fit_ctx)
                note("admit", fit_ctx=fit_ctx)
                self._reserve(model_id, fit_ctx)
                return ("admit", fit_ctx)
            # Evict for the job's need only; the recheck after eviction sizes
            # the load to whatever the measured headroom then allows.
            need_cost = await self._adapter.load_cost(model_id, need_context)
            if need_cost is not None:
                cost = need_cost

        shortfall = cost.total_mb - available
        evictable_ids = set(self._reaper.evictable_models(resident))
        evictable = [
            entry for entry in entries
            if entry.id in evictable_ids and not self._is_busy(entry)
        ]
        # Credit idle victims with weights + overhead only: KV they no longer
        # hold is not counted, so eviction is chosen only when it surely helps.
        credit = 0
        for entry in evictable:
            victim = await self._adapter.load_cost(entry.id, entry.context_length)
            if victim is not None:
                credit += victim.weights_mb + victim.overhead_mb
        if not allow_evict:
            note("queue", shortfall_mb=shortfall, evictable_credit_mb=credit)
            return None
        if credit >= shortfall:
            note("evict", shortfall_mb=shortfall, evictable_credit_mb=credit)
            return ("evict", shortfall, priority)
        if allow_measured_eviction and evictable:
            # The credit omits KV and allocator memory a victim releases, so
            # measure instead of predicting: evict once, then decide again on
            # live headroom. Same victims as above: idle, managed, unreserved.
            note("evict-measured", shortfall_mb=shortfall, evictable_credit_mb=credit)
            return ("evict", shortfall, priority, True)
        note("queue", shortfall_mb=shortfall, evictable_credit_mb=credit)
        return None

    def _idle_smaller_copy(
        self, model_id: str, entries: list[LoadedModel]
    ) -> LoadedModel | None:
        """The one resident copy of this model, if idle and safe to unload now.

        Only reached when that copy is below the need (_covered_by_existing
        returned None). Same victim rules as eviction: managed, not loading,
        not keep_loaded, nothing in flight or reserved, runtime idle.
        """
        matches = [entry for entry in entries if self._same_model(model_id, entry.id)]
        if len(matches) != 1:
            return None
        copy = matches[0]
        if self._is_busy(copy) or self._reaper.evictable_models([copy.id]) != [copy.id]:
            return None
        return copy

    async def _replace_gains(
        self, model_id: str, target_ctx: int, copy: LoadedModel
    ) -> bool:
        """Whether reloading ``copy`` at ``target_ctx`` gains REPLACE_MIN_GAIN_MB.

        ``target_ctx`` is the need, already capped by ``_capacity_cap``. True
        (replace, as before) when the copy's context or either cost is unknown.
        """
        if copy.context_length is None:
            return True
        target = await self._adapter.load_cost(model_id, target_ctx)
        current = await self._adapter.load_cost(model_id, copy.context_length)
        if target is None or current is None:
            return True
        return target.total_mb - current.total_mb >= REPLACE_MIN_GAIN_MB

    async def _live_available(
        self, entries: list[LoadedModel], resident: list[str], context_length: int | None
    ) -> tuple[int, int, tuple[str, ...], int, tuple[str, ...]] | None:
        """One live headroom reading with the in-flight and pending reserves.

        None when headroom is unreadable.
        """
        headroom = await self._headroom()
        if headroom is None:
            return None
        inflight_mb, inflight_estimate = await self._inflight_reserve_mb(entries, context_length)
        pending_mb, pending_estimate = await self._pending_load_mb(resident)
        return headroom, inflight_mb, inflight_estimate, pending_mb, pending_estimate

    async def _capacity_cap(
        self,
        model_id: str,
        need: int,
        entries: list[LoadedModel],
        snapshot: tuple[int, int, tuple[str, ...], int, tuple[str, ...]] | None,
    ) -> int | None:
        """Largest context below ``need`` that can ever fit, when ``need`` cannot.

        Capacity is available headroom (after in-flight and pending reserves),
        taken from ``snapshot`` (the reading admission then decides on),
        plus the weights + overhead of a resident copy of this model and of
        idle evictable models: what eviction or replacement surely frees now.
        KV is not credited, as for eviction: a copy's unload measured 6.3-8.6 GB
        less than its full cost (GLM canary 2026-11-15 03:18, 03:21), and
        crediting it reloaded the copy at a smaller context (03:32).

        A resident copy is achievable as it is, so the cap is never below its
        context. None when the need fits capacity, or any term is unknown
        (then today's behaviour: queue).
        """
        need_cost = await self._adapter.load_cost(model_id, need)
        if need_cost is None or snapshot is None:
            return None
        headroom, inflight_mb, _, pending_mb, _ = snapshot
        resident = [entry.id for entry in entries]
        capacity = headroom - inflight_mb - pending_mb
        evictable_ids = set(self._reaper.evictable_models(resident))
        copy_ctx = 0
        for entry in entries:
            if self._same_model(model_id, entry.id):
                cost = await self._adapter.load_cost(entry.id, entry.context_length)
                if cost is None:
                    return None
                capacity += cost.weights_mb + cost.overhead_mb
                copy_ctx = max(copy_ctx, entry.context_length or 0)
            elif entry.id in evictable_ids and not self._is_busy(entry):
                cost = await self._adapter.load_cost(entry.id, entry.context_length)
                if cost is not None:
                    capacity += cost.weights_mb + cost.overhead_mb
        if need_cost.total_mb <= capacity:
            return None
        low_cost = await self._adapter.load_cost(model_id, 1)
        if low_cost is None or low_cost.total_mb > capacity:
            return copy_ctx if 0 < copy_ctx < need else None
        low, high = 1, need - 1
        while low < high:
            mid = (low + high + 1) // 2
            mid_cost = await self._adapter.load_cost(model_id, mid)
            if mid_cost is not None and mid_cost.total_mb <= capacity:
                low = mid
            else:
                high = mid - 1
        return max(low, copy_ctx) if copy_ctx < need else low

    async def _fit_context_length(
        self, model_id: str, available_mb: int, need: int, ceiling: int
    ) -> int | None:
        """Largest context in [need, ceiling] whose load cost fits ``available_mb``.

        Bisects on the adapter's own ``load_cost``, so sliding-window layers
        and parallel are priced exactly as at load time. Cost never falls as
        context grows. None when even ``need`` does not fit.
        """
        low_cost = await self._adapter.load_cost(model_id, need)
        if low_cost is None or low_cost.total_mb > available_mb:
            return None
        low, high = need, ceiling
        while low < high:
            mid = (low + high + 1) // 2
            mid_cost = await self._adapter.load_cost(model_id, mid)
            if mid_cost is not None and mid_cost.total_mb <= available_mb:
                low = mid
            else:
                high = mid - 1
        return low

    def _unload_in_progress(self, model_id: str, context_length: int | None) -> bool:
        """Queue a request whose model the reaper is unloading right now.

        The model can still be listed as resident, so admitting would ride on
        an instance that is leaving. Checked after the last residency await so
        an unload that began during that await is seen.
        """
        if not self._reaper.is_unloading(model_id):
            return False
        log.info(
            "laplace.broker: load decision model=%s ctx=%s decision=queue "
            "reason=unload-in-progress",
            model_id,
            context_length,
        )
        return True

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
        return Reaper.reported_busy(entry)

    def _covered_by_existing(
        self, model_id: str, context_length: int | None, entries: list[LoadedModel]
    ) -> str | None:
        """Whether a resident or pending instance already pays for this request.

        ``admit``: one resident instance, or a pending load, at a context at
        least as large. ``wait``: a pending load at a smaller or unknown
        context, which this request must not ride on. ``None``: decide on cost.
        Parallel is fixed per model by [model_parallel], so context is the only
        per-request dimension.
        """
        matches = [entry for entry in entries if self._same_model(model_id, entry.id)]
        if matches:
            if len(matches) == 1 and (
                context_length is None
                or (matches[0].context_length or 0) >= context_length
            ):
                return "admit"
            return None
        pending = [
            reserved for reserved in self._reserved if self._same_model(model_id, reserved)
        ]
        if not pending:
            return None
        if context_length is None or any(
            (self._reserved_ctx.get(reserved) or 0) >= context_length for reserved in pending
        ):
            return "admit"
        return "wait"

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
