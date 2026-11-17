"""Per-request admission lifecycle (B2 + r3 first-touch amendment)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from laplace.adapter import InferenceAdapter, LoadGuardActive, ModelLoadError
from laplace.broker import ABANDON_POLL_S, ContentionBroker
from laplace.reaper import Reaper

log = logging.getLogger(__name__)

# Retry just after the guard window ends, not on its last instant. A wake that
# is still early only meets the guard again and waits out the remainder.
GUARD_RETRY_MARGIN_S = 0.05
# Indirection so tests can drive the wait without real seconds passing.
_sleep = asyncio.sleep


@dataclass(frozen=True)
class Admission:
    """Outcome of the admission decision for one request.

    ``decision`` is one of admit, backpressure, fail-open, or abandoned.
    ``proxy`` is False for backpressure and abandoned (the client is gone);
    admit and fail-open both forward upstream (fail-open forwards ungoverned).
    """

    decision: str
    wait_ms: int

    @property
    def proxy(self) -> bool:
        return self.decision not in ("backpressure", "abandoned")


@asynccontextmanager
async def govern(
    broker: ContentionBroker,
    reaper: Reaper,
    adapter: InferenceAdapter,
    *,
    model: str | None,
    ctx: int | None,
    tier: str,
    fail_open: bool,
    need_ctx: int | None = None,
    abandoned: Callable[[], bool] | None = None,
) -> AsyncIterator[Admission]:
    """Govern one request across its lifecycle.

    Order (per brokered_call.py): admit -> mark_managed -> ensure_loaded ->
    acquire -> [caller proxies] -> release -> done. The r3 amendment adds
    first-touch capture with unmark-on-failure so a failed pre-acquire request
    never permanently captures an externally-resident model.

    Errors raised by the caller's governed body are propagated unchanged; only
    daemon-internal failures (broker/reaper/adapter) are subject to fail-open.

    ``need_ctx`` (fit-sized context admission): the job's estimated context
    need. The broker then picks the context, and the load uses that context.

    ``abandoned`` (cancel_admission_on_disconnect): True once the client is
    gone. The admission wait then ends without a load or eviction
    (``decision="abandoned"``). It is also checked once after admission,
    before the load starts; a load already running is not interrupted.

    A load refused only by a timed guard window (``LoadGuardActive``) waits
    the window out and is admitted again from the start: the reservation and
    first-touch capture are released first, no broker lock is held, and the
    wait polls ``abandoned``. Admission, guard waits and re-admission share
    the broker's one admission timeout; a window that would outlast it fails
    at once, as before. Every other ``ModelLoadError`` still fails.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()

    def waited() -> int:
        return int((loop.time() - start) * 1000)

    if not model:
        log.warning("admission: request has no model; forwarding ungoverned")
        yield Admission("fail-open", 0)
        return

    configured_ctx = ctx
    # Empty on the first pass: the broker call shape is unchanged until a
    # guard wait needs the remaining overall budget passed on.
    budget: dict[str, float] = {}
    while True:
        ctx = configured_ctx
        try:
            if abandoned is None:
                # Unchanged call shape when the flag is off.
                if need_ctx is None:
                    decision = await broker.admit(model, ctx, tier, **budget)
                else:
                    decision, ctx = await broker.admit_sized(
                        model, ctx, tier, need_context=need_ctx, **budget
                    )
            elif need_ctx is None:
                decision = await broker.admit(model, ctx, tier, abandoned=abandoned, **budget)
            else:
                decision, ctx = await broker.admit_sized(
                    model, ctx, tier, need_context=need_ctx, abandoned=abandoned, **budget
                )
        except Exception as exc:  # noqa: BLE001 - broker failure is daemon-internal
            if not fail_open:
                raise
            log.warning("admission: broker error, forwarding ungoverned: %s", exc)
            yield Admission("fail-open", waited())
            return

        if decision == "abandoned":
            yield Admission("abandoned", waited())
            return

        if decision != "admit":
            yield Admission("backpressure", waited())
            return

        if abandoned is not None and abandoned():
            # Admitted just as the client left: release before any load.
            await broker.done(model)
            yield Admission("abandoned", waited())
            return

        # Admitted: a reservation is now held and must be released via broker.done.
        first_touch = not reaper.is_managed(model)
        acquired = False
        try:
            if first_touch:
                reaper.mark_managed(model)
            reaper.start_loading(model)
            try:
                # The pin above stops new reaper unloads of this model. One that
                # began earlier must finish first, or we would proxy to (or load
                # into) a model the reaper is still removing.
                await reaper.wait_unloaded(model)
                await adapter.ensure_loaded(model, ctx)
            finally:
                reaper.finish_loading(model)
            reaper.acquire(model)
            acquired = True
        except BaseException as exc:  # noqa: BLE001 - covers cancellation too
            # Restore external status only if we took first touch, never acquired,
            # and no other in-flight call holds the model.
            if first_touch and not acquired and reaper.in_flight(model) == 0:
                reaper.unmark_managed(model)
            await broker.done(model)
            if fail_open and isinstance(exc, Exception) and not isinstance(exc, ModelLoadError):
                log.warning("admission: load/acquire error, forwarding ungoverned: %s", exc)
                yield Admission("fail-open", waited())
                return
            if not isinstance(exc, LoadGuardActive):
                raise
            outcome = await _wait_out_guard(
                exc, start + broker.timeout_s - loop.time(), abandoned, model
            )
            if outcome == "give-up":
                raise
        else:
            break
        if outcome == "abandoned":
            yield Admission("abandoned", waited())
            return
        budget = {"timeout_s": max(0.0, start + broker.timeout_s - loop.time())}

    try:
        yield Admission("admit", waited())
    finally:
        reaper.release(model)
        await broker.done(model)


async def _wait_out_guard(
    exc: LoadGuardActive,
    budget_s: float,
    abandoned: Callable[[], bool] | None,
    model: str,
) -> str:
    """Wait until a timed load guard ends, holding no reservation or lock.

    Returns ``retry`` once the window is over, ``abandoned`` when the client
    leaves first, and ``give-up`` when the window outlasts ``budget_s`` (the
    admission timeout left): then nothing is waited and the caller fails.
    """
    wait_s = max(0.0, exc.retry_after_s) + GUARD_RETRY_MARGIN_S
    if wait_s > budget_s:
        log.info(
            "admission: model=%s decision=load-guard-fail guard_remaining_s=%.1f "
            "admission_budget_s=%.1f",
            model,
            exc.retry_after_s,
            budget_s,
        )
        return "give-up"
    log.info(
        "admission: model=%s decision=queue reason=load-guard-window wait_s=%.1f",
        model,
        wait_s,
    )
    left = wait_s
    while left > 0:
        if abandoned is not None and abandoned():
            log.info("admission: model=%s decision=abandoned stage=load-guard-wait", model)
            return "abandoned"
        step = left if abandoned is None else min(ABANDON_POLL_S, left)
        await _sleep(step)
        left -= step
    return "retry"
