"""Per-request admission lifecycle (B2 + r3 first-touch amendment)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from laplace.adapter import InferenceAdapter
from laplace.broker import ContentionBroker
from laplace.reaper import Reaper

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Admission:
    """Outcome of the admission decision for one request.

    ``decision`` is one of admit, backpressure, or fail-open. ``proxy`` is False
    only for backpressure; admit and fail-open both forward upstream (fail-open
    forwards ungoverned).
    """

    decision: str
    wait_ms: int

    @property
    def proxy(self) -> bool:
        return self.decision != "backpressure"


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
) -> AsyncIterator[Admission]:
    """Govern one request across its lifecycle.

    Order (per brokered_call.py): admit -> mark_managed -> ensure_loaded ->
    acquire -> [caller proxies] -> release -> done. The r3 amendment adds
    first-touch capture with unmark-on-failure so a failed pre-acquire request
    never permanently captures an externally-resident model.

    Errors raised by the caller's governed body are propagated unchanged; only
    daemon-internal failures (broker/reaper/adapter) are subject to fail-open.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()

    def waited() -> int:
        return int((loop.time() - start) * 1000)

    if not model:
        log.warning("admission: request has no model; forwarding ungoverned")
        yield Admission("fail-open", 0)
        return

    try:
        decision = await broker.admit(model, ctx, tier)
    except Exception as exc:  # noqa: BLE001 - broker failure is daemon-internal
        if not fail_open:
            raise
        log.warning("admission: broker error, forwarding ungoverned: %s", exc)
        yield Admission("fail-open", waited())
        return

    if decision != "admit":
        yield Admission("backpressure", waited())
        return

    # Admitted: a reservation is now held and must be released via broker.done.
    first_touch = not reaper.is_managed(model)
    acquired = False
    try:
        if first_touch:
            reaper.mark_managed(model)
        await adapter.ensure_loaded(model, ctx)
        reaper.acquire(model)
        acquired = True
    except BaseException as exc:  # noqa: BLE001 - covers cancellation too
        # Restore external status only if we took first touch, never acquired,
        # and no other in-flight call holds the model.
        if first_touch and not acquired and reaper.in_flight(model) == 0:
            reaper.unmark_managed(model)
        await broker.done(model)
        if fail_open and isinstance(exc, Exception):
            log.warning("admission: load/acquire error, forwarding ungoverned: %s", exc)
            yield Admission("fail-open", waited())
            return
        raise

    try:
        yield Admission("admit", waited())
    finally:
        reaper.release(model)
        await broker.done(model)
