from __future__ import annotations

import asyncio
import logging

import pytest

from laplace.adapter import LoadCost
from laplace.broker import ContentionBroker
from laplace.priority import PriorityTier
from laplace.reaper import Reaper
from tests.fakes import FakeAdapter, loaded_model


MODEL = "org/model"
CTX = 131072


class Headroom:
    def __init__(self, value: int, shadow: int | None = None):
        self.value = value
        self.shadow_headroom_mb = shadow

    async def __call__(self):
        return self.value


class DynamicAdapter(FakeAdapter):
    def __init__(self, *, configured_parallel: int = 2):
        super().__init__()
        self.configured_parallel = configured_parallel
        self.cost_calls: list[int | None] = []

    async def load_cost(self, model_id, context_length, parallel=None):
        self.cost_calls.append(parallel)
        selected = self.configured_parallel if parallel is None else parallel
        return LoadCost(100, 100 * selected, 10, context_length, selected)


def broker(adapter, headroom, *, dynamic=True, timeout=0.01):
    return ContentionBroker(
        adapter,
        Reaper(adapter),
        headroom_mb=headroom,
        dynamic_load_sizing=dynamic,
        parallel_ceilings={MODEL: 2},
        max_concurrent_lms=8,
        timeout_s=timeout,
    )


def add_waiter(target: ContentionBroker, model=MODEL):
    future = asyncio.get_running_loop().create_future()
    target._waiters.append((int(PriorityTier.SCHEDULED), 1, model, future))
    return future


@pytest.mark.asyncio
async def test_flag_off_uses_configured_parallel_as_before():
    adapter = DynamicAdapter(configured_parallel=2)
    target = broker(adapter, Headroom(1000), dynamic=False)

    assert await target.admit(MODEL, CTX) == "admit"
    assert adapter.cost_calls == [None]
    assert target._reserved_parallel == {}


@pytest.mark.asyncio
async def test_demand_one_selects_parallel_one(caplog):
    adapter = DynamicAdapter()
    target = broker(adapter, Headroom(1000, shadow=9000))

    with caplog.at_level(logging.INFO, logger="laplace.broker"):
        assert await target.admit(MODEL, CTX, return_parallel=True) == ("admit", 1)

    assert "demand=1 chosen_parallel=1" in caplog.text
    assert "shadow_headroom_mb=9000" in caplog.text


@pytest.mark.asyncio
async def test_demand_two_with_room_selects_parallel_two():
    adapter = DynamicAdapter()
    target = broker(adapter, Headroom(1000))
    waiting = add_waiter(target)

    assert await target.admit(MODEL, CTX, return_parallel=True) == ("admit", 2)
    waiting.cancel()


@pytest.mark.asyncio
async def test_demand_two_falls_back_to_parallel_one_when_two_does_not_fit():
    adapter = DynamicAdapter()
    target = broker(adapter, Headroom(250))  # P2 costs 310; P1 costs 210.
    waiting = add_waiter(target)

    assert await target.admit(MODEL, CTX, return_parallel=True) == ("admit", 1)
    waiting.cancel()


@pytest.mark.asyncio
async def test_nothing_fits_backpressures():
    adapter = DynamicAdapter()
    target = broker(adapter, Headroom(200))

    assert await target.admit(MODEL, CTX, return_parallel=True) == (
        "backpressure", None
    )


@pytest.mark.asyncio
async def test_pending_reserve_uses_chosen_parallel():
    adapter = DynamicAdapter()
    target = broker(adapter, Headroom(1000))

    assert await target.admit(MODEL, CTX, return_parallel=True) == ("admit", 1)
    pending, _ = await target._pending_load_mb([])

    assert pending == 210
    assert adapter.cost_calls[-1] == 1


@pytest.mark.asyncio
async def test_busy_kv_reserve_uses_actual_loaded_parallel():
    adapter = DynamicAdapter(configured_parallel=2)
    entry = loaded_model(
        MODEL,
        status="generating",
        context_length=CTX,
        parallel=1,
    )
    target = broker(adapter, Headroom(1000))

    reserve, _ = await target._inflight_reserve_mb([entry], CTX)

    assert reserve == 100
    assert adapter.cost_calls == [1]


@pytest.mark.asyncio
async def test_shadow_estimate_never_changes_decision():
    adapter = DynamicAdapter()
    target = broker(adapter, Headroom(200, shadow=100000))

    assert await target.admit(MODEL, CTX, return_parallel=True) == (
        "backpressure", None
    )
