"""cancel_admission_on_disconnect: no load or eviction for a client that left.

GLM canary 2026-11-15 00:03: OpenClaw aborted at 00:03:19, but its queued
request kept being evaluated and at 00:03:54 chose an eviction for it
(laplaced.log 11595). The proxy's disconnect watch covers forwarding only.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web

import laplace.broker as broker_module
from laplace.broker import ContentionBroker
from laplace.reaper import Reaper
from laplace_serve.config import LaplacedConfig
from tests.fakes import FakeAdapter, loaded_model
from tests.serve.harness import make_harness

_GIB = 1024 ** 3


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(broker_module, "ABANDON_POLL_S", 0.01)


def _full_pool():
    """'big' is resident and in flight, so 'm' cannot fit and nothing is evictable."""
    adapter = FakeAdapter(
        loaded=[loaded_model("big", size_bytes=10 * _GIB)],
        footprints={"big": 900, "m": 500},
    )
    reaper = Reaper(adapter)
    reaper.acquire("big")
    broker = ContentionBroker(adapter, reaper, budget_mb=1000, timeout_s=10.0)
    return adapter, reaper, broker


async def _free_pool(adapter, reaper, broker):
    reaper.release("big")
    adapter.loaded.pop("big")
    await broker.notify_freed()


async def _until(predicate, steps=300):
    for _ in range(steps):
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


# --- broker ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queued_admission_ends_when_abandoned():
    _, _, broker = _full_pool()
    gone = False
    task = asyncio.ensure_future(broker.admit_sized("m", abandoned=lambda: gone))
    assert await _until(lambda: len(broker._waiters) == 1)
    gone = True
    assert await asyncio.wait_for(task, 2.0) == ("abandoned", None)
    assert broker._waiters == []
    assert broker._active_calls() == 0


@pytest.mark.asyncio
async def test_no_eviction_starts_for_an_abandoned_admission():
    adapter = FakeAdapter(
        loaded=[loaded_model("idle", size_bytes=10 * _GIB)],
        footprints={"idle": 900, "m": 500},
    )
    reaper = Reaper(adapter)
    reaper.mark_managed("idle")  # evictable
    broker = ContentionBroker(adapter, reaper, budget_mb=1000, timeout_s=2.0)
    calls = 0

    def abandoned():
        nonlocal calls
        calls += 1
        return calls > 1  # gone after the decision to evict, before it runs

    assert await broker.admit_sized("m", abandoned=abandoned) == ("abandoned", None)
    assert adapter.force_unload_calls == []
    assert broker._active_calls() == 0


@pytest.mark.asyncio
async def test_abandoning_waiter_does_not_strand_the_next_one():
    adapter, reaper, broker = _full_pool()
    gone = False
    first = asyncio.ensure_future(broker.admit_sized("m", abandoned=lambda: gone))
    assert await _until(lambda: len(broker._waiters) == 1)
    second = asyncio.ensure_future(broker.admit_sized("m", abandoned=lambda: False))
    # Two queued waiters wake each other on every re-park (pre-existing,
    # broker.admit_sized), so the heap size flickers; just let both queue.
    await asyncio.sleep(0.05)
    assert not first.done() and not second.done()
    await _free_pool(adapter, reaper, broker)  # wakes the head waiter (first)
    gone = True
    results = await asyncio.wait_for(asyncio.gather(first, second), 2.0)
    assert ("admit", None) in results
    assert broker._waiters == []


@pytest.mark.asyncio
async def test_without_abandoned_queued_admission_still_waits():
    adapter, reaper, broker = _full_pool()
    task = asyncio.ensure_future(broker.admit_sized("m"))
    assert await _until(lambda: len(broker._waiters) == 1)
    await asyncio.sleep(0.05)
    assert not task.done()
    await _free_pool(adapter, reaper, broker)
    assert await asyncio.wait_for(task, 2.0) == ("admit", None)


# --- HTTP (report Q4.3) ---------------------------------------------------------


def _upstream():
    app = web.Application()

    async def complete(request):
        await request.read()
        return web.json_response({"choices": []})

    app.router.add_post("/v1/chat/completions", complete)
    return app


async def _disconnect_while_queued(config):
    adapter, reaper, broker = _full_pool()
    async with make_harness(
        upstream_app=_upstream(), config=config,
        adapter=adapter, reaper=reaper, broker=broker,
    ) as h:
        call = asyncio.ensure_future(
            h.client.post(
                "/hermes/v1/chat/completions",
                data=json.dumps({"model": "m", "stream": False}).encode(),
                headers={"Content-Type": "application/json"},
            )
        )
        assert await _until(lambda: len(broker._waiters) == 1)
        call.cancel()  # client gives up and closes its connection
        with pytest.raises(asyncio.CancelledError):
            await call
        waiter_gone = await _until(lambda: not broker._waiters, steps=100)
        await _free_pool(adapter, reaper, broker)
        await asyncio.sleep(0.2)
        loads = [model for model, _ in adapter.ensure_loaded_calls]
        return waiter_gone, loads, list(adapter.force_unload_calls), broker._active_calls()


@pytest.mark.asyncio
async def test_http_disconnect_while_queued_cancels_admission():
    config = LaplacedConfig.from_dict(
        {"reaper_sweep": False, "cancel_admission_on_disconnect": True}
    )
    waiter_gone, loads, unloads, active = await _disconnect_while_queued(config)
    assert waiter_gone is True
    assert loads == []  # no load for the dead client, even once memory frees
    assert unloads == []
    assert active == 0


@pytest.mark.asyncio
async def test_http_disconnect_while_queued_flag_off_keeps_todays_behaviour():
    config = LaplacedConfig.from_dict({"reaper_sweep": False})
    waiter_gone, loads, _, _ = await _disconnect_while_queued(config)
    assert waiter_gone is False  # still queued for a client that left
    assert loads == ["m"]  # and loads for it once memory frees


def test_cancel_flag_defaults_off_and_is_validated():
    assert LaplacedConfig().cancel_admission_on_disconnect is False
    with pytest.raises(ValueError, match="cancel_admission_on_disconnect"):
        LaplacedConfig.from_dict({"cancel_admission_on_disconnect": 1})
