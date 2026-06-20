from __future__ import annotations

import asyncio

import pytest

from laplace.broker import ContentionBroker
from laplace.priority import from_legacy_string
from laplace.reaper import Reaper
from tests.fakes import FakeAdapter, loaded_model

_GIB = 1024 ** 3


def _broker(adapter: FakeAdapter, *, budget_mb: int = 87040, timeout_s: float = 0.1, max_concurrent: int = 2):
    reaper = Reaper(adapter)
    broker = ContentionBroker(
        adapter,
        reaper,
        budget_mb=budget_mb,
        timeout_s=timeout_s,
        max_concurrent_lms=max_concurrent,
    )
    return broker, reaper


@pytest.mark.asyncio
async def test_resident_model_admits_without_new_memory():
    adapter = FakeAdapter(
        loaded=[loaded_model("openai/gpt-oss-120b", size_bytes=65 * _GIB)],
        footprints={"openai/gpt-oss-120b": 85000},
    )
    broker, _ = _broker(adapter)
    assert await broker.admit("openai/gpt-oss-120b") == "admit"
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_unknown_footprint_fails_open():
    adapter = FakeAdapter()
    broker, _ = _broker(adapter)
    assert await broker.admit("anthropic/claude-opus") == "admit"


@pytest.mark.asyncio
async def test_admit_evicts_idle_oldest_first():
    adapter = FakeAdapter(
        loaded=[
            loaded_model("lmstudio/qwen3.6-27b-mlx", size_bytes=20 * _GIB),
            loaded_model("qwen/qwen3-coder-30b", size_bytes=17 * _GIB),
        ],
        footprints={
            "lmstudio/qwen3.6-27b-mlx": 30000,
            "qwen/qwen3-coder-30b": 22000,
            "openai/gpt-oss-120b": 85000,
        },
    )
    broker, reaper = _broker(adapter, timeout_s=5.0)
    reaper.acquire("lmstudio/qwen3.6-27b-mlx")
    reaper.release("lmstudio/qwen3.6-27b-mlx")
    reaper.acquire("qwen/qwen3-coder-30b")
    reaper.release("qwen/qwen3-coder-30b")
    reaper._last_used["lmstudio/qwen3.6-27b-mlx"] = 1.0
    reaper._last_used["qwen/qwen3-coder-30b"] = 2.0

    assert await broker.admit("openai/gpt-oss-120b") == "admit"
    assert adapter.force_unload_calls == [
        "lmstudio/qwen3.6-27b-mlx",
        "qwen/qwen3-coder-30b",
    ]


@pytest.mark.asyncio
async def test_in_flight_model_blocks_admit_and_backpressures():
    adapter = FakeAdapter(
        loaded=[loaded_model("qwen/qwen3-coder-30b", size_bytes=17 * _GIB)],
        footprints={
            "qwen/qwen3-coder-30b": 22000,
            "openai/gpt-oss-120b": 85000,
        },
    )
    broker, reaper = _broker(adapter, timeout_s=0.05)
    reaper.acquire("qwen/qwen3-coder-30b")
    assert await broker.admit("openai/gpt-oss-120b") == "backpressure"
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_queue_then_wake_on_done():
    adapter = FakeAdapter(
        loaded=[loaded_model("qwen/qwen3-coder-30b", size_bytes=17 * _GIB)],
        footprints={
            "qwen/qwen3-coder-30b": 22000,
            "openai/gpt-oss-120b": 85000,
        },
    )
    broker, reaper = _broker(adapter, timeout_s=5.0)
    reaper.acquire("qwen/qwen3-coder-30b")
    assert await broker.admit("qwen/qwen3-coder-30b") == "admit"

    task = asyncio.create_task(broker.admit("openai/gpt-oss-120b"))
    await asyncio.sleep(0.05)
    assert not task.done()
    assert len(broker._waiters) == 1

    reaper.release("qwen/qwen3-coder-30b")
    adapter.loaded.pop("qwen/qwen3-coder-30b", None)
    await broker.done("qwen/qwen3-coder-30b")

    assert await asyncio.wait_for(task, timeout=2.0) == "admit"
    assert broker._waiters == []


@pytest.mark.asyncio
async def test_priority_order_interactive_before_scheduled():
    adapter = FakeAdapter(
        loaded=[loaded_model("qwen/qwen3-coder-30b", size_bytes=17 * _GIB)],
        footprints={
            "qwen/qwen3-coder-30b": 22000,
            "lmstudio/qwen3.6-27b-mlx": 30000,
        },
    )
    broker, reaper = _broker(adapter, timeout_s=5.0, max_concurrent=1)
    reaper.acquire("qwen/qwen3-coder-30b")
    assert await broker.admit("qwen/qwen3-coder-30b") == "admit"

    served: list[tuple[str, str]] = []

    async def admit_tag(tag: str, prio: str):
        served.append((tag, await broker.admit("lmstudio/qwen3.6-27b-mlx", priority=prio)))

    t_dream = asyncio.create_task(admit_tag("dream", "dreaming"))
    await asyncio.sleep(0.01)
    t_sched = asyncio.create_task(admit_tag("sched", "scheduled"))
    await asyncio.sleep(0.01)
    t_inter = asyncio.create_task(admit_tag("inter", "interactive"))
    await asyncio.sleep(0.05)
    assert len(broker._waiters) == 3

    reaper.release("qwen/qwen3-coder-30b")
    adapter.loaded.pop("qwen/qwen3-coder-30b", None)
    await broker.done("qwen/qwen3-coder-30b")
    await asyncio.sleep(0.05)
    assert served and served[0][0] == "inter"

    for task in (t_dream, t_sched):
        task.cancel()
    await asyncio.gather(t_dream, t_sched, return_exceptions=True)
    await t_inter


@pytest.mark.asyncio
async def test_notify_freed_wakes_waiter():
    adapter = FakeAdapter(
        loaded=[loaded_model("qwen/qwen3-coder-30b", size_bytes=17 * _GIB)],
        footprints={
            "qwen/qwen3-coder-30b": 22000,
            "openai/gpt-oss-120b": 85000,
        },
    )
    broker, reaper = _broker(adapter, timeout_s=5.0)
    reaper.acquire("qwen/qwen3-coder-30b")

    task = asyncio.create_task(broker.admit("openai/gpt-oss-120b"))
    await asyncio.sleep(0.05)
    assert not task.done()
    assert len(broker._waiters) == 1

    reaper.release("qwen/qwen3-coder-30b")
    adapter.loaded.pop("qwen/qwen3-coder-30b", None)
    await broker.notify_freed(22000)

    assert await asyncio.wait_for(task, timeout=2.0) == "admit"


@pytest.mark.asyncio
async def test_broker_passes_priority_tier_to_request_free(monkeypatch: pytest.MonkeyPatch):
    adapter = FakeAdapter(
        loaded=[
            loaded_model("lmstudio/qwen3.6-27b-mlx", size_bytes=20 * _GIB),
            loaded_model("qwen/qwen3-coder-30b", size_bytes=17 * _GIB),
        ],
        footprints={
            "lmstudio/qwen3.6-27b-mlx": 30000,
            "qwen/qwen3-coder-30b": 22000,
            "openai/gpt-oss-120b": 85000,
        },
    )
    broker, reaper = _broker(adapter, timeout_s=5.0)
    reaper.acquire("lmstudio/qwen3.6-27b-mlx")
    reaper.release("lmstudio/qwen3.6-27b-mlx")
    reaper.acquire("qwen/qwen3-coder-30b")
    reaper.release("qwen/qwen3-coder-30b")
    reaper._last_used["lmstudio/qwen3.6-27b-mlx"] = 1.0
    reaper._last_used["qwen/qwen3-coder-30b"] = 2.0

    seen: list[object] = []
    real_request_free = reaper.request_free

    async def wrapped(needed_mb: int, priority=None) -> bool:
        seen.append(priority)
        return await real_request_free(needed_mb, priority)

    monkeypatch.setattr(reaper, "request_free", wrapped)
    assert await broker.admit("openai/gpt-oss-120b", priority="interactive") == "admit"
    assert seen == [from_legacy_string("interactive")]
