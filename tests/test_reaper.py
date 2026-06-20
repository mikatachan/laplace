from __future__ import annotations

import asyncio
import time

import pytest

from laplace.reaper import (
    Reaper,
    _EXTERNAL_IDLE_SWEEPS,
    _EXTERNAL_IDLE_TTL,
    _LARGE_MODEL_BYTES,
    _MEDIUM_MODEL_BYTES,
)
from tests.fakes import FakeAdapter, loaded_model

_GIB = 1024 ** 3
_EXTERNAL_ID = "openclaw/qwen3.6-27b-mlx"


def _reaper(*, loaded=None, footprints=None, keep_loaded=None):
    adapter = FakeAdapter(loaded=loaded, footprints=footprints)
    return Reaper(adapter, keep_loaded=keep_loaded), adapter


def _external_entry(
    *,
    idle_s: float = _EXTERNAL_IDLE_TTL + 60.0,
    status: str | None = "idle",
    queued: int | None = 0,
    last_used_ms: float | None = None,
):
    if last_used_ms is None:
        last_used_ms = int(time.time() * 1000) - int(idle_s * 1000)
    return loaded_model(
        _EXTERNAL_ID,
        size_bytes=30 * _GIB,
        status=status,
        queued=queued,
        last_used_ms=last_used_ms,
    )


def test_model_ttl_uses_footprint_tiers():
    reaper, _adapter = _reaper()
    reaper._loaded_size_bytes = {
        "big/model": 31 * _GIB,
        "mid/model": 20 * _GIB,
        "small/model": 14 * _GIB,
    }
    assert reaper._model_ttl("big/model") == 120.0
    assert reaper._model_ttl("mid/model") == 180.0
    assert reaper._model_ttl("small/model") == 300.0


def test_model_ttl_boundaries():
    assert _LARGE_MODEL_BYTES == 30 * _GIB
    assert _MEDIUM_MODEL_BYTES == 15 * _GIB


@pytest.mark.asyncio
async def test_sweep_evicts_big_model_after_ttl():
    reaper, adapter = _reaper(loaded=[loaded_model("big/model", size_bytes=31 * _GIB)])
    reaper.acquire("big/model")
    reaper.release("big/model")
    reaper._last_used["big/model"] = time.monotonic() - 130.0
    await reaper.sweep_loaded()
    assert adapter.force_unload_calls == ["big/model"]


@pytest.mark.asyncio
async def test_sweep_keeps_small_model_under_ttl():
    reaper, adapter = _reaper(loaded=[loaded_model("small/model", size_bytes=14 * _GIB)])
    reaper.acquire("small/model")
    reaper.release("small/model")
    reaper._last_used["small/model"] = time.monotonic() - 130.0
    await reaper.sweep_loaded()
    assert adapter.force_unload_calls == []


def test_evictable_models_excludes_external_and_orders_oldest_first():
    reaper, _adapter = _reaper()
    reaper.acquire("qwen/bot-model")
    reaper.release("qwen/bot-model")
    reaper.acquire("a/old")
    reaper.release("a/old")
    reaper.acquire("b/new")
    reaper.release("b/new")
    reaper._last_used["a/old"] = 1.0
    reaper._last_used["qwen/bot-model"] = 5.0
    reaper._last_used["b/new"] = 9.0
    resident = ["openclaw/external-model", "b/new", "a/old", "qwen/bot-model"]
    assert reaper.evictable_models(resident) == ["a/old", "qwen/bot-model", "b/new"]


@pytest.mark.asyncio
async def test_reap_duplicates_keeps_in_flight_and_unloads_extra():
    loaded = [
        loaded_model("txgsync/gpt-oss-120b"),
        loaded_model("txgsync/gpt-oss-120b:2"),
    ]
    reaper, adapter = _reaper(loaded=loaded)
    reaper.acquire("txgsync/gpt-oss-120b")
    reaped = await reaper.reap_duplicates()
    assert reaped == ["txgsync/gpt-oss-120b:2"]
    assert adapter.force_unload_calls == ["txgsync/gpt-oss-120b:2"]


@pytest.mark.asyncio
async def test_request_free_evicts_until_enough_freed():
    loaded = [
        loaded_model("a/old", size_bytes=10 * _GIB),
        loaded_model("b/new", size_bytes=10 * _GIB),
    ]
    reaper, adapter = _reaper(loaded=loaded)
    for model_id in ("a/old", "b/new"):
        reaper.acquire(model_id)
        reaper.release(model_id)
    reaper._last_used["a/old"] = 1.0
    reaper._last_used["b/new"] = 9.0

    assert await reaper.request_free(5 * 1024) is True
    assert adapter.force_unload_calls == ["a/old"]


@pytest.mark.asyncio
async def test_request_free_skips_external_and_in_flight():
    loaded = [
        loaded_model("openclaw/external", size_bytes=40 * _GIB),
        loaded_model("qwen/busy", size_bytes=40 * _GIB),
    ]
    reaper, adapter = _reaper(loaded=loaded)
    reaper.acquire("qwen/busy")
    assert await reaper.request_free(10 * 1024) is False
    assert adapter.force_unload_calls == []


def test_external_constants():
    assert _EXTERNAL_IDLE_TTL == 300.0
    assert _EXTERNAL_IDLE_SWEEPS == 2


@pytest.mark.asyncio
async def test_external_idle_reaped_on_second_consecutive_sweep():
    reaper, adapter = _reaper(loaded=[_external_entry()])
    await reaper.sweep_loaded()
    assert adapter.force_unload_calls == []
    assert reaper._ext_idle_sweeps[_EXTERNAL_ID] == 1
    await reaper.sweep_loaded()
    assert adapter.force_unload_calls == [_EXTERNAL_ID]


@pytest.mark.asyncio
async def test_external_generating_resets_counter():
    entry = _external_entry()
    reaper, adapter = _reaper(loaded=[entry])
    await reaper.sweep_loaded()
    assert reaper._ext_idle_sweeps[_EXTERNAL_ID] == 1
    adapter.loaded[_EXTERNAL_ID] = _external_entry(status="generating")
    await reaper.sweep_loaded()
    assert adapter.force_unload_calls == []
    assert reaper._ext_idle_sweeps[_EXTERNAL_ID] == 0


@pytest.mark.asyncio
async def test_external_queued_nonzero_not_reaped():
    reaper, adapter = _reaper(loaded=[_external_entry(queued=3)])
    await reaper.sweep_loaded()
    await reaper.sweep_loaded()
    assert adapter.force_unload_calls == []
    assert reaper._ext_idle_sweeps[_EXTERNAL_ID] == 0


@pytest.mark.asyncio
async def test_external_unknown_status_is_fail_safe():
    reaper, adapter = _reaper(loaded=[_external_entry(status=None)])
    await reaper.sweep_loaded()
    await reaper.sweep_loaded()
    assert adapter.force_unload_calls == []
    assert reaper._ext_idle_sweeps[_EXTERNAL_ID] == 0


@pytest.mark.asyncio
async def test_external_pre_unload_recheck_aborts_when_last_used_advances():
    first_last_used = int(time.time() * 1000) - int((_EXTERNAL_IDLE_TTL + 60.0) * 1000)
    reaper, adapter = _reaper(loaded=[_external_entry(last_used_ms=first_last_used)])
    await reaper.sweep_loaded()
    assert reaper._ext_idle_sweeps[_EXTERNAL_ID] == 1

    idle_entry = _external_entry(last_used_ms=first_last_used)
    active_entry = _external_entry(last_used_ms=first_last_used + 10_000)
    calls = 0
    real_list_loaded = adapter.list_loaded

    async def scripted_list_loaded():
        nonlocal calls
        calls += 1
        if calls == 1:
            adapter.loaded[_EXTERNAL_ID] = idle_entry
        elif calls == 2:
            adapter.loaded[_EXTERNAL_ID] = active_entry
        return await real_list_loaded()

    adapter.list_loaded = scripted_list_loaded
    await reaper.sweep_loaded()
    assert adapter.force_unload_calls == []
    assert _EXTERNAL_ID in adapter.loaded
    assert reaper._ext_idle_sweeps[_EXTERNAL_ID] == 0


@pytest.mark.asyncio
async def test_sweep_notifies_broker_after_unload():
    reaper, _adapter = _reaper(loaded=[loaded_model("big/model", size_bytes=31 * _GIB)])
    reaper.acquire("big/model")
    reaper.release("big/model")
    reaper._last_used["big/model"] = time.monotonic() - 130.0

    notified: list[int | None] = []

    class BrokerSpy:
        async def notify_freed(self, freed_mb=None):
            notified.append(freed_mb)

    reaper.set_broker(BrokerSpy())
    await reaper.sweep_loaded()
    assert notified == [int(31 * _GIB / (1024 * 1024))]


@pytest.mark.asyncio
async def test_run_sweep_loop_survives_exception_and_stops(monkeypatch: pytest.MonkeyPatch):
    reaper, _adapter = _reaper()
    call_count = 0
    ran_after_error = asyncio.Event()

    async def fake_sweep():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        ran_after_error.set()

    monkeypatch.setattr(reaper, "sweep_loaded", fake_sweep)
    task = reaper.start_sweep_loop(interval=0.01)
    assert reaper.start_sweep_loop(interval=0.02) is task
    await asyncio.wait_for(ran_after_error.wait(), timeout=0.3)
    reaper.stop_sweep_loop()
    with pytest.raises(asyncio.CancelledError):
        await task
