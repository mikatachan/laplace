"""Adopting already-resident models at startup (card 2dc5e7c2).

2026-11-14 16:35:26 laplaced restarted with gemma and qwen3-coder resident,
both loaded by the previous process. Managed state is in memory only, so both
read as external: a GLM request queued 358 s at evictable_credit_mb=0 until
qwen hit the external-idle sweep and gemma was touched by a new request.
"""

from __future__ import annotations

import logging

import pytest
from aiohttp import web

from laplace.broker import ContentionBroker
from laplace.reaper import Reaper
from laplace_serve.config import LaplacedConfig
from tests.fakes import FakeAdapter, loaded_model
from tests.serve.harness import make_harness
from tests.test_fit_context import GLM, GLM_CTX, H_1119, QWEN, ArchAdapter, Probe, _idle_qwen

_MIB = 1024 ** 2
_EMBED = "nomic-embed-text"


def _upstream() -> web.Application:
    return web.Application()


@pytest.mark.asyncio
async def test_restart_without_adoption_queues_behind_previous_residents(tmp_path):
    # The incident: a fresh Reaper (just restarted) sees the idle qwen as external.
    adapter = ArchAdapter(tmp_path, loaded=[_idle_qwen()])
    released = H_1119 + 24101
    broker = ContentionBroker(
        adapter, Reaper(adapter), headroom_mb=Probe(H_1119, H_1119, released, released),
        timeout_s=0.1, fit_context=True, measured_eviction=True,
    )
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=20000) == ("backpressure", None)
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_restart_with_adoption_evicts_previous_residents(tmp_path):
    adapter = ArchAdapter(tmp_path, loaded=[_idle_qwen()])
    reaper = Reaper(adapter)
    assert reaper.adopt_resident(await adapter.list_loaded()) == [QWEN]
    released = H_1119 + 24101
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=Probe(H_1119, H_1119, released, released),
        timeout_s=1.0, fit_context=True, measured_eviction=True,
    )
    decision, _ = await broker.admit_sized(GLM, GLM_CTX, need_context=20000)
    assert decision == "admit"
    assert adapter.force_unload_calls == [QWEN]


@pytest.mark.asyncio
async def test_adopted_model_still_protected_while_generating(tmp_path):
    busy = loaded_model(QWEN, size_bytes=16395 * _MIB, status="generating",
                        queued=0, context_length=131072)
    adapter = ArchAdapter(tmp_path, loaded=[busy])
    reaper = Reaper(adapter)
    reaper.adopt_resident(await adapter.list_loaded())
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=Probe(H_1119), timeout_s=0.1,
        fit_context=True, measured_eviction=True,
    )
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=20000) == ("backpressure", None)
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_adopted_model_is_not_reaped_by_the_first_sweep():
    # mark_managed records a touch, so the managed-idle TTL starts at adoption
    # instead of reading an unset last-use as infinitely idle.
    adapter = FakeAdapter(loaded=[loaded_model(QWEN, size_bytes=16 * 1024 ** 3, status="idle", queued=0)])
    reaper = Reaper(adapter)
    reaper.adopt_resident(await adapter.list_loaded())
    await reaper.sweep_loaded()
    assert adapter.force_unload_calls == []


def _residents() -> FakeAdapter:
    return FakeAdapter(loaded=[
        loaded_model(QWEN, size_bytes=16 * 1024 ** 3, status="idle", queued=0),
        loaded_model(_EMBED, size_bytes=1024 ** 3, status="idle", queued=0),
    ])


@pytest.mark.asyncio
async def test_startup_adopts_residents_when_flag_on(caplog):
    adapter = _residents()
    config = LaplacedConfig.from_dict({"adopt_resident_on_start": True, "reaper_sweep": False})
    with caplog.at_level(logging.INFO, logger="laplace_serve.app"):
        reaper = Reaper(adapter, keep_loaded=set(config.keep_loaded))
        async with make_harness(
            upstream_app=_upstream(), config=config, adapter=adapter, reaper=reaper,
        ) as h:
            assert h.reaper.is_managed(QWEN)
            assert not h.reaper.is_managed(_EMBED)  # keep_loaded is never adopted
            assert h.reaper.evictable_models([QWEN, _EMBED]) == [QWEN]
    assert any("adopted 1 resident model(s)" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_startup_does_not_adopt_when_flag_off():
    adapter = _residents()
    config = LaplacedConfig.from_dict({"reaper_sweep": False})
    async with make_harness(upstream_app=_upstream(), config=config, adapter=adapter) as h:
        assert not h.reaper.is_managed(QWEN)
        assert h.reaper.evictable_models([QWEN, _EMBED]) == []


class _ListRaises(FakeAdapter):
    async def list_loaded(self):
        raise RuntimeError("lms ps exploded")


@pytest.mark.asyncio
async def test_startup_survives_adoption_failure(caplog):
    config = LaplacedConfig.from_dict({"adopt_resident_on_start": True, "reaper_sweep": False})
    async with make_harness(upstream_app=_upstream(), config=config, adapter=_ListRaises()) as h:
        resp = await h.client.get("/healthz")
        assert resp.status == 200
    assert any("adopting resident models at startup failed" in r.getMessage()
               for r in caplog.records)


def test_adopt_flag_defaults_off_and_is_validated():
    assert LaplacedConfig().adopt_resident_on_start is False
    with pytest.raises(ValueError, match="adopt_resident_on_start"):
        LaplacedConfig.from_dict({"adopt_resident_on_start": "yes"})
