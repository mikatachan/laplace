from __future__ import annotations

import asyncio
import time

import pytest

from laplace.adapter import ModelLoadError
from laplace.broker import ContentionBroker
from laplace.reaper import Reaper
from laplace_serve.admission import govern
from tests.fakes import FakeAdapter, loaded_model

_GIB = 1024 ** 3
_MODEL = "qwen3-next-80b-a3b-thinking"


class _FailingLoadAdapter(FakeAdapter):
    async def ensure_loaded(self, model_id, context_length):
        raise RuntimeError("simulated load failure")


class _ModelLoadErrorAdapter(FakeAdapter):
    async def ensure_loaded(self, model_id, context_length):
        raise ModelLoadError("simulated model load failure")


class _ExplodingBroker:
    async def admit(self, *_args, **_kwargs):
        raise RuntimeError("broker exploded")

    async def done(self, *_args, **_kwargs):
        return None


class _BackpressureBroker:
    async def admit(self, *_args, **_kwargs):
        return "backpressure"

    async def done(self, *_args, **_kwargs):
        return None


def _resident_adapter(cls=FakeAdapter):
    return cls(
        loaded=[loaded_model(_MODEL, size_bytes=40 * _GIB)],
        footprints={_MODEL: 48000},
    )


@pytest.mark.asyncio
async def test_backpressure_does_not_proxy():
    adapter = _resident_adapter()
    reaper = Reaper(adapter)
    async with govern(
        _BackpressureBroker(), reaper, adapter,
        model=_MODEL, ctx=131072, tier="interactive", fail_open=True,
    ) as adm:
        assert adm.decision == "backpressure"
        assert adm.proxy is False


@pytest.mark.asyncio
async def test_fail_open_on_broker_exception():
    adapter = _resident_adapter()
    reaper = Reaper(adapter)
    async with govern(
        _ExplodingBroker(), reaper, adapter,
        model=_MODEL, ctx=131072, tier="interactive", fail_open=True,
    ) as adm:
        assert adm.decision == "fail-open"
        assert adm.proxy is True


@pytest.mark.asyncio
async def test_broker_exception_raises_when_fail_open_off():
    adapter = _resident_adapter()
    reaper = Reaper(adapter)
    with pytest.raises(RuntimeError):
        async with govern(
            _ExplodingBroker(), reaper, adapter,
            model=_MODEL, ctx=131072, tier="interactive", fail_open=False,
        ):
            pass


@pytest.mark.asyncio
async def test_missing_model_forwards_ungoverned():
    adapter = FakeAdapter()
    reaper = Reaper(adapter)
    broker = ContentionBroker(adapter, reaper, budget_mb=90000)
    async with govern(
        broker, reaper, adapter,
        model=None, ctx=None, tier="scheduled", fail_open=True,
    ) as adm:
        assert adm.decision == "fail-open"
        assert adm.proxy is True


# --- First-touch unmark trio (plan B2 r3 amendment, wrapper level) ---


@pytest.mark.asyncio
async def test_i_failed_load_on_external_model_stays_external():
    adapter = _resident_adapter(_FailingLoadAdapter)
    reaper = Reaper(adapter)
    broker = ContentionBroker(adapter, reaper, budget_mb=90000)

    assert reaper.is_managed(_MODEL) is False
    async with govern(
        broker, reaper, adapter,
        model=_MODEL, ctx=131072, tier="interactive", fail_open=True,
    ) as adm:
        assert adm.decision == "fail-open"

    assert reaper.is_managed(_MODEL) is False
    assert broker._active_calls() == 0


@pytest.mark.asyncio
async def test_ii_failed_load_on_managed_model_stays_managed():
    adapter = _resident_adapter(_FailingLoadAdapter)
    reaper = Reaper(adapter)
    broker = ContentionBroker(adapter, reaper, budget_mb=90000)

    reaper.mark_managed(_MODEL)
    assert reaper.is_managed(_MODEL) is True
    async with govern(
        broker, reaper, adapter,
        model=_MODEL, ctx=131072, tier="interactive", fail_open=True,
    ) as adm:
        assert adm.decision == "fail-open"

    assert reaper.is_managed(_MODEL) is True
    assert broker._active_calls() == 0


@pytest.mark.asyncio
async def test_model_load_error_does_not_fail_open_when_flag_on():
    adapter = _resident_adapter(_ModelLoadErrorAdapter)
    reaper = Reaper(adapter)
    broker = ContentionBroker(adapter, reaper, budget_mb=90000)

    with pytest.raises(ModelLoadError):
        async with govern(
            broker, reaper, adapter,
            model=_MODEL, ctx=131072, tier="interactive", fail_open=True,
        ):
            pass

    assert broker._active_calls() == 0


class _CoordinatedAdapter(FakeAdapter):
    """First ensure_loaded blocks then fails (A); later calls succeed (B)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0
        self.a_reached = asyncio.Event()
        self.a_proceed = asyncio.Event()

    async def ensure_loaded(self, model_id, context_length):
        self.calls += 1
        if self.calls == 1:
            self.a_reached.set()
            await self.a_proceed.wait()
            raise RuntimeError("simulated load failure for A")
        await super().ensure_loaded(model_id, context_length)


class _BlockingLoadAdapter(FakeAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_started = asyncio.Event()
        self.finish_load = asyncio.Event()

    async def ensure_loaded(self, model_id, context_length):
        self.load_started.set()
        await self.finish_load.wait()
        await super().ensure_loaded(model_id, context_length)


@pytest.mark.asyncio
async def test_sweep_does_not_reap_model_while_load_is_in_progress():
    adapter = _resident_adapter(_BlockingLoadAdapter)
    reaper = Reaper(adapter)
    broker = ContentionBroker(adapter, reaper, budget_mb=90000)
    reaper.mark_managed(_MODEL)
    reaper._last_used[_MODEL] = time.monotonic() - 3600.0

    async def load_model():
        async with govern(
            broker, reaper, adapter,
            model=_MODEL, ctx=131072, tier="interactive", fail_open=False,
        ):
            pass

    loading = asyncio.create_task(load_model())
    try:
        await asyncio.wait_for(adapter.load_started.wait(), timeout=2.0)
        # Prove the explicit load pin protects even a load longer than its TTL.
        reaper._last_used[_MODEL] = time.monotonic() - 3600.0
        await reaper.sweep_loaded()
        assert adapter.force_unload_calls == []
        assert not loading.done()
    finally:
        adapter.finish_load.set()
        await loading


@pytest.mark.asyncio
async def test_iii_concurrent_a_fails_b_succeeds_stays_managed():
    adapter = _CoordinatedAdapter(
        loaded=[loaded_model(_MODEL, size_bytes=40 * _GIB)],
        footprints={_MODEL: 48000},
    )
    reaper = Reaper(adapter)
    broker = ContentionBroker(adapter, reaper, budget_mb=90000)

    b_acquired = asyncio.Event()
    b_hold = asyncio.Event()

    async def run_a():
        async with govern(
            broker, reaper, adapter,
            model=_MODEL, ctx=131072, tier="interactive", fail_open=True,
        ) as adm:
            assert adm.decision == "fail-open"

    async def run_b():
        async with govern(
            broker, reaper, adapter,
            model=_MODEL, ctx=131072, tier="interactive", fail_open=True,
        ) as adm:
            assert adm.decision == "admit"
            b_acquired.set()
            await b_hold.wait()

    task_a = asyncio.create_task(run_a())
    await asyncio.wait_for(adapter.a_reached.wait(), timeout=2.0)  # A marked, blocked in load
    task_b = asyncio.create_task(run_b())
    await asyncio.wait_for(b_acquired.wait(), timeout=2.0)  # B acquired, in_flight == 1

    assert reaper.in_flight(_MODEL) == 1
    adapter.a_proceed.set()  # A fails now; its unmark must no-op (B holds the model)
    await asyncio.wait_for(task_a, timeout=2.0)

    assert reaper.is_managed(_MODEL) is True

    b_hold.set()
    await asyncio.wait_for(task_b, timeout=2.0)
    assert reaper.is_managed(_MODEL) is True
    assert broker._active_calls() == 0


_RACE_MODEL = "qwen3.8-27b-uncensored-mlx"


def _unloading_pool():
    """A resident, managed, idle model whose reaper unload the test holds open."""
    from tests.fakes import GatedUnloadAdapter

    adapter = GatedUnloadAdapter(
        loaded=[loaded_model(_RACE_MODEL, size_bytes=20 * _GIB, context_length=131072)],
        footprints={_RACE_MODEL: 26000},
    )
    reaper = Reaper(adapter)
    reaper.mark_managed(_RACE_MODEL)
    reaper._last_used[_RACE_MODEL] = time.monotonic() - 3600.0
    return adapter, reaper


async def _govern_and_record(broker, reaper, adapter):
    async with govern(
        broker, reaper, adapter,
        model=_RACE_MODEL, ctx=131072, tier="interactive", fail_open=False,
    ) as adm:
        adapter.events.append(f"proxy:{adm.decision}")
        return _RACE_MODEL in adapter.loaded


@pytest.mark.asyncio
async def test_admission_waits_for_inflight_reaper_unload_of_same_model():
    # 2026-11-10 15:06:52: request_free began unloading the model, a request for
    # that model was admitted while lms ps still listed it, and was proxied to
    # an instance that was going away (HTTP 400).
    adapter, reaper = _unloading_pool()
    broker = ContentionBroker(adapter, reaper, budget_mb=87040, timeout_s=5.0)

    unload = asyncio.create_task(reaper.request_free(10000))
    await asyncio.wait_for(adapter.unload_started.wait(), 1.0)

    request = asyncio.create_task(_govern_and_record(broker, reaper, adapter))
    await asyncio.sleep(0.05)
    proxied_mid_unload = [e for e in adapter.events if e.startswith(("proxy", "ensure_loaded"))]
    adapter.gate.set()

    resident_at_proxy = await asyncio.wait_for(request, 2.0)
    assert await asyncio.wait_for(unload, 2.0) is True
    assert proxied_mid_unload == []
    assert adapter.events == [
        f"unload-start:{_RACE_MODEL}",
        f"unload-end:{_RACE_MODEL}",
        f"ensure_loaded:{_RACE_MODEL}",
        "proxy:admit",
    ]
    assert resident_at_proxy is True


class _AdmitBroker:
    async def admit(self, *_args, **_kwargs):
        return "admit"

    async def done(self, *_args, **_kwargs):
        return None


@pytest.mark.asyncio
async def test_govern_waits_for_unload_that_began_after_broker_admit():
    # Backstop: the broker decided before the unload started, so only govern's
    # wait stands between the request and the leaving instance.
    adapter, reaper = _unloading_pool()

    unload = asyncio.create_task(reaper.request_free(10000))
    await asyncio.wait_for(adapter.unload_started.wait(), 1.0)
    request = asyncio.create_task(_govern_and_record(_AdmitBroker(), reaper, adapter))
    await asyncio.sleep(0.05)
    assert "proxy:admit" not in adapter.events
    adapter.gate.set()

    assert await asyncio.wait_for(request, 2.0) is True
    await asyncio.wait_for(unload, 2.0)
    assert adapter.events.index(f"unload-end:{_RACE_MODEL}") < adapter.events.index(
        f"ensure_loaded:{_RACE_MODEL}"
    )
