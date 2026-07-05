from __future__ import annotations

import asyncio

import pytest

from laplace.broker import ContentionBroker
from laplace.reaper import Reaper
from laplace_serve.admission import govern
from tests.fakes import FakeAdapter, loaded_model

_GIB = 1024 ** 3
_MODEL = "qwen3-next-80b-a3b-thinking"


class _FailingLoadAdapter(FakeAdapter):
    async def ensure_loaded(self, model_id, context_length):
        raise RuntimeError("simulated load failure")


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
