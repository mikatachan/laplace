"""Bounded recovery probes: fake CLI and clock, never a live lms process."""
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from laplace.adapter import ModelLoadError
from laplace.adapters import lmstudio
from test_load_recovery import A, B, Runtime, resident


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    # Patch only the adapter's clock, not asyncio's event-loop clock.
    monkeypatch.setattr(lmstudio, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    return clock


def timeout_runtime(clock):
    rt = Runtime()
    rt.adapter = lmstudio.LMStudioAdapter(load_timeout_s=10)
    original = rt.run

    async def run(args, timeout):
        if args[0] == 'load' and not rt.loads:
            rt.loads.append(args[1])
            clock.now += timeout
            raise TimeoutError()
        return await original(args, timeout)

    rt.adapter._run = run
    return rt


@pytest.mark.asyncio
async def test_timeout_repeated_admissions_recover_after_grace(clock):
    rt = timeout_runtime(clock)
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    for now in (110, 150, 199.999):
        clock.now = now
        with pytest.raises(ModelLoadError, match='unresolved'):
            await rt.adapter.ensure_loaded(A, 4096)
        assert rt.loads == [A]
    clock.now = 200  # load start + timeout (10) + default grace (90)
    await rt.adapter.ensure_loaded(A, 4096)
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A, A]
    assert rt.rows == [resident(A)]


@pytest.mark.asyncio
async def test_grace_blocks_other_cold_load_but_resident_admits(clock):
    rt = timeout_runtime(clock)
    rt.rows = [resident('already-resident')]
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    await rt.adapter.ensure_loaded('already-resident', 4096)
    with pytest.raises(ModelLoadError, match='unresolved'):
        await rt.adapter.ensure_loaded(B, 4096)
    assert rt.loads == [A]
    clock.now = 200
    await rt.adapter.ensure_loaded(B, 4096)
    assert rt.loads == [A, B]


@pytest.mark.asyncio
async def test_run_timeout_kills_then_reaps_process(monkeypatch):
    events = []
    proc = SimpleNamespace(
        communicate=AsyncMock(return_value=(b"", b"")),
        kill=Mock(side_effect=lambda: events.append('kill')),
        wait=AsyncMock(side_effect=lambda: events.append('wait')),
    )
    monkeypatch.setattr(lmstudio.asyncio, 'create_subprocess_exec', AsyncMock(return_value=proc))
    with pytest.raises(TimeoutError):
        await lmstudio.LMStudioAdapter()._run(['load', A], timeout=0)
    proc.kill.assert_called_once_with()
    proc.wait.assert_awaited_once_with()
    assert events == ['kill', 'wait']


@pytest.mark.asyncio
async def test_nonzero_exit_late_instance_does_not_double_load(clock):
    rt = Runtime()
    rt.fail_load = True
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    rt.fail_load = False
    for now in (100, 140, 189.999):
        clock.now = now
        with pytest.raises(ModelLoadError):
            await rt.adapter.ensure_loaded(A, 4096)
        assert rt.loads == [A]
    rt.rows.append(resident(A))  # background load finishes late
    await rt.adapter.ensure_loaded(A, 4096)
    clock.now = 190
    await rt.adapter.ensure_loaded(B, 4096)
    assert rt.loads == [A, B]


@pytest.mark.asyncio
async def test_nonzero_exit_window_is_grace_only_from_cli_return(clock, caplog):
    rt = Runtime()
    rt.adapter = lmstudio.LMStudioAdapter(load_timeout_s=300, load_grace_s=7)
    original = rt.run

    async def run(args, timeout):
        if args[0] == 'load' and not rt.loads:
            rt.loads.append(args[1])
            clock.now = 112
            return 1, '', 'Operation canceled'
        return await original(args, timeout)

    rt.adapter._run = run
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    clock.now = 118.999
    with pytest.raises(ModelLoadError, match='unresolved'):
        await rt.adapter.ensure_loaded(A, 4096)
    clock.now = 119
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A, A]
    assert any(
        record.levelno == logging.ERROR
        and A in record.message
        and 'reason=rc' in record.message
        and 'window=7s' in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_timeout_message_contains_duration_model_and_context(clock, caplog):
    rt = timeout_runtime(clock)
    message = f'lms load timed out after 10s for {A} at ctx=4096'
    with pytest.raises(ModelLoadError, match=message):
        await rt.adapter.ensure_loaded(A, 4096)
    assert message in caplog.text
    assert any(
        record.levelno == logging.ERROR
        and A in record.message
        and 'reason=timeout' in record.message
        and 'window=100s' in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_custom_grace_and_fresh_failed_inventory(clock):
    rt = Runtime()
    rt.adapter = lmstudio.LMStudioAdapter(load_timeout_s=10, load_grace_s=7)
    rt.adapter._run = rt.run
    rt.fail_load = True
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    rt.fail_load = False
    clock.now = 106.999
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    clock.now = 107
    rt.flaky_ps = True
    with pytest.raises(ModelLoadError, match='cannot establish residency'):
        await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A]
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A, A]


@pytest.mark.asyncio
async def test_unidentified_new_instance_prevents_expired_retry(clock):
    rt = timeout_runtime(clock)
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    rt.rows = [resident('unidentified')]
    clock.now = 200
    with pytest.raises(ModelLoadError, match='unresolved'):
        await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A]


@pytest.mark.asyncio
async def test_newly_cataloged_external_instance_does_not_wedge_expired_retry(clock, caplog):
    rt = timeout_runtime(clock)
    original = rt.adapter._run
    catalog = {A, B}

    async def run(args, timeout):
        if args[0] == 'ls':
            return 0, json.dumps([{'modelKey': key} for key in catalog]), ''
        return await original(args, timeout)

    rt.adapter._run = run
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    catalog.add('catalog/newly-downloaded')
    rt.rows = [resident('catalog/newly-downloaded')]
    clock.now = 200
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A, A]
    assert any(
        record.levelno == logging.ERROR
        and 'window expired' in record.message
        and A in record.message
        and 'window=100s' in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_persisting_unidentified_wedge_logs_once_at_error(clock, caplog):
    rt = timeout_runtime(clock)
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    rt.rows = [resident('unidentified')]
    clock.now = 200
    for _ in range(2):
        with pytest.raises(ModelLoadError, match='operator reconciliation required'):
            await rt.adapter.ensure_loaded(A, 4096)
    wedge_logs = [
        record for record in caplog.records
        if record.levelno == logging.ERROR
        and 'operator reconciliation required' in record.message
    ]
    assert len(wedge_logs) == 1


@pytest.mark.asyncio
async def test_known_external_instance_does_not_wedge_expired_retry(clock):
    rt = timeout_runtime(clock)
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    rt.rows = [resident(B)]
    clock.now = 200
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A, A]


@pytest.mark.asyncio
async def test_matching_late_instance_clears_timeout_before_deadline(clock):
    rt = timeout_runtime(clock)
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    rt.rows = [resident(A)]
    await rt.adapter.ensure_loaded(A, 4096)
    await rt.adapter.ensure_loaded(B, 4096)
    assert rt.loads == [A, B]
