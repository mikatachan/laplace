"""Recovery regressions; stateful fake CLI only, no daemon or live config."""
import asyncio
import json

import pytest

from laplace.adapter import ModelLoadError
from laplace.adapters.lmstudio import LMStudioAdapter

A, B = 'catalog/a', 'catalog/b'


def resident(key):
    return {'identifier': key, 'modelKey': key, 'contextLength': 4096}


class Runtime:
    def __init__(self):
        self.rows = []
        self.loads = []
        self.fail_load = False
        self.flaky_ps = False
        self.external = False
        self.started = asyncio.Event()
        self.release = None
        self.adapter = LMStudioAdapter(load_timeout_s=0)
        self.adapter._run = self.run

    async def run(self, args, timeout):
        if args[0] == 'ps':
            if self.flaky_ps and self.loads:
                self.flaky_ps = False
                return 1, '', 'transient inventory error'
            return 0, json.dumps(self.rows), ''
        if args[0] == 'ls':
            return 0, json.dumps([{'modelKey': A}, {'modelKey': B}]), ''
        assert args[0] == 'load', f'Unexpected mutation: {args}'
        self.loads.append(args[1])
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.fail_load:
            return 1, '', 'Operation canceled'
        self.rows.append(resident(args[1]))
        if self.external:
            self.rows.append(resident(B))
        return 0, '', ''


@pytest.mark.asyncio
async def test_unrelated_definite_failure_then_resident_admits():
    rt = Runtime()
    rt.rows = [resident(B)]
    rt.fail_load = True
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    await rt.adapter.ensure_loaded(B, 4096)
    assert rt.loads == [A]


@pytest.mark.asyncio
async def test_single_flaky_ps_after_success_recovers_next_admission():
    rt = Runtime()
    rt.flaky_ps = True
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A]


@pytest.mark.asyncio
async def test_external_concurrent_known_model_does_not_wedge():
    rt = Runtime()
    rt.external = True
    await rt.adapter.ensure_loaded(A, 4096)
    await rt.adapter.ensure_loaded(B, 4096)
    assert rt.loads == [A]
    assert rt.rows == [resident(A), resident(B)]


@pytest.mark.asyncio
async def test_resident_admits_within_bound_while_other_load_in_progress():
    rt = Runtime()
    rt.rows = [resident(B)]
    rt.release = asyncio.Event()
    loading = asyncio.create_task(rt.adapter.ensure_loaded(A, 4096))
    try:
        await asyncio.wait_for(rt.started.wait(), 1)
        # The other load stays blocked until AFTER this admission returns.
        await asyncio.wait_for(rt.adapter.ensure_loaded(B, 4096), 0.2)
        assert not loading.done()
    finally:
        rt.release.set()
        await loading
    assert rt.loads == [A]


@pytest.mark.asyncio
async def test_definite_failure_without_new_instance_can_retry():
    rt = Runtime()
    rt.fail_load = True
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    rt.fail_load = False
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A, A]


@pytest.mark.asyncio
async def test_uncertain_model_does_not_block_unrelated_cold_load():
    rt = Runtime()
    rt.flaky_ps = True
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    await rt.adapter.ensure_loaded(B, 4096)
    assert rt.loads == [A, B]


@pytest.mark.asyncio
async def test_failed_command_with_flaky_inventory_recovers_when_absent():
    rt = Runtime()
    rt.fail_load = rt.flaky_ps = True
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    rt.fail_load = False
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A, A]


@pytest.mark.asyncio
async def test_cancelled_command_recovers_when_exact_resident_appears():
    rt = Runtime()
    rt.release = asyncio.Event()
    loading = asyncio.create_task(rt.adapter.ensure_loaded(A, 4096))
    await asyncio.wait_for(rt.started.wait(), 1)
    loading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loading
    rt.rows = [resident(A)]
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A]


@pytest.mark.asyncio
async def test_fuzzy_failure_clears_only_after_verified_residency_without_unloading():
    rt = Runtime()
    original = rt.run
    fixed = False

    async def run(args, timeout):
        if args[0] == 'load' and not fixed:
            rt.loads.append(args[1])
            rt.rows.append(resident('unknown-fuzzy'))
            return 0, '', ''
        if args[0] == 'ls' and fixed:
            return 0, json.dumps([{'modelKey': A}]), ''
        return await original(args, timeout)

    rt.adapter._run = run
    for _ in range(2):
        with pytest.raises(ModelLoadError):
            await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A]
    fixed = True  # unrelated catalog changes must not permit a fuzzy retry
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A]
    rt.rows.append(resident(A))  # operator/external reconciliation
    await rt.adapter.ensure_loaded(A, 4096)
    rt.rows.remove(resident(A))  # later eviction permits a fresh load
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A, A]
    assert resident('unknown-fuzzy') in rt.rows


@pytest.mark.asyncio
async def test_foreign_known_instance_is_not_fuzzy_cleanup_candidate():
    rt = Runtime()
    original = rt.run

    async def run(args, timeout):
        if args[0] == 'load':
            rt.loads.append(args[1])
            rt.rows.append(resident(B))
            return 0, '', ''
        return await original(args, timeout)

    rt.adapter._run = run
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    await rt.adapter.ensure_loaded(B, 4096)
    with pytest.raises(ModelLoadError, match='unresolved'):
        await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A]
    assert rt.rows == [resident(B)]


@pytest.mark.asyncio
async def test_recovered_state_clears_even_during_other_model_load():
    rt = Runtime()
    rt.flaky_ps = True
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    rt.release = asyncio.Event()
    rt.started.clear()
    loading = asyncio.create_task(rt.adapter.ensure_loaded(B, 4096))
    try:
        await asyncio.wait_for(rt.started.wait(), 1)
        await asyncio.wait_for(rt.adapter.ensure_loaded(A, 4096), 0.2)
    finally:
        rt.release.set()
        await loading
    rt.rows = [resident(B)]  # external eviction after verified recovery
    await rt.adapter.ensure_loaded(A, 4096)
    assert rt.loads == [A, B, A]
