"""Startup model-validation retry: the boot race with LM Studio.

Covers the validate_model_ids bool contract, the background retry loop
(stops at first success / gives up after six retries), and cleanup
cancellation of the retry task.
"""

from __future__ import annotations

import json
import logging

import pytest
from aiohttp import web

from laplace.adapters.lmstudio import LMStudioAdapter

import laplace_serve.app as app_module
from laplace_serve.config import LaplacedConfig
from tests.serve.harness import make_harness

# Injectable retry schedule, monkeypatched to zeros so tests never wait.
ZERO_DELAYS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class _ScriptedAdapter(LMStudioAdapter):
    """LMStudioAdapter whose validate_model_ids follows a scripted outcome list."""

    def __init__(self, outcomes: list[bool]):
        super().__init__()
        self._outcomes = list(outcomes)
        self.validate_calls = 0

    async def validate_model_ids(self, model_ids: set[str]) -> bool:
        self.validate_calls += 1
        if self._outcomes:
            return self._outcomes.pop(0)
        return False


def _unavailable_adapter() -> LMStudioAdapter:
    """Real adapter whose lms CLI always fails (catalog unreachable)."""
    adapter = LMStudioAdapter()

    async def run(args, timeout):
        return 1, "", "catalog unavailable"

    adapter._run = run
    return adapter


@pytest.mark.asyncio
async def test_validate_model_ids_false_when_catalog_unavailable():
    adapter = _unavailable_adapter()
    assert await adapter.validate_model_ids({"some/model"}) is False


@pytest.mark.asyncio
async def test_validate_model_ids_true_when_catalog_available():
    adapter = LMStudioAdapter()

    async def run(args, timeout):
        return 0, json.dumps([{"modelKey": "some/model"}]), ""

    adapter._run = run
    assert await adapter.validate_model_ids({"some/model"}) is True


@pytest.mark.asyncio
async def test_validate_model_ids_true_for_empty_set():
    adapter = LMStudioAdapter()

    async def run(args, timeout):
        raise AssertionError("catalog must not be queried for an empty id set")

    adapter._run = run
    assert await adapter.validate_model_ids(set()) is True


@pytest.mark.asyncio
async def test_validate_model_ids_true_even_when_reporting_stale_id(caplog):
    adapter = LMStudioAdapter()

    async def run(args, timeout):
        return 0, json.dumps([{"modelKey": "other/model"}]), ""

    adapter._run = run
    with caplog.at_level(logging.ERROR, logger="laplace.adapters.lmstudio"):
        assert await adapter.validate_model_ids({"some/model"}) is True
    # The catalog was reachable, so validation ran (and reported the stale id).
    assert "no exact catalog match" in caplog.text


@pytest.mark.asyncio
async def test_retry_loop_stops_at_first_true(monkeypatch):
    monkeypatch.setattr(app_module, "STARTUP_VALIDATION_RETRY_DELAYS", ZERO_DELAYS)
    adapter = _ScriptedAdapter([False, False, True])

    await app_module._retry_startup_validation(adapter, {"some/model"})
    assert adapter.validate_calls == 3


@pytest.mark.asyncio
async def test_retry_loop_gives_up_after_six_retries(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "STARTUP_VALIDATION_RETRY_DELAYS", ZERO_DELAYS)
    adapter = _ScriptedAdapter([False] * 6)

    with caplog.at_level(logging.ERROR, logger="laplace_serve.app"):
        await app_module._retry_startup_validation(adapter, {"some/model"})
    assert adapter.validate_calls == 6
    assert "lmstudio startup model validation gave up after 6 retries" in caplog.text


@pytest.mark.asyncio
async def test_retry_task_cancelled_on_cleanup(monkeypatch):
    # Long delays keep the retry task sleeping so cleanup must cancel it.
    monkeypatch.setattr(app_module, "STARTUP_VALIDATION_RETRY_DELAYS", (3600.0,) * 6)

    config = LaplacedConfig(reaper_sweep=False, model_context={"some/model": 4096})
    async with make_harness(
        upstream_app=web.Application(),
        config=config,
        adapter=_unavailable_adapter(),
    ) as harness:
        task = harness.client.app[app_module.STARTUP_VALIDATION_RETRY_TASK]
        assert not task.done()

    # client.close() ran _on_cleanup: the retry task is cancelled, not pending.
    assert task.cancelled()
