"""Regression probes use a stateful CLI stub; no LM Studio process is invoked."""
import asyncio
import json
import logging

import pytest
from aiohttp.test_utils import TestClient, TestServer

from laplace.adapter import ModelLoadError
from laplace.adapters.lmstudio import LMStudioAdapter
from laplace.broker import ContentionBroker
from laplace.reaper import Reaper
from laplace_serve.admission import govern
from laplace_serve.app import build_app
from laplace_serve.config import LaplacedConfig


OLD = "huihui-qwen3.6-27b-abliterated-4-msq"
NEW = "huihui-qwen3.6-27b-abliterated-4.5bit-msq"


class CLI:
    def __init__(self, requested=OLD, actual=None, residents=()):
        self.requested = requested
        self.actual = actual or requested
        self.residents = list(residents)
        self.calls = []
        self.ps_error = False
        self.unload_error = False
        self.catalog = [requested]

    async def run(self, args, timeout):
        self.calls.append(args)
        await asyncio.sleep(0)  # expose concurrent snapshot/load races
        if args == ["ps", "--json"]:
            return (1, "", "inventory unavailable") if self.ps_error else (0, json.dumps(self.residents), "")
        if args == ["ls", "--json"]:
            return 0, json.dumps([{"modelKey": key} for key in self.catalog]), ""
        if args[0] == "load":
            self.residents.append({"identifier": self.actual, "modelKey": self.actual, "contextLength": 4096})
            return 0, "", ""
        if args[0] == "unload":
            if self.unload_error:
                return 1, "", "unload failed"
            self.residents = [entry for entry in self.residents if entry["identifier"] != args[1]]
            return 0, "", ""
        raise AssertionError(args)

    def adapter(self):
        adapter = LMStudioAdapter(load_timeout_s=0, unload_verify_delay_s=0, footprint_overrides={self.requested: 100})
        adapter._run = self.run
        return adapter

    @property
    def loads(self):
        return [args for args in self.calls if args[0] == "load"]


@pytest.mark.asyncio
async def test_mismatch_never_unloads_unowned_instance_or_retries(caplog):
    cli = CLI(actual=NEW + ":2")
    adapter = cli.adapter()
    for _ in range(3):
        with pytest.raises(ModelLoadError) as caught:
            await adapter.ensure_loaded(OLD, 4096)
        assert OLD in str(caught.value)
        assert NEW + ":2" in str(caught.value)
    assert len(cli.loads) == 1
    assert not any(call[0] == "unload" for call in cli.calls)
    assert cli.residents[0]["identifier"] == NEW + ":2"
    assert any(record.levelno == logging.ERROR and OLD in record.message and NEW in record.message for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", [False, True])
async def test_concurrent_admissions_issue_exactly_one_load(mismatch):
    cli = CLI(actual=NEW if mismatch else OLD)
    adapter = cli.adapter()
    reaper = Reaper(adapter, delay_seconds=300)
    broker = ContentionBroker(adapter, reaper, budget_mb=90000, max_concurrent_lms=20)

    # Make all 20 admissions reach the adapter before allowing inventory reads.
    # This exposes the race deterministically instead of relying on scheduler luck.
    entered = 0
    all_entered = asyncio.Event()
    ensure_loaded = adapter.ensure_loaded

    async def concurrent_load(model, ctx):
        nonlocal entered
        entered += 1
        if entered == 20:
            all_entered.set()
        await asyncio.wait_for(all_entered.wait(), timeout=2)
        await ensure_loaded(model, ctx)

    adapter.ensure_loaded = concurrent_load

    async def request():
        async with govern(broker, reaper, adapter, model=OLD, ctx=4096, tier="interactive", fail_open=True) as admission:
            assert admission.decision == "admit"
            await asyncio.sleep(0)

    results = await asyncio.gather(*(request() for _ in range(20)), return_exceptions=True)
    assert len(cli.loads) == 1
    assert all(isinstance(result, ModelLoadError) if mismatch else result is None for result in results)
    assert broker._reserved == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["identifier", "modelKey", "path"])
@pytest.mark.parametrize("context", [4096, 1024, None])
async def test_existing_suffixed_instance_prevents_load(identity, context):
    entry = {"identifier": "custom-instance:2", identity: OLD + ":2", "contextLength": context}
    cli = CLI(residents=[entry])
    adapter = cli.adapter()
    if context == 4096:
        await adapter.ensure_loaded(OLD, 4096)
    else:
        with pytest.raises(ModelLoadError):
            await adapter.ensure_loaded(OLD, 4096)
    assert cli.loads == []


@pytest.mark.asyncio
async def test_missing_exact_catalog_key_prevents_fuzzy_duplicate():
    cli = CLI(residents=[{"identifier": NEW, "modelKey": NEW, "contextLength": 4096}])
    cli.catalog = [NEW]
    with pytest.raises(ModelLoadError) as caught:
        await cli.adapter().ensure_loaded(OLD, 4096)
    assert OLD in str(caught.value) and NEW in str(caught.value)
    assert cli.loads == []


@pytest.mark.asyncio
async def test_inventory_failure_prevents_load():
    cli = CLI()
    cli.ps_error = True
    with pytest.raises(ModelLoadError):
        await cli.adapter().ensure_loaded(OLD, 4096)
    assert cli.loads == []


@pytest.mark.asyncio
async def test_fuzzy_failure_does_not_block_other_model_loads():
    cli = CLI(actual=NEW)
    adapter = cli.adapter()
    with pytest.raises(ModelLoadError, match="ownership unknown"):
        await adapter.ensure_loaded(OLD, 4096)
    cli.catalog.append("another-model")
    cli.actual = "another-model"
    await adapter.ensure_loaded("another-model", 4096)
    assert len(cli.loads) == 2
    assert not any(call[0] == "unload" for call in cli.calls)


@pytest.mark.asyncio
async def test_unload_inventory_failure_is_not_success():
    cli = CLI()
    cli.ps_error = True
    assert not await cli.adapter().force_unload(OLD)


@pytest.mark.asyncio
async def test_startup_flags_ids_from_both_config_sections(caplog):
    cli = CLI(requested=NEW)
    adapter = cli.adapter()
    reaper = Reaper(adapter)
    broker = ContentionBroker(adapter, reaper, budget_mb=90000)
    config = LaplacedConfig(model_context={OLD: 131072, NEW: 131072}, model_footprint_mb={"missing-footprint": 20500}, reaper_sweep=False)
    app = build_app(config, adapter=adapter, reaper=reaper, broker=broker)
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/healthz")).status == 200
    errors = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2
    assert any(OLD in message and NEW in message for message in errors)
    assert any("missing-footprint" in message and NEW in message for message in errors)
    assert cli.calls == [["ls", "--json"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["broken json", "{}", '[{}]', '[{"identifier":"x"},{"identifier":"x"}]'])
async def test_invalid_inventory_prevents_load(output):
    cli = CLI()
    adapter = cli.adapter()

    async def run(args, timeout):
        if args == ["ps", "--json"]:
            return 0, output, ""
        return await cli.run(args, timeout)

    adapter._run = run
    with pytest.raises(ModelLoadError):
        await adapter.ensure_loaded(OLD, 4096)
    assert cli.loads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_uncertain_command_outcome_blocks_retry(cancel):
    cli = CLI()
    adapter = cli.adapter()

    async def run(args, timeout):
        if args[0] == "load":
            cli.calls.append(args)
            raise asyncio.CancelledError() if cancel else TimeoutError("load timed out")
        return await cli.run(args, timeout)

    adapter._run = run
    with pytest.raises(asyncio.CancelledError if cancel else ModelLoadError):
        await adapter.ensure_loaded(OLD, 4096)
    with pytest.raises(ModelLoadError, match="unresolved"):
        await adapter.ensure_loaded(OLD, 4096)
    assert len(cli.loads) == 1


@pytest.mark.asyncio
async def test_startup_catalog_failure_logs_without_crashing(caplog):
    adapter = LMStudioAdapter()

    async def run(args, timeout):
        return 1, "", "catalog unavailable"

    adapter._run = run
    await adapter.validate_model_ids({OLD})
    assert "startup model validation unavailable" in caplog.text
