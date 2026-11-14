"""Fit-sized load context and measured eviction (live-memory admission)."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from laplace.adapters.lmstudio import LMStudioAdapter
from laplace.broker import ContentionBroker
from laplace.reaper import Reaper
from laplace_serve.__main__ import build_components
from laplace_serve.admission import govern
from laplace_serve.app import _estimate_need_tokens
from laplace_serve.config import LaplacedConfig
from tests.fakes import FakeAdapter, loaded_model
from tests.serve.harness import UpstreamState, make_harness

_MIB = 1024 ** 2
GLM = "glm-4.5-air"
OSS = "gpt-oss-120b"
QWEN = "qwen/qwen3-coder-30b"
GLM_CTX = 131072
OSS_CTX = 98304

# Attention fields from the local MLX config.json files (see the 2026-11-14
# design reports): GLM-4.5-Air 46 layers x 8 KV heads x 128 dim = 184 KiB per
# token; gpt-oss-120b 18 full + 18 sliding (window 128) layers x 8 x 64.
GLM_CONFIG = {
    "num_hidden_layers": 46, "num_attention_heads": 96, "num_key_value_heads": 8,
    "head_dim": 128, "hidden_size": 4096,
}
OSS_CONFIG = {
    "num_hidden_layers": 36, "num_attention_heads": 64, "num_key_value_heads": 8,
    "head_dim": 64, "hidden_size": 2880, "sliding_window": 128,
    "layer_types": ["sliding_attention", "full_attention"] * 18,
}
QWEN_CONFIG = {
    "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4,
    "head_dim": 128, "hidden_size": 2048,
}
# Catalog weights that round to the logged figures (laplaced.log, 2026-11-09/14).
WEIGHTS_MB = {GLM: 57374, OSS: 62227, QWEN: 16395}
QWEN_CREDIT = 16395 + 1640  # weights + overhead: the broker's idle-victim credit
# Live headroom reconstructed from the 2026-11-14 11:19:32 vm_stat sample.
H_1119 = 44163


def _real_adapter(tmp_path) -> LMStudioAdapter:
    """The production load_cost, fed a fake catalog and on-disk config.json."""
    catalog = []
    for model_id, config in ((GLM, GLM_CONFIG), (OSS, OSS_CONFIG), (QWEN, QWEN_CONFIG)):
        directory = tmp_path / model_id.replace("/", "_")
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config))
        catalog.append({
            "modelKey": model_id, "path": str(directory),
            "sizeBytes": WEIGHTS_MB[model_id] * _MIB,
        })

    async def fake_run(args, timeout):
        assert args == ["ls", "--json"]
        return (0, json.dumps(catalog), "")

    adapter = LMStudioAdapter(
        models_dir=str(tmp_path), parallel_overrides={GLM: 1, OSS: 1, QWEN: 2}
    )
    adapter._run = fake_run
    return adapter


class ArchAdapter(FakeAdapter):
    """FakeAdapter whose load_cost is LMStudioAdapter.load_cost."""

    def __init__(self, tmp_path, **kwargs):
        super().__init__(**kwargs)
        self.real = _real_adapter(tmp_path)

    async def load_cost(self, model_id, context_length):
        return await self.real.load_cost(model_id, context_length)


class Probe:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]


def _idle_qwen():
    return loaded_model(QWEN, size_bytes=16395 * _MIB, status="idle", queued=0,
                        context_length=131072)


def _managed_reaper(adapter, *model_ids):
    reaper = Reaper(adapter)
    for model_id in model_ids:
        reaper.mark_managed(model_id)
    return reaper


# --- ctx_fit arithmetic -------------------------------------------------------


@pytest.mark.asyncio
async def test_ctx_fit_glm_matches_closed_form(tmp_path):
    adapter = _real_adapter(tmp_path)
    broker = ContentionBroker(adapter, Reaper(adapter), headroom_mb=Probe(0), fit_context=True)
    full = await adapter.load_cost(GLM, GLM_CTX)
    assert (full.weights_mb, full.kv_mb, full.overhead_mb, full.total_mb) == (
        57374, 23552, 5738, 86664,
    )
    per_token = 184 * 1024
    for available in (63112 + 1, 68264, 70000, 80000):
        expected = (available - 57374 - 5738) * _MIB // per_token
        assert await broker._fit_context_length(GLM, available, 1, GLM_CTX) == expected
    # 11:19 headroom plus qwen's 18035 MiB credit: short even at context 1.
    assert await broker._fit_context_length(GLM, H_1119 + QWEN_CREDIT, 1, GLM_CTX) is None
    # The measured release that gives ~28k (see PR body).
    assert await broker._fit_context_length(GLM, H_1119 + 24101, 1, GLM_CTX) == 28672
    # Never above the configured context.
    assert await broker._fit_context_length(GLM, 200000, 1, GLM_CTX) == GLM_CTX


@pytest.mark.asyncio
async def test_ctx_fit_gpt_oss_respects_sliding_layers(tmp_path):
    adapter = _real_adapter(tmp_path)
    broker = ContentionBroker(adapter, Reaper(adapter), headroom_mb=Probe(0), fit_context=True)
    assert (await adapter.load_cost(OSS, OSS_CTX)).total_mb == 71911
    per_layer = 2 * 8 * 64 * 2
    full_per_token = 18 * per_layer
    sliding_fixed = 18 * per_layer * 128  # sliding layers stop at the window
    available = H_1119 + 24576
    expected = ((available - 62227 - 6223) * _MIB - sliding_fixed) // full_per_token
    fit = await broker._fit_context_length(OSS, available, 1, OSS_CTX)
    assert fit == expected == 8092
    # Counting sliding layers per token, as if full, would give about half.
    naive = (available - 62227 - 6223) * _MIB // (36 * per_layer)
    assert naive < fit
    # 11:19 headroom plus qwen's credit cannot hold the weights + overhead.
    assert await broker._fit_context_length(OSS, H_1119 + QWEN_CREDIT, 1, OSS_CTX) is None


# --- Change 1: admission at a fitted context ----------------------------------


@pytest.mark.asyncio
async def test_admits_at_reduced_context_when_full_context_does_not_fit(tmp_path, caplog):
    adapter = ArchAdapter(tmp_path)
    broker = ContentionBroker(
        adapter, Reaper(adapter), headroom_mb=Probe(70000), timeout_s=0.1, fit_context=True
    )
    with caplog.at_level(logging.INFO, logger="laplace.broker"):
        decision, ctx = await broker.admit_sized(GLM, GLM_CTX, need_context=20000)
    assert decision == "admit"
    assert 20000 <= ctx < GLM_CTX
    assert ctx == (70000 - 57374 - 5738) * _MIB // (184 * 1024)
    assert broker._reserved_ctx[GLM] == ctx
    line = next(r.getMessage() for r in caplog.records if "decision=admit" in r.getMessage())
    assert f"ctx={ctx}" in line and "need_ctx=20000" in line and f"fit_ctx={ctx}" in line
    assert f"configured_ctx={GLM_CTX}" in line


@pytest.mark.asyncio
async def test_full_context_still_used_when_it_fits(tmp_path):
    adapter = ArchAdapter(tmp_path)
    broker = ContentionBroker(
        adapter, Reaper(adapter), headroom_mb=Probe(90000), timeout_s=0.1, fit_context=True
    )
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=20000) == ("admit", GLM_CTX)


@pytest.mark.asyncio
async def test_refuses_when_even_the_need_does_not_fit(tmp_path):
    adapter = ArchAdapter(tmp_path)
    # 64000 MiB covers weights + overhead (63112) and ~5k tokens, not 20000.
    broker = ContentionBroker(
        adapter, Reaper(adapter), headroom_mb=Probe(64000), timeout_s=0.1, fit_context=True
    )
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=20000) == ("backpressure", None)
    assert GLM not in broker._reserved


@pytest.mark.asyncio
async def test_fit_context_off_keeps_configured_context_behaviour(tmp_path):
    adapter = ArchAdapter(tmp_path)
    broker = ContentionBroker(adapter, Reaper(adapter), headroom_mb=Probe(70000), timeout_s=0.1)
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=20000) == ("backpressure", None)


@pytest.mark.asyncio
async def test_loader_receives_the_chosen_context(tmp_path):
    adapter = ArchAdapter(tmp_path)
    reaper = Reaper(adapter)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=Probe(70000), timeout_s=0.1, fit_context=True
    )
    async with govern(
        broker, reaper, adapter, model=GLM, ctx=GLM_CTX, tier="interactive",
        fail_open=False, need_ctx=20000,
    ) as admission:
        assert admission.decision == "admit"
    fitted = (70000 - 57374 - 5738) * _MIB // (184 * 1024)
    assert adapter.ensure_loaded_calls == [(GLM, fitted)]
    assert adapter.loaded[GLM].context_length == fitted


@pytest.mark.asyncio
async def test_reuse_refuses_instance_smaller_than_the_need(tmp_path):
    adapter = ArchAdapter(
        tmp_path,
        loaded=[loaded_model(GLM, status="idle", queued=0, context_length=16384)],
    )
    probe = Probe(200000)
    broker = ContentionBroker(
        adapter, _managed_reaper(adapter, GLM), headroom_mb=probe, timeout_s=0.1,
        fit_context=True,
    )
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=40000) == ("backpressure", None)
    assert probe.calls == 0  # never priced as a second instance
    assert adapter.ensure_loaded_calls == []


@pytest.mark.asyncio
async def test_reuse_accepts_instance_that_covers_the_need(tmp_path):
    adapter = ArchAdapter(
        tmp_path,
        loaded=[loaded_model(GLM, status="idle", queued=0, context_length=28672)],
    )
    probe = Probe(0)
    reaper = _managed_reaper(adapter, GLM)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=probe, timeout_s=0.1, fit_context=True
    )
    async with govern(
        broker, reaper, adapter, model=GLM, ctx=GLM_CTX, tier="interactive",
        fail_open=False, need_ctx=20000,
    ) as admission:
        assert admission.decision == "admit"
    assert probe.calls == 0
    # ensure_loaded sees the need, which the resident 28672 instance covers.
    assert adapter.ensure_loaded_calls == [(GLM, 20000)]


@pytest.mark.asyncio
async def test_pending_fitted_load_is_reserved_at_the_fitted_context(tmp_path):
    adapter = ArchAdapter(tmp_path)
    broker = ContentionBroker(
        adapter, Reaper(adapter), headroom_mb=Probe(70000), timeout_s=0.1,
        fit_context=True, max_concurrent_lms=4,
    )
    _, ctx = await broker.admit_sized(GLM, GLM_CTX, need_context=20000)
    # A need above the pending load's context must wait for it.
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=ctx + 1) == ("backpressure", None)
    # A need within it rides on the pending load.
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=ctx - 1) == ("admit", ctx - 1)


# --- Change 2: measured eviction ----------------------------------------------


@pytest.mark.asyncio
async def test_measured_eviction_when_credit_below_shortfall_then_recheck(tmp_path, caplog):
    adapter = ArchAdapter(tmp_path, loaded=[_idle_qwen()])
    reaper = _managed_reaper(adapter, QWEN)
    released = H_1119 + 24101
    # decide, request_free baseline, after unload, recheck
    probe = Probe(H_1119, H_1119, released, released)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=probe, timeout_s=1.0,
        fit_context=True, measured_eviction=True,
    )
    with caplog.at_level(logging.INFO, logger="laplace.broker"):
        decision, ctx = await broker.admit_sized(GLM, GLM_CTX, need_context=20000)
    assert adapter.force_unload_calls == [QWEN]
    assert (decision, ctx) == ("admit", 28672)
    messages = [r.getMessage() for r in caplog.records if "load decision" in r.getMessage()]
    assert any("decision=evict-measured" in m and f"evictable_credit_mb={QWEN_CREDIT}" in m
               for m in messages)
    assert "fit_ctx=28672" in messages[-1] and "decision=admit" in messages[-1]


@pytest.mark.asyncio
async def test_without_measured_eviction_credit_below_shortfall_queues(tmp_path):
    adapter = ArchAdapter(tmp_path, loaded=[_idle_qwen()])
    broker = ContentionBroker(
        adapter, _managed_reaper(adapter, QWEN), headroom_mb=Probe(H_1119),
        timeout_s=0.1, fit_context=True,
    )
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=20000) == ("backpressure", None)
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["generating", "in-flight", "unmanaged"])
async def test_measured_eviction_never_takes_busy_or_unmanaged_victims(tmp_path, how):
    victim = _idle_qwen()
    if how == "generating":
        victim = loaded_model(QWEN, size_bytes=16395 * _MIB, status="generating",
                              queued=0, context_length=131072)
    adapter = ArchAdapter(tmp_path, loaded=[victim])
    reaper = Reaper(adapter) if how == "unmanaged" else _managed_reaper(adapter, QWEN)
    if how == "in-flight":
        reaper.acquire(QWEN)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=Probe(H_1119), timeout_s=0.1,
        fit_context=True, measured_eviction=True,
    )
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=20000) == ("backpressure", None)
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_measured_eviction_runs_once_per_admission_no_loop(tmp_path):
    adapter = ArchAdapter(tmp_path, loaded=[_idle_qwen()])
    reaper = _managed_reaper(adapter, QWEN)
    # The eviction frees only 2000 MiB: still short after the recheck.
    probe = Probe(H_1119, H_1119, H_1119 + 2000)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=probe, timeout_s=0.6,
        fit_context=True, measured_eviction=True,
    )
    request = asyncio.create_task(broker.admit_sized(GLM, GLM_CTX, need_context=20000))
    await asyncio.sleep(0.1)
    assert adapter.force_unload_calls == [QWEN]
    # Something reloads the victim; it goes idle; the parked request is woken.
    for _ in range(3):
        adapter.loaded[QWEN] = _idle_qwen()
        await broker.notify_freed()
        await asyncio.sleep(0.05)
    assert await request == ("backpressure", None)
    assert adapter.force_unload_calls == [QWEN]


@pytest.mark.asyncio
async def test_measured_eviction_alone_without_fit_context(tmp_path):
    # Change 2 without Change 1: measured eviction toward the configured context.
    adapter = ArchAdapter(tmp_path, loaded=[_idle_qwen()])
    reaper = _managed_reaper(adapter, QWEN)
    probe = Probe(60000, 60000, 90000, 90000)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=probe, timeout_s=1.0, measured_eviction=True
    )
    # Shortfall 86664 - 60000 = 26664 exceeds the 18035 credit.
    assert await broker.admit(GLM, GLM_CTX) == "admit"
    assert adapter.force_unload_calls == [QWEN]


# --- request need estimate and wiring -----------------------------------------


def test_estimate_need_tokens():
    body = json.dumps({"model": GLM, "messages": [{"role": "user", "content": "x" * 3000}]})
    prompt = -(-len(body.encode()) // 3)
    assert _estimate_need_tokens(body.encode(), 8192) == prompt + 8192
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        sized = json.dumps({"model": GLM, key: 500}).encode()
        assert _estimate_need_tokens(sized, 8192) == -(-len(sized) // 3) + 500
    # Multi-byte text counts by UTF-8 bytes, not characters.
    wide = json.dumps({"input": "中" * 300}, ensure_ascii=False).encode()
    assert _estimate_need_tokens(wide, 1) >= 300
    assert _estimate_need_tokens(b"not json", 100) == 3 + 100
    # Booleans and non-positive limits are not output limits.
    for bad in (True, 0, -5, "10"):
        sized = json.dumps({"max_tokens": bad}).encode()
        assert _estimate_need_tokens(sized, 77) == -(-len(sized) // 3) + 77


def test_fit_context_flags_are_opt_in_and_validated():
    config = LaplacedConfig()
    assert config.fit_context_admission is False
    assert config.measured_eviction is False
    assert config.fit_context_min_output_tokens == 8192
    _, _, broker = build_components(config)
    assert broker._fit_context is False and broker._measured_eviction is False
    on = LaplacedConfig.from_dict({
        "live_memory_admission": True, "fit_context_admission": True,
        "measured_eviction": True, "fit_context_min_output_tokens": 4096,
    })
    _, _, broker = build_components(on)
    assert broker._fit_context is True and broker._measured_eviction is True
    assert on.fit_context_min_output_tokens == 4096
    for key, bad in (
        ("fit_context_admission", "yes"), ("measured_eviction", 1),
        ("fit_context_min_output_tokens", 0), ("fit_context_min_output_tokens", True),
    ):
        with pytest.raises(ValueError, match=key):
            LaplacedConfig.from_dict({key: bad})


@pytest.mark.asyncio
async def test_http_request_loads_at_fitted_context(tmp_path):
    from aiohttp import web

    state = UpstreamState()

    async def chat(request):
        await request.read()
        return web.json_response({"ok": True})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    adapter = ArchAdapter(tmp_path)
    reaper = Reaper(adapter, delay_seconds=300.0)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=Probe(70000), timeout_s=0.5, fit_context=True
    )
    config = LaplacedConfig(
        reaper_sweep=False, fit_context_admission=True,
        fit_context_min_output_tokens=4096, model_context={GLM: GLM_CTX},
    )
    async with make_harness(
        upstream_app=upstream, upstream_state=state, config=config,
        adapter=adapter, reaper=reaper, broker=broker,
    ) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=json.dumps({"model": GLM, "messages": [{"role": "user", "content": "hi"}]}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
    fitted = (70000 - 57374 - 5738) * _MIB // (184 * 1024)
    assert adapter.ensure_loaded_calls == [(GLM, fitted)]
