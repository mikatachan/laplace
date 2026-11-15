"""Fit-sized load context and measured eviction (live-memory admission)."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from laplace.adapter import LoadCost
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
    # A need within it rides on the pending load, at the pending load's context.
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=ctx - 1) == ("admit", ctx)


@pytest.mark.asyncio
async def test_follower_of_pending_load_loads_at_the_pending_context(tmp_path):
    # PR 28 review: a smaller follower that wins the adapter lock must load at
    # the pending fitted context, or the larger admitted request is refused as
    # a duplicate load against the smaller instance.
    adapter = ArchAdapter(tmp_path)
    reaper = Reaper(adapter)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=Probe(70000), timeout_s=0.1,
        fit_context=True, max_concurrent_lms=4,
    )
    decision, larger_ctx = await broker.admit_sized(GLM, GLM_CTX, need_context=20000)
    assert decision == "admit"
    async with govern(
        broker, reaper, adapter, model=GLM, ctx=GLM_CTX, tier="interactive",
        fail_open=False, need_ctx=10000,
    ) as admission:
        assert admission.decision == "admit"
    assert adapter.ensure_loaded_calls == [(GLM, larger_ctx)]
    assert broker._reserved_ctx[GLM] == larger_ctx


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
async def test_no_eviction_of_any_kind_after_a_measured_eviction(tmp_path):
    # PR 28 review repro: the measured eviction frees nothing and the request
    # parks. After the wake, headroom is 50,000 MiB, so the same credit now
    # covers the smaller shortfall; the ordinary branch must not evict again.
    adapter = ArchAdapter(tmp_path, loaded=[_idle_qwen()])
    reaper = _managed_reaper(adapter, QWEN)
    probe = Probe(H_1119, H_1119, H_1119, H_1119, 50_000)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=probe, timeout_s=0.5,
        fit_context=True, measured_eviction=True,
    )
    request = asyncio.create_task(broker.admit_sized(GLM, GLM_CTX, need_context=20000))
    while not adapter.force_unload_calls or not broker._waiters:
        await asyncio.sleep(0.01)
    adapter.loaded[QWEN] = _idle_qwen()
    await broker.notify_freed()
    assert await request == ("backpressure", None)
    assert adapter.force_unload_calls == [QWEN]


@pytest.mark.asyncio
async def test_without_measured_eviction_ordinary_eviction_repeats_across_wakes(tmp_path):
    # Flag off keeps main's behaviour: each wake may evict again when the
    # credit covers the shortfall.
    adapter = ArchAdapter(tmp_path, loaded=[_idle_qwen()])
    reaper = _managed_reaper(adapter, QWEN)
    probe = Probe(50_000)
    broker = ContentionBroker(
        adapter, reaper, headroom_mb=probe, timeout_s=0.5, fit_context=True
    )
    request = asyncio.create_task(broker.admit_sized(GLM, GLM_CTX, need_context=20000))
    while not adapter.force_unload_calls or not broker._waiters:
        await asyncio.sleep(0.01)
    adapter.loaded[QWEN] = _idle_qwen()
    await broker.notify_freed()
    assert await request == ("backpressure", None)
    assert adapter.force_unload_calls == [QWEN, QWEN]


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
    from laplace_serve.app import _TEMPLATE_ALLOWANCE_TOKENS as allowance

    body = json.dumps({"model": GLM, "messages": [{"role": "user", "content": "x" * 3000}]})
    prompt = len(body.encode()) + allowance
    assert _estimate_need_tokens(body.encode(), 8192) == prompt + 8192
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        sized = json.dumps({"model": GLM, key: 500}).encode()
        assert _estimate_need_tokens(sized, 8192) == len(sized) + allowance + 500
    # Multi-byte text counts by UTF-8 bytes, not characters.
    wide = json.dumps({"input": "中" * 300}, ensure_ascii=False).encode()
    assert _estimate_need_tokens(wide, 1) >= 900
    assert _estimate_need_tokens(b"not json", 100) == 8 + allowance + 100
    # Booleans and non-positive limits are not output limits.
    for bad in (True, 0, -5, "10"):
        sized = json.dumps({"max_tokens": bad}).encode()
        assert _estimate_need_tokens(sized, 77) == len(sized) + allowance + 77


def test_estimate_need_covers_token_dense_prompt():
    # PR 28 review repro: 12,000 random alphanumerics. The GLM-4.5-Air
    # tokenizer with its chat template counts 8,680 prompt tokens (measured
    # offline with transformers by the reviewer); bytes / 3 gave 4,025.
    import random
    import string

    rng = random.Random(28)
    content = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(12_000))
    body = json.dumps(
        {"model": "glm", "messages": [{"role": "user", "content": content}], "max_tokens": 1},
        separators=(",", ":"),
    ).encode()
    assert len(body) == 12072
    assert _estimate_need_tokens(body, 8192) >= 8680 + 1


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


# --- fit_context_best_effort (GLM canary, 2026-11-14 23:53) --------------------
# Second turn: need 75533 costs 76685 MB; the empty pool frees 75703 MB
# (laplaced.log 11580). The first turn's GLM copy is resident, idle, at 69459.

GLM_FIRST_CTX = 69459
GLM_NEED = 75533
POOL_MB = 75703
GLM_CAP = 70071  # largest ctx with load_cost <= POOL_MB (75703 MB)


def _idle_glm(ctx=GLM_FIRST_CTX, status="idle"):
    return loaded_model(GLM, size_bytes=57374 * _MIB, status=status, queued=0,
                        context_length=ctx)


def _best_effort_broker(adapter, reaper, probe, *, best_effort=True, timeout_s=1.0):
    return ContentionBroker(
        adapter, reaper, headroom_mb=probe, timeout_s=timeout_s,
        fit_context=True, measured_eviction=True, best_effort=best_effort,
    )


@pytest.mark.asyncio
async def test_best_effort_capacity_cap_matches_incident_numbers(tmp_path):
    adapter = _real_adapter(tmp_path)
    assert (await adapter.load_cost(GLM, GLM_NEED)).total_mb == 76685
    assert (await adapter.load_cost(GLM, GLM_CAP)).total_mb <= POOL_MB
    assert (await adapter.load_cost(GLM, GLM_CAP + 1)).total_mb > POOL_MB


@pytest.mark.asyncio
async def test_best_effort_incident_replaces_idle_copy_and_admits_at_capacity(tmp_path, caplog):
    adapter = ArchAdapter(tmp_path, loaded=[_idle_glm()])
    reaper = _managed_reaper(adapter, GLM)
    # Headroom with the copy resident, then after its unload.
    probe = Probe(POOL_MB - 75593, POOL_MB)
    broker = _best_effort_broker(adapter, reaper, probe)
    with caplog.at_level(logging.INFO, logger="laplace.broker"):
        result = await broker.admit_sized(GLM, GLM_CTX, need_context=GLM_NEED)
    assert result == ("admit", GLM_CAP)
    assert adapter.force_unload_calls == [GLM]
    messages = [r.getMessage() for r in caplog.records]
    assert any("reason=need-exceeds-capacity" in m and f"capped_ctx={GLM_CAP}" in m
               for m in messages)
    assert any("decision=replace" in m for m in messages)


@pytest.mark.asyncio
async def test_best_effort_nothing_resident_admits_at_fitted_capacity(tmp_path):
    adapter = ArchAdapter(tmp_path)
    broker = _best_effort_broker(adapter, Reaper(adapter), Probe(POOL_MB))
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=GLM_NEED) == ("admit", GLM_CAP)
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_best_effort_admits_onto_copy_at_capacity(tmp_path, caplog):
    adapter = ArchAdapter(tmp_path, loaded=[_idle_glm(ctx=GLM_CAP)])
    reaper = _managed_reaper(adapter, GLM)
    broker = _best_effort_broker(adapter, reaper, Probe(0))
    with caplog.at_level(logging.INFO, logger="laplace.broker"):
        result = await broker.admit_sized(GLM, GLM_CTX, need_context=GLM_NEED)
    assert result == ("admit", GLM_CAP)
    assert adapter.force_unload_calls == []
    assert any("reason=need-capped-to-capacity" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_best_effort_replaces_idle_smaller_copy_when_need_is_achievable(tmp_path):
    adapter = ArchAdapter(tmp_path, loaded=[_idle_glm(ctx=8192)])
    reaper = _managed_reaper(adapter, GLM)
    broker = _best_effort_broker(adapter, reaper, Probe(200_000))
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=20000) == ("admit", GLM_CTX)
    assert adapter.force_unload_calls == [GLM]


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["generating", "in-flight", "unmanaged"])
async def test_best_effort_never_replaces_busy_or_unmanaged_copy(tmp_path, how):
    copy = _idle_glm(ctx=8192, status="generating" if how == "generating" else "idle")
    adapter = ArchAdapter(tmp_path, loaded=[copy])
    reaper = Reaper(adapter) if how == "unmanaged" else _managed_reaper(adapter, GLM)
    if how == "in-flight":
        reaper.acquire(GLM)
    broker = _best_effort_broker(adapter, reaper, Probe(200_000), timeout_s=0.1)
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=20000) == ("backpressure", None)
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_best_effort_off_incident_still_queues(tmp_path):
    adapter = ArchAdapter(tmp_path, loaded=[_idle_glm()])
    reaper = _managed_reaper(adapter, GLM)
    broker = _best_effort_broker(
        adapter, reaper, Probe(POOL_MB - 75593, POOL_MB), best_effort=False, timeout_s=0.1,
    )
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=GLM_NEED) == ("backpressure", None)
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_best_effort_off_nothing_resident_still_queues(tmp_path):
    adapter = ArchAdapter(tmp_path)
    broker = _best_effort_broker(
        adapter, Reaper(adapter), Probe(POOL_MB), best_effort=False, timeout_s=0.1,
    )
    assert await broker.admit_sized(GLM, GLM_CTX, need_context=GLM_NEED) == ("backpressure", None)


def test_best_effort_flag_is_opt_in_validated_and_wired():
    assert LaplacedConfig().fit_context_best_effort is False
    _, _, broker = build_components(LaplacedConfig())
    assert broker._best_effort is False
    _, _, broker = build_components(LaplacedConfig.from_dict({
        "live_memory_admission": True, "fit_context_admission": True,
        "fit_context_best_effort": True,
    }))
    assert broker._best_effort is True
    with pytest.raises(ValueError, match="fit_context_best_effort"):
        LaplacedConfig.from_dict({"fit_context_best_effort": "yes"})


# Two admissions that pick the same adopted idle copy must unload it once.
# The adopted-model live recheck awaits, so the claim has to precede it.

class _PausedAdoptedAdapter(FakeAdapter):
    def __init__(self):
        super().__init__(loaded=[loaded_model(
            "glm", size_bytes=1_000 * _MIB, status="idle", queued=0, context_length=8192,
        )])
        self.list_calls = 0
        self.guard_arrivals = 0
        self.guards_ready = asyncio.Event()
        self.replace_calls = []

    async def load_cost(self, model_id, context_length):
        return LoadCost(
            weights_mb=1_000, kv_mb=(context_length or 0) // 100, overhead_mb=100,
            context_length=context_length, parallel=1,
        )

    async def list_loaded(self):
        self.list_calls += 1
        # Call 1 adopts. Calls 2 and 4 are the brokers' selection snapshots.
        # Calls 3 and 5 would be the adopted-model live rechecks; hold the first
        # until a second arrives (or a few loop turns pass, once it is fixed).
        if self.list_calls in (3, 5):
            self.guard_arrivals += 1
            if self.guard_arrivals == 2:
                self.guards_ready.set()
            for _ in range(100):
                if self.guards_ready.is_set():
                    break
                await asyncio.sleep(0)
        return list(self.loaded.values())

    async def force_unload(self, model_id, reason=""):
        self.replace_calls.append((model_id, reason))
        # Keep the first unload in progress long enough for a second caller.
        if len(self.replace_calls) == 1:
            for _ in range(100):
                if len(self.replace_calls) == 2:
                    break
                await asyncio.sleep(0)
        # A successful unload removes the copy, as LM Studio does.
        return await super().force_unload(model_id, reason=reason)


@pytest.mark.asyncio
async def test_best_effort_concurrent_replace_unloads_adopted_copy_once():
    adapter = _PausedAdoptedAdapter()
    reaper = Reaper(adapter)
    assert reaper.adopt_resident(await adapter.list_loaded()) == ["glm"]
    broker = _best_effort_broker(adapter, reaper, Probe(200_000), timeout_s=0.02)

    results = await asyncio.wait_for(asyncio.gather(
        broker.admit_sized("glm", 100_000, need_context=20_000),
        broker.admit_sized("glm", 100_000, need_context=20_000),
    ), timeout=5)

    assert len(adapter.replace_calls) == 1, adapter.replace_calls
    # The loser queues behind the unload (no loop, no hang), then is covered by
    # the winner's load at the fitted context.
    assert results == [("admit", 100_000), ("admit", 100_000)]
    assert not reaper.is_unloading("glm")


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["guard-blocks", "unload-raises"])
async def test_replace_releases_unloading_claim_on_every_exit(how):
    adapter = FakeAdapter(loaded=[loaded_model(
        "glm", status="busy" if how == "guard-blocks" else "idle", queued=0,
    )])
    reaper = Reaper(adapter)
    reaper.adopt_resident(await adapter.list_loaded())
    if how == "unload-raises":
        async def boom(model_id, *, reason=""):
            raise RuntimeError("lms unload failed")
        adapter.force_unload = boom
        with pytest.raises(RuntimeError):
            await reaper.unload_for_replace("glm")
    else:
        assert await reaper.unload_for_replace("glm") is False
    assert not reaper.is_unloading("glm")


# --- best-effort cap and admission share one headroom reading (2026-11-15 02:59)
# After the replace unload, _capacity_cap read 76584 MB and capped need 114331 at
# 74974 (cost 76584). Admission then read headroom again, got 76583, and queued
# with shortfall_mb=1 and nothing resident (laplaced.log 02:59:51).

GLM_CANARY3_FIRST_CTX = 69904
GLM_CANARY3_NEED = 114331
GLM_CANARY3_CAP = 74974


@pytest.mark.asyncio
async def test_best_effort_cap_and_admission_use_the_same_headroom_reading(tmp_path, caplog):
    adapter = ArchAdapter(tmp_path, loaded=[_idle_glm(ctx=GLM_CANARY3_FIRST_CTX)])
    assert (await adapter.load_cost(GLM, GLM_CANARY3_FIRST_CTX)).total_mb == 75673
    cap_cost = await adapter.load_cost(GLM, GLM_CANARY3_CAP)
    assert (cap_cost.weights_mb, cap_cost.kv_mb, cap_cost.overhead_mb) == (57374, 13472, 5738)
    assert cap_cost.total_mb == 76584
    assert (await adapter.load_cost(GLM, GLM_CANARY3_CAP + 1)).total_mb > 76584
    reaper = _managed_reaper(adapter, GLM)
    # Copy resident: the full need (83656) fits headroom + the copy's 75673, so
    # replace without a cap (log line 02:59:48). After the unload, two readings
    # 1 MB apart, as the cap and admission lines 7 ms apart show.
    assert (await adapter.load_cost(GLM, GLM_CANARY3_NEED)).total_mb == 83656
    probe = Probe(83656 - 75673, 76584, 76583)
    broker = _best_effort_broker(adapter, reaper, probe, timeout_s=0.1)
    with caplog.at_level(logging.INFO, logger="laplace.broker"):
        result = await broker.admit_sized(GLM, GLM_CTX, need_context=GLM_CANARY3_NEED)
    assert result == ("admit", GLM_CANARY3_CAP)
    assert adapter.force_unload_calls == [GLM]
    messages = [r.getMessage() for r in caplog.records]
    assert any(f"need_ctx={GLM_CANARY3_NEED} decision=replace" in m for m in messages)
    assert not any("decision=queue" in m for m in messages), messages
    assert any(f"ctx={GLM_CANARY3_CAP} " in m and "decision=admit" in m
               and "headroom_mb=76584" in m for m in messages), messages


@pytest.mark.asyncio
async def test_best_effort_cap_recomputed_on_measured_headroom_after_short_eviction(tmp_path):
    # The cap counts the idle victim's credit; its unload frees less than that.
    # The next pass must re-cap against the measured reading and load, not
    # queue on the first cap with nothing left to evict.
    adapter = ArchAdapter(tmp_path, loaded=[_idle_qwen()])
    reaper = _managed_reaper(adapter, QWEN)
    # The first cap, on 60000 + the credit, is larger than the measured one.
    assert (await adapter.load_cost(GLM, GLM_CANARY3_CAP + 1)).total_mb <= 60000 + QWEN_CREDIT
    # decide, request_free baseline, after unload, recheck, any later reading
    probe = Probe(60000, 60000, 76584, 76584, 76583)
    broker = _best_effort_broker(adapter, reaper, probe, timeout_s=0.1)
    result = await broker.admit_sized(GLM, GLM_CTX, need_context=GLM_CANARY3_NEED)
    assert adapter.force_unload_calls == [QWEN]
    assert result == ("admit", GLM_CANARY3_CAP)
