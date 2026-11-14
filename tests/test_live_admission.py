from __future__ import annotations

import json
import logging

import pytest

from laplace.adapter import LoadCost
from laplace.adapters.lmstudio import (
    LMStudioAdapter,
    kv_cache_bytes,
    kv_layer_specs,
)
from laplace.broker import ContentionBroker
from laplace.memory import VmStatHeadroom, parse_vm_stat
from laplace.reaper import Reaper
from laplace_serve.__main__ import build_components
from laplace_serve.config import LaplacedConfig
from tests.fakes import FakeAdapter, loaded_model

_GIB = 1024 ** 3
_MIB = 1024 ** 2
CTX = 131072

# Attention fields copied from the local MLX config.json files.
QWEN36_27B = {"text_config": {
    "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
    "head_dim": 256, "hidden_size": 5120, "full_attention_interval": 4,
}}
QWEN3_NEXT_80B = {
    "num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 2,
    "head_dim": 256, "hidden_size": 2048, "full_attention_interval": 4,
}
GLM45_AIR = {
    "num_hidden_layers": 46, "num_attention_heads": 96, "num_key_value_heads": 8,
    "head_dim": 128, "hidden_size": 4096,
}
GPT_OSS_120B = {
    "num_hidden_layers": 4, "num_attention_heads": 64, "num_key_value_heads": 8,
    "head_dim": 64, "hidden_size": 2880, "sliding_window": 128,
    "layer_types": ["sliding_attention", "full_attention"] * 2,
}


class CostAdapter(FakeAdapter):
    """FakeAdapter with fixed per-model LoadCost figures."""

    def __init__(self, *, costs: dict[str, LoadCost], **kwargs):
        super().__init__(**kwargs)
        self.costs = costs

    async def load_cost(self, model_id, context_length):
        return self.costs.get(self.base_id(model_id))


def _cost(weights, kv, *, estimate=()):
    return LoadCost(weights, kv, weights // 10, CTX, 2, tuple(estimate))


class Probe:
    """Injectable live-headroom reading."""

    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]


# The 2026-11-07 case: the static table sums to 76000 MB, under the 89088 MB
# auto budget, but live headroom is short and three models are generating.
STATIC = {"hermes/80b": 46000, "bot/a": 10000, "bot/b": 10000, "bot/c": 10000}
COSTS = {
    "hermes/80b": _cost(43000, 6144),   # 24 KiB/token x 131072 x 2
    "bot/a": _cost(16000, 16384),       # 64 KiB/token x 131072 x 2
    "bot/b": _cost(16000, 16384),
    "bot/c": _cost(16000, 16384),
}


def _busy_pool(**kwargs):
    adapter = CostAdapter(
        costs=COSTS,
        footprints=STATIC,
        loaded=[
            loaded_model(mid, size_bytes=10 * _GIB, status="generating", context_length=CTX)
            for mid in ("hermes/80b", "bot/a", "bot/b")
        ],
        **kwargs,
    )
    return adapter


@pytest.mark.asyncio
async def test_static_sum_fits_but_live_headroom_short_backpressures():
    legacy = _busy_pool()
    legacy_broker = ContentionBroker(legacy, Reaper(legacy), budget_mb=89088, timeout_s=0.1)
    assert await legacy_broker.admit("bot/c", CTX) == "admit"  # the bug

    adapter = _busy_pool()
    reaper = Reaper(adapter)
    reaper.acquire("hermes/80b")  # the running Hermes build
    broker = ContentionBroker(
        adapter, reaper, budget_mb=89088, timeout_s=0.1,
        headroom_mb=Probe(20000), max_concurrent_lms=8,
    )
    assert await broker.admit("bot/c", CTX) == "backpressure"
    assert adapter.ensure_loaded_calls == []
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_admits_when_cost_fits_live_headroom(caplog):
    adapter = CostAdapter(costs=COSTS)
    broker = ContentionBroker(adapter, Reaper(adapter), headroom_mb=Probe(40000), timeout_s=0.1)
    with caplog.at_level(logging.INFO, logger="laplace.broker"):
        assert await broker.admit("bot/c", CTX) == "admit"
    line = next(r.getMessage() for r in caplog.records if "load decision" in r.getMessage())
    assert "cost_mb=33984" in line and "basis=derived" in line and "decision=admit" in line


@pytest.mark.asyncio
async def test_inflight_reservation_protects_running_request():
    # 60000 MB free covers bot/c (33984) only if bot/a's KV is not reserved.
    def pool(status):
        return CostAdapter(
            costs=COSTS,
            loaded=[loaded_model("bot/a", size_bytes=16 * _GIB, status=status, context_length=CTX)],
        )

    idle = pool("idle")
    idle_reaper = Reaper(idle)
    idle_reaper.mark_managed("bot/a")
    assert await ContentionBroker(
        idle, idle_reaper, headroom_mb=Probe(45000), timeout_s=0.1
    ).admit("bot/c", CTX) == "admit"

    busy = pool("idle")
    busy_reaper = Reaper(busy)
    busy_reaper.acquire("bot/a")  # in flight through laplaced
    assert await ContentionBroker(
        busy, busy_reaper, headroom_mb=Probe(45000), timeout_s=0.1
    ).admit("bot/c", CTX) == "backpressure"
    assert busy.force_unload_calls == []

    external = pool("generating")  # external tenant, reported by lms ps
    assert await ContentionBroker(
        external, Reaper(external), headroom_mb=Probe(45000), timeout_s=0.1
    ).admit("bot/c", CTX) == "backpressure"


@pytest.mark.asyncio
async def test_pending_admitted_load_counts_against_headroom():
    adapter = CostAdapter(costs=COSTS)
    broker = ContentionBroker(
        adapter, Reaper(adapter), headroom_mb=Probe(50000), timeout_s=0.1, max_concurrent_lms=4
    )
    assert await broker.admit("bot/b", CTX) == "admit"  # reserved, not yet resident
    assert await broker.admit("bot/c", CTX) == "backpressure"


@pytest.mark.asyncio
async def test_evicts_idle_model_then_admits_on_measured_headroom():
    adapter = CostAdapter(
        costs=COSTS,
        loaded=[loaded_model("bot/a", size_bytes=16 * _GIB, status="idle", context_length=CTX)],
    )
    reaper = Reaper(adapter)
    reaper.acquire("bot/a")
    reaper.release("bot/a")
    probe = Probe(20000, 20000, 40000, 40000)  # decide, baseline, after unload, recheck
    broker = ContentionBroker(adapter, reaper, headroom_mb=probe, timeout_s=1.0)
    assert await broker.admit("bot/c", CTX) == "admit"
    assert adapter.force_unload_calls == ["bot/a"]


@pytest.mark.asyncio
async def test_queues_when_eviction_cannot_cover_shortfall():
    adapter = CostAdapter(
        costs=COSTS,
        loaded=[loaded_model("bot/a", size_bytes=16 * _GIB, status="idle", context_length=CTX)],
    )
    reaper = Reaper(adapter)
    reaper.acquire("bot/a")
    reaper.release("bot/a")
    # Shortfall 33984 - 5000 exceeds bot/a's weights + overhead (17600).
    broker = ContentionBroker(adapter, reaper, headroom_mb=Probe(5000), timeout_s=0.1)
    assert await broker.admit("bot/c", CTX) == "backpressure"
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_unreadable_headroom_fails_closed():
    adapter = CostAdapter(costs=COSTS)
    broker = ContentionBroker(adapter, Reaper(adapter), headroom_mb=Probe(None), timeout_s=0.1)
    assert await broker.admit("bot/c", CTX) == "backpressure"


@pytest.mark.asyncio
async def test_resident_model_admits_without_probe():
    adapter = CostAdapter(
        costs=COSTS, loaded=[loaded_model("bot/a", status="generating", context_length=CTX)]
    )
    probe = Probe(0)
    broker = ContentionBroker(adapter, Reaper(adapter), headroom_mb=probe, timeout_s=0.1)
    assert await broker.admit("bot/a", CTX) == "admit"
    assert probe.calls == 0


@pytest.mark.asyncio
async def test_resident_at_smaller_context_is_a_cost_decision():
    adapter = CostAdapter(
        costs=COSTS, loaded=[loaded_model("bot/a", status="idle", context_length=65536)]
    )
    probe = Probe(0)
    broker = ContentionBroker(adapter, Reaper(adapter), headroom_mb=probe, timeout_s=0.1)
    assert await broker.admit("bot/a", CTX) == "backpressure"
    assert probe.calls >= 1


@pytest.mark.asyncio
async def test_larger_context_does_not_ride_on_pending_smaller_reservation():
    # Reviewer reproduction: a 64K admit reserved; a 131K admit of the same model
    # returned admit with no second headroom read.
    adapter = CostAdapter(costs=COSTS)
    probe = Probe(35000)
    broker = ContentionBroker(
        adapter, Reaper(adapter), headroom_mb=probe, timeout_s=0.1, max_concurrent_lms=4
    )
    assert await broker.admit("bot/c", 65536) == "admit"
    assert await broker.admit("bot/c", CTX) == "backpressure"
    assert broker._reserved_ctx["bot/c"] == 65536
    # A smaller context may share the pending load.
    assert await broker.admit("bot/c", 32768) == "admit"
    assert probe.calls == 1


@pytest.mark.asyncio
async def test_live_eviction_skips_model_reported_generating():
    # Reviewer reproduction: a managed resident with lms status "generating"
    # and zero laplace refcount was evicted.
    adapter = CostAdapter(
        costs=COSTS,
        loaded=[loaded_model("victim", size_bytes=16 * _GIB, status="generating",
                             queued=0, context_length=CTX)],
    )
    adapter.costs["victim"] = _cost(16000, 16384)
    reaper = Reaper(adapter)
    reaper.acquire("victim")
    reaper.release("victim")
    assert "victim" in reaper.evictable_models(["victim"])
    # 40000 - 16384 reserved KV leaves a 10368 MB shortfall that evicting it would cover.
    broker = ContentionBroker(adapter, reaper, headroom_mb=Probe(40000), timeout_s=0.1)
    requested: list[int] = []
    real_request_free = reaper.request_free

    async def record(needed_mb, priority=None, **kwargs):
        requested.append(needed_mb)
        return await real_request_free(needed_mb, priority, **kwargs)

    reaper.request_free = record
    assert await broker.admit("bot/c", CTX) == "backpressure"
    assert requested == []  # the broker never counts a busy model as evictable
    reaper.request_free = real_request_free
    assert await reaper.request_free(10_000, headroom_mb=Probe(0)) is False
    assert adapter.force_unload_calls == []


@pytest.mark.asyncio
async def test_footprint_table_fallback_logs_estimate(caplog):
    adapter = LMStudioAdapter(
        footprint_overrides={"org/model": 20000}, parallel_overrides={"org/model": 2}
    )

    async def fake_run(args, timeout):
        return (1, "", "catalog unavailable")

    adapter._run = fake_run
    adapter.list_loaded = lambda: _async([])
    broker = ContentionBroker(adapter, Reaper(adapter), headroom_mb=Probe(200000), timeout_s=0.1)
    with caplog.at_level(logging.INFO, logger="laplace.broker"):
        assert await broker.admit("org/model", 8192) == "admit"
    line = next(r.getMessage() for r in caplog.records if "load decision" in r.getMessage())
    assert "basis=estimate(" in line
    assert "weights=footprint-table" in line and "kv=no-architecture" in line


async def _async(value):
    return value


@pytest.mark.asyncio
async def test_request_free_credits_measured_headroom_not_file_size():
    def pool():
        adapter = FakeAdapter(loaded=[
            loaded_model("a/old", size_bytes=10 * _GIB, status="idle", queued=0),
            loaded_model("b/new", size_bytes=10 * _GIB, status="idle", queued=0),
        ])
        reaper = Reaper(adapter)
        for model_id in ("a/old", "b/new"):
            reaper.acquire(model_id)
            reaper.release(model_id)
        reaper._last_used["a/old"] = 1.0
        reaper._last_used["b/new"] = 9.0
        return adapter, reaper

    # Unloading a/old frees 30 GiB (weights + KV), more than its 10 GiB file.
    adapter, reaper = pool()
    assert await reaper.request_free(20 * 1024, headroom_mb=Probe(1000, 1000 + 30 * 1024)) is True
    assert adapter.force_unload_calls == ["a/old"]

    # The file says 10 GiB but only 2 GiB came back, so keep evicting.
    adapter, reaper = pool()
    probe = Probe(1000, 1000 + 2 * 1024, 1000 + 12 * 1024)
    assert await reaper.request_free(5 * 1024, headroom_mb=probe) is True
    assert adapter.force_unload_calls == ["a/old", "b/new"]


def test_kv_layer_specs_from_architecture():
    # Qwen3.6-27B: 16 of 64 layers are full attention, 4 KV heads x 256 dim.
    specs = kv_layer_specs(QWEN36_27B)
    assert len(specs) == 16
    assert kv_cache_bytes(specs, CTX) * 2 // _MIB == 16384
    assert kv_cache_bytes(kv_layer_specs(QWEN3_NEXT_80B), CTX) * 2 // _MIB == 6144
    # No layer types: every layer is sized as full attention.
    assert kv_cache_bytes(kv_layer_specs(GLM45_AIR), 1) == 46 * 2 * 8 * 128 * 2
    # Sliding layers stop growing at the window.
    sliding = kv_layer_specs(GPT_OSS_120B)
    per_layer = 2 * 8 * 64 * 2
    assert kv_cache_bytes(sliding, 1000) == 2 * per_layer * 128 + 2 * per_layer * 1000


def test_kv_layer_specs_unmodelled_layouts_return_none():
    assert kv_layer_specs({**GLM45_AIR, "kv_lora_rank": 512}) is None
    assert kv_layer_specs({**GPT_OSS_120B, "layer_types": ["mystery"] * 4}) is None
    assert kv_layer_specs({"num_hidden_layers": 4}) is None


@pytest.mark.asyncio
async def test_lmstudio_load_cost_reads_catalog_size_and_config(tmp_path):
    model_dir = tmp_path / "mlx-community" / "Qwen3.6-27B-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(QWEN36_27B))
    gguf_dir = tmp_path / "org" / "embed-GGUF"
    gguf_dir.mkdir(parents=True)
    (gguf_dir / "embed.gguf").write_bytes(b"")
    catalog = [
        {"modelKey": "qwen3.6-27b", "path": "mlx-community/Qwen3.6-27B-4bit",
         "sizeBytes": 16 * 1024 * _MIB},
        {"modelKey": "org/embed", "path": "org/embed-GGUF/embed.gguf", "sizeBytes": 80 * _MIB},
    ]

    async def fake_run(args, timeout):
        assert args == ["ls", "--json"]
        return (0, json.dumps(catalog), "")

    adapter = LMStudioAdapter(
        models_dir=str(tmp_path), parallel_overrides={"qwen3.6-27b": 2, "org/embed": 1}
    )
    adapter._run = fake_run
    cost = await adapter.load_cost("qwen3.6-27b", CTX)
    assert (cost.weights_mb, cost.kv_mb, cost.overhead_mb) == (16384, 16384, 1639)
    assert cost.estimate == ()

    embed = await adapter.load_cost("org/embed", 2048)
    assert embed.estimate == ("kv=no-architecture",)
    assert embed.kv_mb == 256 * 1024 * 2048 // _MIB

    # Hub entries whose path is not on disk resolve via indexedModelIdentifier.
    catalog.append({"modelKey": "google/gemma", "path": "google/gemma",
                    "indexedModelIdentifier": "mlx-community/Qwen3.6-27B-4bit",
                    "sizeBytes": 1024 * _MIB})
    hub = await adapter.load_cost("google/gemma", CTX)
    assert "kv=no-architecture" not in hub.estimate

    unpinned = LMStudioAdapter(models_dir=str(tmp_path))
    unpinned._run = fake_run
    assert "parallel=unconfigured(4)" in (await unpinned.load_cost("qwen3.6-27b", CTX)).estimate
    assert await unpinned.load_cost("not/in-catalog", CTX) is None


@pytest.mark.asyncio
async def test_lmstudio_load_cost_resolves_hub_variant_directory(tmp_path):
    # Reproduces `lms ls --json` for gemma-4-26b: path and indexedModelIdentifier
    # are the hub id; the files live under the variant's real directory.
    real = tmp_path / "lmstudio-community" / "gemma-4-26B-A4B-it-QAT-MLX-4bit"
    real.mkdir(parents=True)
    (real / "config.json").write_text(json.dumps(QWEN36_27B))
    hub = {"modelKey": "google/gemma-4-26b-a4b-qat", "path": "google/gemma-4-26b-a4b-qat",
           "indexedModelIdentifier": "google/gemma-4-26b-a4b-qat",
           "variants": ["google/gemma-4-26b-a4b-qat@4bit"],
           "selectedVariant": "google/gemma-4-26b-a4b-qat@4bit", "sizeBytes": 1024 * _MIB}
    variants = [
        {"modelKey": "google/gemma-4-26b-a4b-qat@8bit",
         "indexedModelIdentifier": "google/gemma-4-26b-a4b-qat@other/Missing-8bit"},
        {"modelKey": "google/gemma-4-26b-a4b-qat@4bit",
         "indexedModelIdentifier": "google/gemma-4-26b-a4b-qat@"
                                   "lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit"},
    ]

    async def fake_run(args, timeout):
        if args == ["ls", "--json"]:
            return (0, json.dumps([hub]), "")
        assert args == ["ls", "google/gemma-4-26b-a4b-qat", "--json"]
        return (0, json.dumps(variants), "")

    adapter = LMStudioAdapter(models_dir=str(tmp_path),
                              parallel_overrides={"google/gemma-4-26b-a4b-qat": 1})
    adapter._run = fake_run
    cost = await adapter.load_cost("google/gemma-4-26b-a4b-qat", CTX)
    assert "kv=no-architecture" not in cost.estimate
    assert cost.kv_mb == 8192  # QWEN36_27B config at CTX, parallel 1

    # Variant lookup yielding nothing falls back to the reported identities.
    variants.clear()
    fallback = LMStudioAdapter(models_dir=str(tmp_path),
                               parallel_overrides={"google/gemma-4-26b-a4b-qat": 1})
    fallback._run = fake_run
    assert "kv=no-architecture" in (await fallback.load_cost("google/gemma-4-26b-a4b-qat", CTX)).estimate


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  {free}.
Pages active:                                 {active}.
Pages inactive:                               {inactive}.
Pages speculative:                             {speculative}.
Pages wired down:                            {wired}.
"Translation faults":                    18296142287.
File-backed pages:                            {file}.
Pages occupied by compressor:                  {compressor}.
"""


def _vm(free=1024, active=0, inactive=512, speculative=64, file=256, wired=0, compressor=0):
    return VM_STAT.format(
        free=free, active=active, inactive=inactive, speculative=speculative,
        file=file, wired=wired, compressor=compressor,
    )


def test_parse_vm_stat():
    rows = parse_vm_stat(_vm())
    assert rows["page_size"] == 16384
    assert rows["Pages free"] == 1024 and rows["Translation faults"] == 18296142287


@pytest.mark.asyncio
async def test_vm_stat_headroom_counts_file_cache_and_compressor_growth():
    # File pool = min(100 + 512, 600 - 64) = 536; default margin credits 509 pages.
    state = dict(active=100, inactive=512, speculative=64, file=600)
    outputs = {
        "vm_stat": [_vm(compressor=100, **state), _vm(compressor=100 + 640, **state)],
        "sysctl": ["0\n", "0\n"],
    }

    async def run(args):
        return outputs[args[0]].pop(0)

    probe = VmStatHeadroom(run=run)
    assert await probe() == (1024 + 64 + 482) * 16384 // _MIB == 24
    # 640 pages of compressor growth (10 MB) is subtracted as pressure.
    assert await probe() == 14


@pytest.mark.asyncio
async def test_vm_stat_headroom_measured_sample_and_margin():
    text = _vm(
        free=1526869, active=1687597, inactive=2408129,
        speculative=261194, file=2565333, wired=354190,
    )

    async def run(args):
        return text if args[0] == "vm_stat" else "0\n"

    assert await VmStatHeadroom(run=run)() == 62140


@pytest.mark.asyncio
async def test_vm_stat_headroom_does_not_double_count_speculative_and_margin_is_exact():
    # 6464 file-backed pages include 64 speculative; the candidate is 6400 (100 MB).
    text = _vm(free=0, active=6400, inactive=0, speculative=64, file=6464)

    async def run(args):
        return text if args[0] == "vm_stat" else "0\n"

    full = await VmStatHeadroom(run=run, file_cache_reclaim_margin=0)()
    margin = await VmStatHeadroom(run=run, file_cache_reclaim_margin=0.10)()
    file_credit_mb = 6400 * 16384 // _MIB
    assert full == (64 + 6400) * 16384 // _MIB == 101
    assert full - margin == file_credit_mb * 0.10 == 10


@pytest.mark.asyncio
async def test_vm_stat_headroom_capped_by_wired_limit_and_fails_closed():
    async def run(args):
        return _vm(free=10 * 65536, wired=64 * 1000) if args[0] == "vm_stat" else "1100\n"

    assert await VmStatHeadroom(run=run)() == 1100 - 1000

    async def broken(args):
        return None

    assert await VmStatHeadroom(run=broken)() is None


def test_live_memory_admission_is_opt_in():
    _, _, default = build_components(LaplacedConfig())
    assert default._headroom is None
    _, _, live = build_components(LaplacedConfig.from_dict({"live_memory_admission": True}))
    assert isinstance(live._headroom, VmStatHeadroom)
    assert live._headroom._file_cache_reclaim_margin == 0.05
    with pytest.raises(ValueError):
        LaplacedConfig.from_dict({"live_memory_admission": "yes"})
    assert LaplacedConfig().file_cache_reclaim_margin == 0.05
    for margin in (-0.01, 1.01):
        with pytest.raises(ValueError, match="file_cache_reclaim_margin"):
            LaplacedConfig.from_dict({"file_cache_reclaim_margin": margin})
