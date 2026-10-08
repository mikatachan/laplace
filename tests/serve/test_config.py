from __future__ import annotations

import pytest

from laplace_serve.config import LaplacedConfig, missing_context_ids
from laplace_serve.__main__ import build_components


def test_defaults_match_plan():
    config = LaplacedConfig.from_dict({})
    assert config.port == 4242
    assert config.upstream == "http://127.0.0.1:1234"
    assert config.max_concurrent == 2
    assert config.default_context_length == 64000
    assert config.tier_for("hermes") == "interactive"
    assert config.tier_for("bot") == "interactive"
    assert config.tier_for("default") == "scheduled"


def test_budget_zero_means_auto():
    assert LaplacedConfig.from_dict({"budget_mb": 0}).budget_mb_or_none() is None
    assert LaplacedConfig.from_dict({"budget_mb": 87040}).budget_mb_or_none() == 87040


def test_full_toml_parses():
    data = {
        "bind": "127.0.0.1",
        "port": 4242,
        "upstream": "http://127.0.0.1:1234",
        "budget_mb": 0,
        "max_concurrent": 2,
        "admit_timeout_s": 120,
        "fail_open": True,
        "reaper_sweep": True,
        "sweep_interval_s": 60,
        "load_timeout_s": 300,
        "load_grace_s": 17,
        "keep_loaded": ["nomic-embed-text"],
        "default_context_length": 64000,
        "log_rotate_mb": 10,
        "log_backups": 5,
        "origins": {"hermes": "interactive", "openclaw": "interactive", "default": "scheduled"},
        "model_context": {"qwen3-next-80b-a3b-thinking": 131072, "nomic-embed-text": 2048},
        "model_footprint_mb": {"qwen3-next-80b-a3b-thinking": 48000},
        "model_parallel": {"qwen3-next-80b-a3b-thinking": 1},
    }
    config = LaplacedConfig.from_dict(data)
    assert config.keep_loaded == ("nomic-embed-text",)
    assert config.model_context["qwen3-next-80b-a3b-thinking"] == 131072
    assert config.model_footprint_mb["qwen3-next-80b-a3b-thinking"] == 48000
    assert config.model_parallel["qwen3-next-80b-a3b-thinking"] == 1
    assert config.load_grace_s == 17


def test_toml_loader_accepts_load_grace_s(tmp_path):
    path = tmp_path / "laplaced.toml"
    path.write_text("load_grace_s = 23\n")

    assert LaplacedConfig.from_toml_file(str(path)).load_grace_s == 23


def test_bad_tier_rejected():
    with pytest.raises(ValueError):
        LaplacedConfig.from_dict({"origins": {"hermes": "urgent"}})


def test_bad_footprint_type_rejected():
    with pytest.raises(ValueError):
        LaplacedConfig.from_dict({"model_footprint_mb": {"m": "lots"}})


def test_bad_parallel_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        LaplacedConfig.from_dict({"model_parallel": {"m": 0}})


def test_missing_context_ids_lists_gaps():
    config = LaplacedConfig.from_dict({"model_context": {"a": 4096, "b": 8192}})
    assert missing_context_ids(["a", "b", "c", "d"], config) == ["c", "d"]
    assert missing_context_ids(["a", "b"], config) == []


def test_canonical_model_id_resolves_unambiguous_bare_id():
    config = LaplacedConfig.from_dict(
        {"model_context": {"qwen/qwen3-coder-30b": 131072}}
    )

    assert config.canonical_model_id("qwen3-coder-30b") == "qwen/qwen3-coder-30b"


def test_canonical_model_id_preserves_ambiguous_bare_id():
    config = LaplacedConfig.from_dict(
        {"model_context": {"org-a/model": 4096, "org-b/model": 8192}}
    )

    assert config.canonical_model_id("model") == "model"


def test_canonical_model_id_warns_for_unconfigured_bare_id(caplog):
    config = LaplacedConfig.from_dict({"model_context": {"qwen/configured": 8192}})

    assert config.canonical_model_id("qwen3.6-27b-mlx") == "qwen3.6-27b-mlx"
    assert "unconfigured model id 'qwen3.6-27b-mlx'" in caplog.text


def test_canonical_model_id_warns_for_ambiguous_bare_id(caplog):
    config = LaplacedConfig.from_dict(
        {"model_context": {"a/model": 8192, "b/model": 16384}}
    )

    assert config.canonical_model_id("model") == "model"
    assert "ambiguous bare model id 'model'" in caplog.text


@pytest.mark.asyncio
async def test_footprint_overrides_plumb_to_adapter(monkeypatch):
    config = LaplacedConfig.from_dict(
        {"model_footprint_mb": {"qwen3-next-80b-a3b-thinking": 48000}}
    )
    adapter, _reaper, _broker = build_components(config)

    async def _boom(*_args, **_kwargs):
        raise AssertionError("override must short-circuit before any lms subprocess")

    monkeypatch.setattr(adapter, "_run", _boom)
    assert await adapter.footprint_mb("qwen3-next-80b-a3b-thinking") == 48000


def test_load_grace_default_and_override_reach_adapter():
    default, _, _ = build_components(LaplacedConfig.from_dict({}))
    assert default._load_grace_s == 90
    custom, _, _ = build_components(LaplacedConfig.from_dict({"load_grace_s": 17}))
    assert custom._load_grace_s == 17


def test_model_parallel_reaches_adapter_only_when_configured():
    default, _, _ = build_components(LaplacedConfig.from_dict({}))
    assert default._parallel_overrides == {}
    custom, _, _ = build_components(
        LaplacedConfig.from_dict({"model_parallel": {"org/model": 1}})
    )
    assert custom._parallel_overrides == {"org/model": 1}
