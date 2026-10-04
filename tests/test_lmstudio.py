from __future__ import annotations

import pytest

from laplace.adapter import ModelLoadError
from laplace.adapters.lmstudio import LMStudioAdapter


@pytest.mark.asyncio
async def test_footprint_override_short_circuits_subprocess():
    adapter = LMStudioAdapter(footprint_overrides={"qwen3-next-80b-a3b-thinking": 48000})

    async def fail_run(args, timeout):
        raise AssertionError("lms subprocess must not run when an override matches")

    adapter._run = fail_run
    assert await adapter.footprint_mb("qwen3-next-80b-a3b-thinking") == 48000


@pytest.mark.asyncio
async def test_footprint_override_matches_base_id():
    adapter = LMStudioAdapter(footprint_overrides={"qwen3.6-27b-mlx": 31500})

    async def fail_run(args, timeout):
        raise AssertionError("lms subprocess must not run when an override matches")

    adapter._run = fail_run
    # A versioned instance id resolves to the same base id as the override key.
    assert await adapter.footprint_mb("qwen3.6-27b-mlx:2") == 31500


@pytest.mark.asyncio
async def test_footprint_absent_override_falls_through():
    adapter = LMStudioAdapter(footprint_overrides={"other-model": 1})
    ran: list[list[str]] = []

    async def fake_run(args, timeout):
        ran.append(args)
        return (1, "", "err")

    adapter._run = fake_run
    # No override for this id: ps/ls/estimate paths are exercised and fail -> None.
    assert await adapter.footprint_mb("qwen/qwen3-coder-30b") is None
    assert ran


@pytest.mark.asyncio
async def test_load_timeout_param_passed_to_run():
    adapter = LMStudioAdapter(load_timeout_s=300.0)
    seen: dict[str, float] = {}
    loaded = False

    async def fake_run(args, timeout):
        if args == ["ls", "--json"]:
            return (0, '[{"modelKey": "some/model"}]', "")
        nonlocal loaded
        if args[:2] == ["ps", "--json"]:
            return (0, '[{"identifier": "some/model", "contextLength": 4096}]' if loaded else "[]", "")
        seen["timeout"] = timeout
        loaded = True
        return (0, "", "")

    adapter._run = fake_run
    await adapter.ensure_loaded("some/model", 4096)
    assert seen["timeout"] == 300.0


@pytest.mark.asyncio
async def test_load_timeout_defaults_to_180():
    adapter = LMStudioAdapter()
    seen: dict[str, float] = {}
    loaded = False

    async def fake_run(args, timeout):
        if args == ["ls", "--json"]:
            return (0, '[{"modelKey": "some/model"}]', "")
        nonlocal loaded
        if args[:2] == ["ps", "--json"]:
            return (0, '[{"identifier": "some/model", "contextLength": 4096}]' if loaded else "[]", "")
        seen["timeout"] = timeout
        loaded = True
        return (0, "", "")

    adapter._run = fake_run
    await adapter.ensure_loaded("some/model", 4096)
    assert seen["timeout"] == 180.0


@pytest.mark.asyncio
async def test_ensure_loaded_raises_when_load_command_fails():
    adapter = LMStudioAdapter()

    async def fake_run(args, timeout):
        if args == ["ls", "--json"]:
            return (0, '[{"modelKey": "some/model"}]', "")
        if args[:2] == ["ps", "--json"]:
            return (0, "[]", "")
        return (1, "", "load failed")

    adapter._run = fake_run
    with pytest.raises(ModelLoadError, match="load failed"):
        await adapter.ensure_loaded("some/model", 4096)


@pytest.mark.asyncio
async def test_ensure_loaded_raises_when_model_never_appears():
    adapter = LMStudioAdapter(load_timeout_s=0.0)
    calls = {"ps": 0}

    async def fake_run(args, timeout):
        if args == ["ls", "--json"]:
            return (0, '[{"modelKey": "some/model"}]', "")
        if args[:2] == ["ps", "--json"]:
            calls["ps"] += 1
            return (0, "[]", "")
        return (0, "", "")

    adapter._run = fake_run
    with pytest.raises(ModelLoadError, match="did not become resident"):
        await adapter.ensure_loaded("some/model", 4096)
    assert calls["ps"] == 3  # fast path, locked recheck, post-load verification
