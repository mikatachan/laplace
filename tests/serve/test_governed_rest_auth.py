from __future__ import annotations

import logging
import sys

import pytest
from aiohttp import web

from laplace_serve.auth import make_verifier
from laplace_serve.config import LaplacedConfig
from laplace_serve.proxy import UpstreamProxy
from tests.serve.harness import UpstreamState, make_harness


CALLER_TOKEN = "caller-fixture-value"
OTHER_TOKEN = "other-caller-fixture-value"
UPSTREAM_KEY = "upstream-fixture-value"


def _config(**overrides) -> LaplacedConfig:
    values = {
        "governed_rest_auth": True,
        "upstream_auth_command": (sys.executable, "-c", f"print({UPSTREAM_KEY!r})"),
        "origins": {"hermes": "interactive", "bot": "interactive", "default": "scheduled"},
        "caller_verifiers": {
            "hermes": make_verifier(CALLER_TOKEN, b"h" * 16),
            "bot": make_verifier(OTHER_TOKEN, b"b" * 16),
        },
        "reaper_sweep": False,
    }
    values.update(overrides)
    return LaplacedConfig(**values)


def _bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


@pytest.mark.asyncio
async def test_default_off_restart_preserves_bare_routes_and_authorization_forwarding():
    seen = []

    async def chat(request):
        seen.append(request.headers.get("Authorization"))
        return web.json_response({"ok": True})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    config = LaplacedConfig.from_dict({"reaper_sweep": False})

    async with make_harness(upstream_app=upstream, config=config) as h:
        response = await h.client.post(
            "/v1/chat/completions",
            json={"model": "m"},
            headers=_bearer("legacy-value"),
        )

    assert config.governed_rest_auth is False
    assert response.status == 200
    assert seen == ["Bearer legacy-value"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "status", "code"),
    [
        ({}, 401, "caller_auth_required"),
        (_bearer("invalid-value"), 401, "caller_auth_required"),
        (_bearer(OTHER_TOKEN), 403, "caller_origin_mismatch"),
    ],
)
async def test_enabled_auth_rejects_missing_invalid_and_cross_origin(headers, status, code):
    async def chat(_request):
        return web.json_response({})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)

    async with make_harness(upstream_app=upstream, config=_config()) as h:
        response = await h.client.post(
            "/hermes/v1/chat/completions", json={"model": "m"}, headers=headers
        )
        assert response.status == status
        assert (await response.json()) == {"error": code}


@pytest.mark.asyncio
async def test_enabled_auth_binds_path_origin_ignores_spoof_headers_and_injects_upstream_key(
    caplog,
):
    seen = []

    async def chat(request):
        seen.append(dict(request.headers))
        return web.json_response({"ok": True})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    caplog.set_level(logging.INFO)

    async with make_harness(upstream_app=upstream, config=_config()) as h:
        response = await h.client.post(
            "/hermes/v1/chat/completions",
            json={"model": "m"},
            headers={
                **_bearer(CALLER_TOKEN),
                "X-Laplace-Origin": "bot",
                "X-Laplace-Priority": "scheduled",
                "X-Laplace-Caller-Authorization": "must-not-forward",
            },
        )

    assert response.status == 200
    assert seen[0]["Authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert "X-Laplace-Caller-Authorization" not in seen[0]
    assert CALLER_TOKEN not in caplog.text
    assert UPSTREAM_KEY not in caplog.text
    assert "admission origin=hermes tier=interactive" in caplog.text


@pytest.mark.asyncio
async def test_enabled_auth_removes_bare_routes_and_protects_models_and_status():
    async def models(_request):
        return web.json_response({"data": []})

    upstream = web.Application()
    upstream.router.add_get("/v1/models", models)

    async with make_harness(upstream_app=upstream, config=_config()) as h:
        bare_post = await h.client.post("/v1/chat/completions", json={"model": "m"})
        bare_get = await h.client.get("/v1/models")
        models_missing = await h.client.get("/hermes/v1/models")
        models_ok = await h.client.get("/hermes/v1/models", headers=_bearer(CALLER_TOKEN))
        ready_missing = await h.client.get("/readyz")
        fleet_missing = await h.client.get("/fleet/status")

    assert bare_post.status == 404
    assert bare_get.status == 404
    assert models_missing.status == 401
    assert models_ok.status == 200
    assert ready_missing.status == 401
    assert fleet_missing.status == 401


@pytest.mark.asyncio
async def test_upstream_401_refreshes_helper_and_retries(monkeypatch):
    supplied = iter(("first-upstream-value", "second-upstream-value"))

    async def fake_helper(self):
        return next(supplied)

    monkeypatch.setattr(UpstreamProxy, "_run_auth_command", fake_helper)
    seen = []

    async def chat(request):
        value = request.headers.get("Authorization")
        seen.append(value)
        if value == "Bearer first-upstream-value":
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response({"ok": True})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)

    async with make_harness(upstream_app=upstream, config=_config()) as h:
        response = await h.client.post(
            "/hermes/v1/chat/completions",
            json={"model": "m"},
            headers=_bearer(CALLER_TOKEN),
        )

    assert response.status == 200
    assert seen == ["Bearer first-upstream-value", "Bearer second-upstream-value"]


@pytest.mark.asyncio
async def test_absent_upstream_credential_fails_startup_with_fixed_error():
    sentinel = "helper-argument-must-not-escape"
    config = _config(
        upstream_auth_command=(sys.executable, "-c", "raise SystemExit(1)", sentinel)
    )
    with pytest.raises(RuntimeError, match="^upstream credential unavailable$") as caught:
        async with make_harness(upstream_app=web.Application(), config=config):
            pass
    assert sentinel not in str(caught.value)


def test_enabled_config_requires_exact_valid_verifiers_and_helper():
    with pytest.raises(ValueError, match="must be a boolean"):
        LaplacedConfig.from_dict({"governed_rest_auth": "false"})

    with pytest.raises(ValueError, match="upstream_auth_command"):
        LaplacedConfig.from_dict({"governed_rest_auth": True})

    with pytest.raises(ValueError, match="exactly one caller verifier"):
        LaplacedConfig.from_dict(
            {
                "governed_rest_auth": True,
                "upstream_auth_command": ["helper"],
                "origins": {"hermes": "interactive", "default": "scheduled"},
            }
        )

    with pytest.raises(ValueError, match="invalid caller verifier"):
        LaplacedConfig.from_dict(
            {
                "governed_rest_auth": True,
                "upstream_auth_command": ["helper"],
                "origins": {"hermes": "interactive", "default": "scheduled"},
                "caller_verifiers": {
                    "hermes": "not-a-verifier",
                    "bot": make_verifier(OTHER_TOKEN, b"b" * 16),
                },
            }
        )
