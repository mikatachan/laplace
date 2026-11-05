from __future__ import annotations

import sys

import pytest
from aiohttp import web

from laplace_serve.auth import make_verifier
from laplace_serve.config import LaplacedConfig
from tests.serve.harness import make_harness


CALLER = "probe-caller-fixture"
UPSTREAM = "probe-upstream-fixture"
AUTH = {"Authorization": f"Bearer {CALLER}"}


def _enabled_config() -> LaplacedConfig:
    return LaplacedConfig(
        governed_rest_auth=True,
        upstream_auth_command=(sys.executable, "-c", f"print({UPSTREAM!r})"),
        origins={"monitor": "scheduled", "default": "scheduled"},
        caller_verifiers={"monitor": make_verifier(CALLER, b"m" * 16)},
        reaper_sweep=False,
    )


async def _true() -> bool:
    return True


async def _ps_ok() -> bool:
    return True


@pytest.mark.asyncio
async def test_probe_routes_absent_when_governed_auth_default_is_off():
    upstream = web.Application()
    async with make_harness(upstream_app=upstream) as h:
        for path in ("/ops/upstream-listener", "/ops/upstream-api", "/ops/catalog/m"):
            response = await h.client.get(path)
            assert response.status == 404


@pytest.mark.asyncio
async def test_listener_reachability_does_not_imply_api_or_aggregate_readiness():
    async def models(_request):
        return web.json_response({"error": "unauthorized"}, status=401)

    async def catalog(_model):
        return "available"

    upstream = web.Application()
    upstream.router.add_get("/v1/models", models)
    async with make_harness(
        upstream_app=upstream,
        config=_enabled_config(),
        upstream_listener_check=_true,
        readyz_ps_check=_ps_ok,
        catalog_check=catalog,
    ) as h:
        listener = await h.client.get("/ops/upstream-listener", headers=AUTH)
        api = await h.client.get("/ops/upstream-api", headers=AUTH)
        ready = await h.client.get("/readyz", headers=AUTH)
        listener_body = await listener.json()
        ready_body = await ready.json()

    assert listener.status == 200
    assert listener_body == {"status": "listening"}
    assert api.status == 503
    assert ready.status == 503
    assert "upstream API" in ready_body["reason"]


@pytest.mark.asyncio
async def test_upstream_api_requires_200_and_models_shape():
    responses = [
        (200, {"not_data": []}, 503),
        (200, {"data": {}}, 503),
        (503, {"data": []}, 503),
        (200, {"data": [{"id": "m"}]}, 200),
    ]
    current = [responses[0]]

    async def models(_request):
        status, body, _expected = current[0]
        return web.json_response(body, status=status)

    upstream = web.Application()
    upstream.router.add_get("/v1/models", models)
    async with make_harness(upstream_app=upstream, config=_enabled_config()) as h:
        for item in responses:
            current[0] = item
            response = await h.client.get("/ops/upstream-api", headers=AUTH)
            assert response.status == item[2]


@pytest.mark.asyncio
async def test_catalog_distinguishes_available_missing_and_control_plane_failure():
    states = {"present": "available", "absent": "missing", "broken": "unavailable"}

    async def catalog(model):
        return states[model]

    upstream = web.Application()
    async with make_harness(
        upstream_app=upstream,
        config=_enabled_config(),
        catalog_check=catalog,
    ) as h:
        available = await h.client.get("/ops/catalog/present", headers=AUTH)
        missing = await h.client.get("/ops/catalog/absent", headers=AUTH)
        broken = await h.client.get("/ops/catalog/broken", headers=AUTH)
        available_body = await available.json()
        missing_body = await missing.json()
        broken_body = await broken.json()

    assert available.status == 200
    assert available_body == {"status": "available", "model": "present"}
    assert missing.status == 404
    assert missing_body == {"status": "missing", "model": "absent"}
    assert broken.status == 503
    assert broken_body == {
        "status": "unavailable",
        "reason": "control-plane",
    }


@pytest.mark.asyncio
async def test_probe_routes_require_caller_authentication():
    upstream = web.Application()
    async with make_harness(
        upstream_app=upstream,
        config=_enabled_config(),
        upstream_listener_check=_true,
    ) as h:
        for path in ("/ops/upstream-listener", "/ops/upstream-api", "/ops/catalog/m"):
            response = await h.client.get(path)
            assert response.status == 401
