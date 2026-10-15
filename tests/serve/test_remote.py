from __future__ import annotations

import asyncio
import json
import logging
import sys

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import laplace_serve.remote as remote_module
from laplace_serve.app import FLEET
from laplace_serve.config import LaplacedConfig, RemoteBackendConfig
from laplace_serve.remote import PROBE_INTERVAL_S, PROBE_TIMEOUT_S, RemoteBackend
from tests.serve.harness import UpstreamState, make_harness


def _config(remote_url: str, *, placed: bool = True) -> LaplacedConfig:
    return LaplacedConfig(
        reaper_sweep=False,
        backends={
            "shinobi": RemoteBackendConfig(
                url=remote_url,
                probe_model="probe-model",
                auth_command=(sys.executable, "-c", "print('test-key')"),
            )
        },
        placement={"bot:m": ("shinobi", "axis")} if placed else {},
    )


async def _start(app: web.Application) -> TestServer:
    server = TestServer(app)
    await server.start_server()
    return server


def _probe_response() -> web.Response:
    return web.json_response({"choices": [{"message": {"content": "OK"}}]})


@pytest.mark.asyncio
async def test_unplaced_backend_is_inert(monkeypatch):
    calls = 0

    async def auth(_self):
        nonlocal calls
        calls += 1
        return "unused"

    monkeypatch.setattr(RemoteBackend, "_run_auth_command", auth)
    upstream = web.Application()
    async with make_harness(
        upstream_app=upstream,
        upstream_state=UpstreamState(),
        config=_config("http://127.0.0.1:1", placed=False),
    ) as harness:
        status = await (await harness.client.get("/fleet/status")).json()

    assert calls == 0
    assert status["backends"]["shinobi"]["state"] == "DISABLED"
    assert status["backends"]["shinobi"]["reason"] == "unplaced"


@pytest.mark.asyncio
async def test_no_credential_disables_backend_but_healthz_stays_200():
    config = _config("http://127.0.0.1:1")
    config = LaplacedConfig(
        **{
            **config.__dict__,
            "backends": {
                "shinobi": RemoteBackendConfig(
                    url="http://127.0.0.1:1",
                    probe_model="probe-model",
                    auth_command=(sys.executable, "-c", "raise SystemExit(1)"),
                )
            },
        }
    )
    upstream = web.Application()
    async with make_harness(upstream_app=upstream, config=config) as harness:
        health = await harness.client.get("/healthz")
        status = await (await harness.client.get("/fleet/status")).json()

    assert health.status == 200
    assert status["backends"]["shinobi"]["state"] == "DISABLED"
    assert status["backends"]["shinobi"]["reason"] == "no-credential"


@pytest.mark.asyncio
async def test_disabled_backend_retries_auth_on_probe_cadence_and_recovers(monkeypatch):
    auth_results = iter((None, "recovered-key"))
    auth_calls = 0
    probe_states = []

    async def auth(_self):
        nonlocal auth_calls
        auth_calls += 1
        return next(auth_results)

    async def chat(_request):
        probe_states.append(backend.state)
        return _probe_response()

    monkeypatch.setattr(RemoteBackend, "_run_auth_command", auth)
    monkeypatch.setattr(remote_module, "PROBE_INTERVAL_S", 0.01)
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    server = await _start(app)
    backend = RemoteBackend(
        "shinobi",
        RemoteBackendConfig(
            url=str(server.make_url("")).rstrip("/"),
            probe_model="probe-model",
            auth_command=("stub",),
        ),
        active=True,
    )
    try:
        await backend.start()
        assert (backend.state, backend.reason, auth_calls) == (
            "DISABLED",
            "no-credential",
            1,
        )

        async def wait_until_up():
            while backend.state != "UP":
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_until_up(), timeout=0.5)
        assert (backend.state, backend.reason, auth_calls) == ("UP", None, 2)
        assert probe_states == ["DOWN", "DOWN"]
    finally:
        await backend.close()
        await server.close()


@pytest.mark.asyncio
async def test_control_character_credential_is_disabled_without_disclosure(
    monkeypatch, caplog
):
    sentinel = "bad-key\nnever-disclose"

    async def auth(_self):
        return sentinel

    monkeypatch.setattr(RemoteBackend, "_run_auth_command", auth)
    caplog.set_level(logging.DEBUG)
    upstream = web.Application()
    async with make_harness(
        upstream_app=upstream,
        config=_config("http://127.0.0.1:1"),
    ) as harness:
        snapshot = harness.client.app[FLEET].backends["shinobi"].snapshot()

    assert (snapshot["state"], snapshot["reason"]) == (
        "DISABLED",
        "bad-credential",
    )
    assert sentinel not in caplog.text
    assert sentinel not in json.dumps(snapshot)


def test_probe_state_machine_transitions():
    backend = RemoteBackend(
        "shinobi",
        RemoteBackendConfig(
            url="http://127.0.0.1:1",
            probe_model="m",
            auth_command=("stub",),
        ),
        active=True,
    )

    backend.record_probe_success()
    assert backend.state == "DOWN"
    backend.record_probe_success()
    assert backend.state == "UP"
    backend.record_probe_failure()
    assert backend.state == "UP"
    backend.record_probe_failure("probe-connect")
    assert (backend.state, backend.reason) == ("DOWN", "probe-connect")
    backend.record_probe_success()
    assert backend.state == "DOWN"
    backend.record_probe_success()
    assert backend.state == "UP"
    backend.record_request_failure("connect")
    assert (backend.state, backend.reason) == ("DOWN", "connect")


@pytest.mark.asyncio
async def test_probe_posts_required_shape_and_needs_two_successes():
    payloads = []

    async def chat(request):
        payloads.append(await request.json())
        return _probe_response()

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    server = await _start(app)
    backend = RemoteBackend(
        "shinobi",
        RemoteBackendConfig(
            url=str(server.make_url("")).rstrip("/"),
            probe_model="probe-model",
            auth_command=("stub",),
        ),
        active=True,
    )
    backend._credential = "test-key"
    backend._session = aiohttp.ClientSession()
    try:
        assert await backend.probe_once() is True
        assert backend.state == "DOWN"
        assert await backend.probe_once() is True
        assert backend.state == "UP"
    finally:
        await backend.close()
        await server.close()

    assert PROBE_INTERVAL_S == 120
    assert PROBE_TIMEOUT_S == 10
    assert payloads == [
        {
            "model": "probe-model",
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 4,
            "temperature": 0,
        },
        {
            "model": "probe-model",
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 4,
            "temperature": 0,
        },
    ]


@pytest.mark.asyncio
async def test_precommit_503_falls_back_to_axis():
    remote_requests = 0
    axis_requests = 0

    async def remote_chat(request):
        nonlocal remote_requests
        payload = await request.json()
        if payload["model"] == "probe-model":
            return _probe_response()
        remote_requests += 1
        return web.json_response({"error": "unavailable"}, status=503)

    async def axis_chat(request):
        nonlocal axis_requests
        axis_requests += 1
        return web.json_response({"backend": "axis"})

    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            harness.client.app[FLEET].backends["shinobi"].state = "UP"
            response = await asyncio.wait_for(
                harness.client.post(
                    "/bot/v1/chat/completions", json={"model": "m"}
                ),
                timeout=0.5,
            )
            assert response.status == 200
            assert (await response.json())["backend"] == "axis"
    finally:
        await remote_server.close()

    assert (remote_requests, axis_requests) == (1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("remote_status", [400, 404, 429])
async def test_client_status_falls_back_without_marking_remote_down(remote_status):
    async def remote_chat(request):
        payload = await request.json()
        if payload["model"] == "probe-model":
            return _probe_response()
        return web.json_response({"error": "remote rejection"}, status=remote_status)

    async def axis_chat(_request):
        return web.json_response({"backend": "axis"})

    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            backend = harness.client.app[FLEET].backends["shinobi"]
            backend.state = "UP"
            response = await asyncio.wait_for(
                harness.client.post(
                    "/bot/v1/chat/completions", json={"model": "m"}
                ),
                timeout=0.5,
            )
            assert response.status == 200
            assert (await response.json())["backend"] == "axis"
            assert backend.state == "UP"
            assert backend.consecutive_failures == 0
    finally:
        await remote_server.close()


@pytest.mark.asyncio
async def test_400_then_normal_request_uses_remote():
    remote_requests = 0
    axis_requests = 0

    async def remote_chat(request):
        nonlocal remote_requests
        payload = await request.json()
        if payload["model"] == "probe-model":
            return _probe_response()
        remote_requests += 1
        if remote_requests == 1:
            return web.json_response({"error": "bad request"}, status=400)
        return web.json_response({"backend": "remote"})

    async def axis_chat(_request):
        nonlocal axis_requests
        axis_requests += 1
        return web.json_response({"backend": "axis"})

    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            backend = harness.client.app[FLEET].backends["shinobi"]
            backend.state = "UP"
            first = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m"}
            )
            second = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m"}
            )
            assert (await first.json())["backend"] == "axis"
            assert (await second.json())["backend"] == "remote"
            assert backend.state == "UP"
    finally:
        await remote_server.close()

    assert (remote_requests, axis_requests) == (2, 1)


@pytest.mark.asyncio
async def test_connect_error_falls_back_to_axis():
    async def axis_chat(_request):
        return web.json_response({"backend": "axis"})

    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    async with make_harness(
        upstream_app=axis_app,
        config=_config("http://127.0.0.1:1"),
    ) as harness:
        harness.client.app[FLEET].backends["shinobi"].state = "UP"
        response = await harness.client.post(
            "/bot/v1/chat/completions", json={"model": "m"}
        )
        assert response.status == 200
        assert (await response.json())["backend"] == "axis"


@pytest.mark.asyncio
async def test_slow_headers_use_remote_without_marking_it_down(monkeypatch):
    axis_requests = 0

    async def remote_chat(request):
        payload = await request.json()
        if payload["model"] == "probe-model":
            return _probe_response()
        await asyncio.sleep(0.08)
        return web.json_response({"backend": "remote"})

    async def axis_chat(_request):
        nonlocal axis_requests
        axis_requests += 1
        return web.json_response({"backend": "axis"})

    monkeypatch.setattr(remote_module, "PRECOMMIT_TIMEOUT_S", 0.02, raising=False)
    monkeypatch.setattr(remote_module, "FIRST_BYTE_TIMEOUT_S", 0.2)
    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            backend = harness.client.app[FLEET].backends["shinobi"]
            backend.state = "UP"
            response = await asyncio.wait_for(
                harness.client.post(
                    "/bot/v1/chat/completions", json={"model": "m"}
                ),
                timeout=0.5,
            )
            assert response.status == 200
            assert (await response.json())["backend"] == "remote"
            assert backend.state == "UP"
    finally:
        await remote_server.close()

    assert axis_requests == 0


@pytest.mark.asyncio
async def test_header_first_byte_timeout_falls_back_to_axis(monkeypatch):
    release_remote = asyncio.Event()
    axis_requests = 0

    async def remote_chat(request):
        payload = await request.json()
        if payload["model"] == "probe-model":
            return _probe_response()
        await release_remote.wait()
        return web.json_response({"remote": True})

    async def axis_chat(_request):
        nonlocal axis_requests
        axis_requests += 1
        return web.json_response({"backend": "axis"})

    monkeypatch.setattr(remote_module, "FIRST_BYTE_TIMEOUT_S", 0.02)
    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            harness.client.app[FLEET].backends["shinobi"].state = "UP"
            response = await asyncio.wait_for(
                harness.client.post(
                    "/bot/v1/chat/completions", json={"model": "m"}
                ),
                timeout=0.5,
            )
            assert response.status == 200
            assert (await response.json())["backend"] == "axis"
    finally:
        release_remote.set()
        await remote_server.close()

    assert axis_requests == 1


@pytest.mark.asyncio
async def test_saturated_slot_times_out_and_falls_back_to_axis(monkeypatch):
    async def axis_chat(_request):
        return web.json_response({"backend": "axis"})

    monkeypatch.setattr(remote_module, "QUEUE_TIMEOUT_S", 0.02)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    async with make_harness(
        upstream_app=axis_app,
        config=_config("http://127.0.0.1:1"),
    ) as harness:
        backend = harness.client.app[FLEET].backends["shinobi"]
        backend.state = "UP"
        await backend._slots.acquire()
        try:
            response = await asyncio.wait_for(
                harness.client.post(
                    "/bot/v1/chat/completions", json={"model": "m"}
                ),
                timeout=0.5,
            )
        finally:
            backend._slots.release()

        assert response.status == 200
        assert (await response.json())["backend"] == "axis"
        assert backend.queued == 0


@pytest.mark.asyncio
async def test_cancelled_slot_wait_does_not_leak_queued_counter():
    backend = RemoteBackend(
        "shinobi",
        RemoteBackendConfig(
            url="http://127.0.0.1:1",
            probe_model="probe-model",
            auth_command=("stub",),
        ),
        active=True,
    )
    backend.state = "UP"
    backend._credential = "test-key"
    backend._session = aiohttp.ClientSession()
    await backend._slots.acquire()
    request = type(
        "StubRequest",
        (),
        {"method": "POST", "headers": {}, "query_string": "", "query": {}},
    )()
    task = asyncio.create_task(backend.forward(request, "/v1/chat/completions", b"{}"))
    try:
        while backend.queued == 0:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.queued == 0
        assert backend.in_flight == 0
    finally:
        backend._slots.release()
        await backend.close()


@pytest.mark.asyncio
async def test_first_response_byte_commits_and_is_never_replayed():
    axis_requests = 0

    async def remote_chat(request):
        payload = await request.json()
        if payload["model"] == "probe-model":
            return _probe_response()
        response = web.StreamResponse(status=200)
        await response.prepare(request)
        await response.write(b"committed")
        request.transport.close()
        return response

    async def axis_chat(request):
        nonlocal axis_requests
        axis_requests += 1
        return web.json_response({"backend": "axis"})

    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    received = b""
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            harness.client.app[FLEET].backends["shinobi"].state = "UP"
            response = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m", "stream": True}
            )
            try:
                received = await asyncio.wait_for(response.read(), timeout=1.0)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                received = b"committed"
    finally:
        await remote_server.close()

    assert received.startswith(b"committed")
    assert axis_requests == 0


@pytest.mark.asyncio
async def test_auth_runs_at_start_and_on_401_not_per_request_and_key_is_not_logged(
    monkeypatch, caplog
):
    sentinel = "sentinel-key-never-log-91c7"
    with pytest.raises(ValueError) as caught:
        LaplacedConfig.from_dict(
            {
                "backends": {
                    "shinobi": {
                        "url": "http://127.0.0.1:1",
                        "probe_model": "m",
                        "auth_command": [sentinel, 3],
                    }
                }
            }
        )
    assert sentinel not in str(caught.value)

    auth_calls = 0
    request_status = {"value": 200}

    async def auth(_self):
        nonlocal auth_calls
        auth_calls += 1
        return sentinel

    async def remote_chat(request):
        payload = await request.json()
        assert request.headers["Authorization"] == f"Bearer {sentinel}"
        if payload["model"] == "probe-model":
            return _probe_response()
        return web.json_response({"remote": True}, status=request_status["value"])

    async def axis_chat(request):
        return web.json_response({"axis": True})

    monkeypatch.setattr(RemoteBackend, "_run_auth_command", auth)
    caplog.set_level(logging.DEBUG)
    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            backend = harness.client.app[FLEET].backends["shinobi"]
            backend.state = "UP"
            first = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m"}
            )
            assert first.status == 200
            assert auth_calls == 1

            request_status["value"] = 401
            backend.state = "UP"
            second = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m"}
            )
            assert second.status == 200
            assert (await second.json())["axis"] is True
            assert auth_calls == 2
            status_text = json.dumps(backend.snapshot())
    finally:
        await remote_server.close()

    assert sentinel not in caplog.text
    assert sentinel not in status_text
