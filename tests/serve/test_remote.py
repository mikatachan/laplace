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
from laplace_serve.remote import (
    PROBE_COLD_TIMEOUT_S,
    PROBE_INTERVAL_S,
    PROBE_TIMEOUT_S,
    RemoteBackend,
)
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


def _two_model_config(remote_url: str) -> LaplacedConfig:
    config = _config(remote_url)
    return LaplacedConfig(
        **{
            **config.__dict__,
            "placement": {
                "bot:coder": ("shinobi", "axis"),
                "bot:gemma": ("shinobi", "axis"),
            },
        }
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
    assert PROBE_COLD_TIMEOUT_S == 60
    assert PROBE_COLD_TIMEOUT_S < PROBE_INTERVAL_S
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
async def test_probe_timeout_is_cold_when_not_up_and_short_when_up(monkeypatch):
    totals = []
    real_timeout = aiohttp.ClientTimeout

    def recording_timeout(*args, **kwargs):
        totals.append(kwargs.get("total"))
        return real_timeout(*args, **kwargs)

    monkeypatch.setattr(remote_module.aiohttp, "ClientTimeout", recording_timeout)

    async def chat(request):
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
        assert backend.state == "DOWN"
        await backend.probe_once()  # DOWN -> cold timeout
        await backend.probe_once()  # second success flips to UP
        assert backend.state == "UP"
        await backend.probe_once()  # UP -> short timeout
        backend._set_model_state("probe-model", "DOWN", "request-failure")
        await backend.probe_once()  # a DOWN probe model may need a cold load
    finally:
        await backend.close()
        await server.close()

    assert totals == [
        PROBE_COLD_TIMEOUT_S,
        PROBE_COLD_TIMEOUT_S,
        PROBE_TIMEOUT_S,
        PROBE_COLD_TIMEOUT_S,
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
@pytest.mark.parametrize("stream", [False, True])
async def test_inband_error_falls_back_before_commit(stream, caplog):
    remote_requests = 0

    async def remote_chat(request):
        nonlocal remote_requests
        payload = await request.json()
        if payload["model"] == "probe-model":
            return _probe_response()
        remote_requests += 1
        if remote_requests > 1:
            return web.json_response({"backend": "remote"})
        if not stream:
            return web.json_response({"error": {"message": "context too long"}})
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        await response.write(
            b'data: {"error":{"message":"context too long"}}\n\n'
        )
        return response

    async def axis_chat(_request):
        return web.json_response({"backend": "axis"})

    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    caplog.set_level(logging.INFO, logger="laplace_serve.app")
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            backend = harness.client.app[FLEET].backends["shinobi"]
            backend.state = "UP"
            response = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m", "stream": stream}
            )
            assert (await response.json()) == {"backend": "axis"}
            assert backend.snapshot()["models"]["m"] == {
                "state": "UP",
                "reason": None,
                "consecutive_failures": 0,
            }
            retry = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m"}
            )
            assert (await retry.json()) == {"backend": "remote"}
    finally:
        await remote_server.close()

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "reason=fallback:upstream-inband-error message=context too long" in message
        for message in messages
    )
    assert sum("reason=served" in message for message in messages) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b'{"choices":[{"message":{"content":"hi"}}],"error":null}',
        b'{"error":"' + b"x" * (64 * 1024) + b'"}',
    ],
    ids=["error-null", "large-body"],
)
async def test_non_stream_inspection_commits_safe_or_oversized_bodies(body):
    axis_requests = 0

    async def remote_chat(_request):
        return web.Response(body=body, content_type="application/json")

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
            harness.client.app[FLEET].backends["shinobi"].state = "UP"
            response = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m"}
            )
            assert await response.read() == body
    finally:
        await remote_server.close()

    assert axis_requests == 0


@pytest.mark.asyncio
async def test_stream_json_decoder_recursion_error_passes_through():
    body = b"[" * 60000 + b"]" * 60000
    axis_requests = 0

    async def remote_chat(_request):
        response = web.StreamResponse(
            headers={"Content-Type": "application/json"}
        )
        await response.prepare(_request)
        await response.write(body[:60000])
        await response.write(body[60000:])
        await response.write_eof()
        return response

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
            harness.client.app[FLEET].backends["shinobi"].state = "UP"
            response = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m", "stream": True}
            )
            assert response.status == 200
            assert await response.read() == body
    finally:
        await remote_server.close()

    assert axis_requests == 0


@pytest.mark.parametrize("error", [ValueError, RecursionError])
def test_non_stream_json_decoder_error_does_not_escape(monkeypatch, error):
    def fail_to_decode(_body):
        raise error("mock decoder failure")

    monkeypatch.setattr(remote_module.json, "loads", fail_to_decode)

    assert remote_module._error_message_from_bytes(b"{}") is None


def test_sanitizes_bare_bearer_secret():
    assert remote_module._sanitize_error_message("Bearer hunter2secret") == (
        "Bearer [redacted]"
    )


@pytest.mark.asyncio
async def test_model_failure_does_not_down_another_model():
    coder_requests = 0

    async def remote_chat(request):
        nonlocal coder_requests
        payload = await request.json()
        if payload["model"] == "coder":
            coder_requests += 1
            return web.json_response({"error": "coder failed"}, status=503)
        return web.json_response({"backend": "remote", "model": payload["model"]})

    async def axis_chat(request):
        payload = await request.json()
        return web.json_response({"backend": "axis", "model": payload["model"]})

    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_two_model_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            backend = harness.client.app[FLEET].backends["shinobi"]
            backend.state = "UP"
            coder = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "coder"}
            )
            coder_retry = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "coder"}
            )
            gemma = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "gemma"}
            )
            assert (await coder.json())["backend"] == "axis"
            assert (await coder_retry.json())["backend"] == "axis"
            assert (await gemma.json())["backend"] == "remote"
            snapshot = backend.snapshot()
            assert snapshot["state"] == "UP"
            assert snapshot["models"]["coder"]["state"] == "DOWN"
            assert snapshot["models"]["gemma"]["state"] == "UP"
    finally:
        await remote_server.close()

    assert coder_requests == 1


@pytest.mark.asyncio
async def test_connect_failure_downs_host_for_every_model():
    class RefusedSession:
        async def request(self, *_args, **_kwargs):
            raise aiohttp.ClientConnectionError("mock connection refused")

    backend = RemoteBackend(
        "shinobi",
        RemoteBackendConfig(
            url="http://mock.invalid",
            probe_model="gemma",
            auth_command=("stub",),
        ),
        active=True,
    )
    backend.state = "UP"
    backend._credential = "test-key"
    backend._session = RefusedSession()
    backend.ensure_model("coder")
    backend.ensure_model("gemma")
    request = type(
        "StubRequest",
        (),
        {"method": "POST", "headers": {}, "query_string": "", "query": {}},
    )()

    result = await backend.forward(
        request, "/v1/chat/completions", b"{}", "coder", False
    )

    assert result.failure_reason == "connect"
    assert (backend.state, backend.reason) == ("DOWN", "connect")
    assert backend.snapshot()["models"]["gemma"]["state"] == "UP"


@pytest.mark.asyncio
async def test_probe_timeout_and_connect_have_distinct_reasons():
    class FailingContext:
        def __init__(self, error):
            self.error = error

        async def __aenter__(self):
            raise self.error

        async def __aexit__(self, *_args):
            return False

    class FailingProbeSession:
        def __init__(self, error):
            self.error = error

        def post(self, *_args, **_kwargs):
            return FailingContext(self.error)

    timeout_backend = RemoteBackend(
        "timeout-host",
        RemoteBackendConfig(
            url="http://mock.invalid",
            probe_model="m",
            auth_command=("stub",),
        ),
        active=True,
    )
    timeout_backend.state = "UP"
    timeout_backend._credential = "test-key"
    timeout_backend._session = FailingProbeSession(asyncio.TimeoutError())
    assert await timeout_backend.probe_once() is False
    assert await timeout_backend.probe_once() is False
    assert (timeout_backend.state, timeout_backend.reason) == (
        "DOWN",
        "probe-timeout",
    )

    connect_backend = RemoteBackend(
        "connect-host",
        RemoteBackendConfig(
            url="http://mock.invalid",
            probe_model="m",
            auth_command=("stub",),
        ),
        active=True,
    )
    connect_backend.state = "UP"
    connect_backend._credential = "test-key"
    connect_backend._session = FailingProbeSession(
        aiohttp.ClientConnectionError("mock connection refused")
    )
    assert await connect_backend.probe_once() is False
    assert await connect_backend.probe_once() is False
    assert (connect_backend.state, connect_backend.reason) == (
        "DOWN",
        "probe-connect",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("http", "probe-http"),
        ("body", "probe-body"),
        ("auth", "auth"),
    ],
)
async def test_probe_response_failures_down_the_host(failure, reason):
    async def chat(_request):
        if failure == "http":
            return web.Response(status=503)
        if failure == "body":
            return web.Response(text="not json", content_type="application/json")
        return web.Response(status=401)

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
    backend.state = "UP"
    backend._credential = "test-key"
    backend._session = aiohttp.ClientSession()

    async def refresh_credential():
        backend._credential = "test-key"
        return True

    backend._refresh_credential = refresh_credential
    try:
        assert await backend.probe_once() is False
        assert backend.state == "UP"
        assert await backend.probe_once() is False
        assert (backend.state, backend.reason) == ("DOWN", reason)
    finally:
        await backend.close()
        await server.close()


@pytest.mark.asyncio
async def test_probe_all_urls_failed_cannot_leave_host_up():
    primary_calls = 0
    alternate_calls = 0

    async def primary(_request):
        nonlocal primary_calls
        primary_calls += 1
        return web.Response(status=503)

    async def alternate(_request):
        nonlocal alternate_calls
        alternate_calls += 1
        return web.Response(text="not json", content_type="application/json")

    primary_app = web.Application()
    primary_app.router.add_post("/v1/chat/completions", primary)
    alternate_app = web.Application()
    alternate_app.router.add_post("/v1/chat/completions", alternate)
    primary_server = await _start(primary_app)
    alternate_server = await _start(alternate_app)
    backend = RemoteBackend(
        "shinobi",
        RemoteBackendConfig(
            url=str(primary_server.make_url("")).rstrip("/"),
            alt_urls=(str(alternate_server.make_url("")).rstrip("/"),),
            probe_model="probe-model",
            auth_command=("stub",),
        ),
        active=True,
    )
    backend.state = "UP"
    backend._credential = "test-key"
    backend._session = aiohttp.ClientSession()
    try:
        assert await backend.probe_once() is False
        assert await backend.probe_once() is False
        assert (backend.state, backend.reason) == ("DOWN", "probe-body")
        assert (primary_calls, alternate_calls) == (2, 2)
    finally:
        await backend.close()
        await primary_server.close()
        await alternate_server.close()


@pytest.mark.asyncio
async def test_probe_loop_survives_unexpected_exception(monkeypatch, caplog):
    backend = RemoteBackend(
        "shinobi",
        RemoteBackendConfig(
            url="http://mock.invalid",
            probe_model="probe-model",
            auth_command=("stub",),
        ),
        active=True,
    )
    backend.state = "UP"
    calls = 0

    async def probe_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("mock unexpected probe failure")
        raise asyncio.CancelledError

    async def no_delay(_seconds):
        return None

    backend.probe_once = probe_once
    monkeypatch.setattr(remote_module.asyncio, "sleep", no_delay)
    caplog.set_level(logging.ERROR, logger="laplace_serve.remote")

    with pytest.raises(asyncio.CancelledError):
        await backend._probe_loop()

    assert calls == 2
    assert backend.consecutive_failures == 1
    assert "probe failed unexpectedly" in caplog.text


def test_two_probe_successes_recover_host_and_model_state():
    backend = RemoteBackend(
        "shinobi",
        RemoteBackendConfig(
            url="http://mock.invalid",
            probe_model="gemma",
            auth_command=("stub",),
        ),
        active=True,
    )
    backend.state = "UP"
    backend.record_request_failure("upstream-5xx", "coder")
    backend.record_request_failure("connect", "gemma", host_wide=True)

    backend.record_probe_success()
    assert backend.state == "DOWN"
    assert backend.snapshot()["models"]["coder"]["state"] == "DOWN"
    backend.record_probe_success()

    assert backend.state == "UP"
    assert backend.snapshot()["models"]["coder"] == {
        "state": "UP",
        "reason": None,
        "consecutive_failures": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "reason"), [(400, "upstream-400"), (503, "upstream-5xx")]
)
async def test_upstream_error_log_is_sanitized_and_truncated(
    status, reason, caplog
):
    secret = "hunter2secret"
    bearer_secret = "sk-lm-Ab12Cd34:Ef56Gh78Ij90Kl12Mn34"
    long_tail = "ordinary upstream detail " * 20

    async def remote_chat(_request):
        return web.json_response(
            {
                "error": (
                    f"bad\npassword: {secret} \x1b[31m"
                    f"Authorization: Bearer {bearer_secret}\x1b[0m {long_tail}"
                )
            },
            status=status,
        )

    async def axis_chat(_request):
        return web.json_response({"backend": "axis"})

    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    axis_app = web.Application()
    axis_app.router.add_post("/v1/chat/completions", axis_chat)
    caplog.set_level(logging.INFO, logger="laplace_serve.app")
    try:
        async with make_harness(
            upstream_app=axis_app,
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            harness.client.app[FLEET].backends["shinobi"].state = "UP"
            response = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m"}
            )
            assert (await response.json())["backend"] == "axis"
    finally:
        await remote_server.close()

    fallback = next(
        record.getMessage()
        for record in caplog.records
        if f"reason=fallback:{reason}" in record.getMessage()
    )
    assert secret not in fallback
    assert bearer_secret not in fallback
    assert "\x1b" not in fallback
    assert "\n" not in fallback
    assert "Authorization=[redacted]" in fallback
    assert fallback.endswith("...")


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
    task = asyncio.create_task(
        backend.forward(request, "/v1/chat/completions", b"{}", "m", False)
    )
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


@pytest.mark.asyncio
async def test_successful_remote_forward_logs_served(caplog):
    async def remote_chat(_request):
        return web.json_response({"backend": "remote"})

    remote_app = web.Application()
    remote_app.router.add_post("/v1/chat/completions", remote_chat)
    remote_server = await _start(remote_app)
    caplog.set_level(logging.INFO, logger="laplace_serve.app")
    try:
        async with make_harness(
            upstream_app=web.Application(),
            config=_config(str(remote_server.make_url("")).rstrip("/")),
        ) as harness:
            harness.client.app[FLEET].backends["shinobi"].state = "UP"
            response = await harness.client.post(
                "/bot/v1/chat/completions", json={"model": "m"}
            )
            assert (await response.json())["backend"] == "remote"
    finally:
        await remote_server.close()

    served = [
        r.getMessage() for r in caplog.records if "reason=served" in r.getMessage()
    ]
    assert served == ["route origin=bot model=m backend=shinobi reason=served"]


def test_state_transitions_log_once_per_change(caplog):
    backend = RemoteBackend(
        "shinobi",
        RemoteBackendConfig(
            url="http://127.0.0.1:1",
            probe_model="probe-model",
            auth_command=("secret-argv",),
        ),
        active=True,
    )
    caplog.set_level(logging.INFO, logger="laplace_serve.remote")

    def transitions():
        return [
            r.getMessage() for r in caplog.records if "state" in r.getMessage()
        ]

    backend.record_probe_success()
    assert transitions() == []
    backend.record_probe_success()
    assert transitions() == ["backend shinobi state DOWN->UP reason=None"]
    backend.record_probe_success()
    assert len(transitions()) == 1

    backend.record_request_failure("connect")
    assert transitions()[1:] == ["backend shinobi state UP->DOWN reason=connect"]
    backend.record_request_failure("connect")
    assert len(transitions()) == 2
    assert "secret-argv" not in caplog.text
