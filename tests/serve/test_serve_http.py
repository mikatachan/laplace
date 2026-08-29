from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
import pytest
from aiohttp import web

from laplace.adapter import ModelLoadError
from laplace.broker import ContentionBroker
from laplace.reaper import Reaper
from laplace_serve.config import LaplacedConfig
from tests.fakes import FakeAdapter, loaded_model
from tests.serve.harness import UpstreamState, make_harness

_GIB = 1024 ** 3
_SSE_CHUNKS = [b'data: {"i": %d}\n\n' % i for i in range(8)] + [b"data: [DONE]\n\n"]


class _ModelLoadErrorAdapter(FakeAdapter):
    async def ensure_loaded(self, model_id: str, context_length: int | None) -> None:
        raise ModelLoadError("simulated model load failure")


def _json_body(**payload) -> bytes:
    return json.dumps(payload).encode()


def _record(state: UpstreamState, request: web.Request, body: bytes) -> None:
    state.requests.append(
        {
            "path": request.path,
            "method": request.method,
            "authorization": request.headers.get("Authorization"),
            "body": body,
        }
    )


def _sse_upstream(state: UpstreamState, path: str) -> web.Application:
    async def handler(request: web.Request) -> web.StreamResponse:
        _record(state, request, await request.read())
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for chunk in _SSE_CHUNKS:
            await resp.write(chunk)  # write flushes each frame immediately
            await asyncio.sleep(0.03)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post(path, handler)
    return app


@pytest.mark.asyncio
async def test_admission_gating_end_to_end(caplog):
    state = UpstreamState()

    async def chat(request):
        body = await request.read()
        _record(state, request, body)
        return web.json_response({"echo_model": json.loads(body)["model"]})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)

    caplog.set_level(logging.INFO, logger="laplace_serve.app")
    async with make_harness(upstream_app=upstream, upstream_state=state) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="m", stream=False),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        assert (await resp.json())["echo_model"] == "m"

    assert state.requests[0]["path"] == "/v1/chat/completions"
    assert "admission origin=hermes tier=interactive model=m decision=admit" in caplog.text
    assert h.broker._active_calls() == 0


@pytest.mark.asyncio
async def test_bare_model_id_uses_canonical_identity_for_admission():
    state = UpstreamState()

    async def chat(request):
        body = await request.read()
        _record(state, request, body)
        return web.json_response({"ok": True})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    canonical = "qwen/qwen3-coder-30b"
    config = LaplacedConfig(
        reaper_sweep=False,
        model_context={canonical: 131072},
        model_footprint_mb={canonical: 22000},
    )

    async with make_harness(upstream_app=upstream, upstream_state=state, config=config) as h:
        response = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="qwen3-coder-30b"),
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 200

    assert h.adapter.ensure_loaded_calls == [(canonical, 131072)]
    assert state.requests[0]["body"] == _json_body(model="qwen3-coder-30b")


@pytest.mark.asyncio
async def test_bare_v1_post_governed_as_default_origin(caplog):
    state = UpstreamState()

    async def chat(request):
        _record(state, request, await request.read())
        return web.json_response({"ok": True})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)

    caplog.set_level(logging.INFO, logger="laplace_serve.app")
    async with make_harness(upstream_app=upstream, upstream_state=state) as h:
        resp = await h.client.post(
            "/v1/chat/completions",
            data=_json_body(model="m"),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
    assert state.requests[0]["path"] == "/v1/chat/completions"
    assert "admission origin=default tier=scheduled model=m decision=admit" in caplog.text


@pytest.mark.asyncio
async def test_streaming_byte_fidelity_and_incremental():
    state = UpstreamState()
    upstream = _sse_upstream(state, "/v1/chat/completions")

    async with make_harness(upstream_app=upstream, upstream_state=state) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="m", stream=True),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        pieces = []
        while True:
            piece = await resp.content.readany()
            if not piece:
                break
            pieces.append(piece)

    assert b"".join(pieces) == b"".join(_SSE_CHUNKS)  # byte identical
    assert len(pieces) >= 2  # relayed incrementally, not aggregated


@pytest.mark.asyncio
async def test_responses_passthrough_frames_identical(caplog):
    state = UpstreamState()
    upstream = _sse_upstream(state, "/v1/responses")

    caplog.set_level(logging.INFO, logger="laplace_serve.app")
    async with make_harness(upstream_app=upstream, upstream_state=state) as h:
        resp = await h.client.post(
            "/openclaw/v1/responses",
            data=_json_body(model="qwen3-next-80b-a3b-thinking", stream=True),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        received = await resp.read()

    assert received == b"".join(_SSE_CHUNKS)
    assert state.requests[0]["path"] == "/v1/responses"
    assert json.loads(state.requests[0]["body"])["model"] == "qwen3-next-80b-a3b-thinking"
    assert "model=qwen3-next-80b-a3b-thinking decision=admit" in caplog.text


@pytest.mark.asyncio
async def test_authorization_passthrough_and_never_logged(caplog):
    state = UpstreamState()
    secret = "Bearer super-secret-token-9f3c"

    async def chat(request):
        _record(state, request, await request.read())
        return web.json_response({"ok": True})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)

    caplog.set_level(logging.DEBUG)
    async with make_harness(upstream_app=upstream, upstream_state=state) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="m"),
            headers={"Content-Type": "application/json", "Authorization": secret},
        )
        assert resp.status == 200

    assert state.requests[0]["authorization"] == secret  # forwarded verbatim
    assert "super-secret-token-9f3c" not in caplog.text  # never logged


@pytest.mark.asyncio
async def test_downstream_disconnect_aborts_upstream_and_releases():
    state = UpstreamState()

    async def stream_forever(request):
        _record(state, request, await request.read())
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            for i in range(500):
                await resp.write(b'data: {"i": %d}\n\n' % i)
                await asyncio.sleep(0.02)
            await resp.write_eof()
        except asyncio.CancelledError:
            state.aborted = True
            raise
        except Exception:  # noqa: BLE001 - any write failure means the client is gone
            state.aborted = True
            raise
        return resp

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", stream_forever)

    adapter = FakeAdapter(
        loaded=[loaded_model("m", size_bytes=10 * _GIB)],
        footprints={"m": 1000},
    )
    reaper = Reaper(adapter)
    broker = ContentionBroker(adapter, reaper, budget_mb=90000)

    async with make_harness(
        upstream_app=upstream, upstream_state=state,
        adapter=adapter, reaper=reaper, broker=broker,
    ) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="m", stream=True),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        first = await resp.content.readany()
        assert first  # stream started
        assert broker._active_calls() == 1
        resp.close()  # abort mid-stream

        for _ in range(200):
            if state.aborted and broker._active_calls() == 0:
                break
            await asyncio.sleep(0.02)

    assert state.aborted is True  # upstream observed the closed connection
    assert broker._active_calls() == 0  # slot released


@pytest.mark.asyncio
async def test_edge_a_upstream_non_200_relayed_as_is():
    state = UpstreamState()

    async def rate_limited(request):
        _record(state, request, await request.read())
        return web.json_response({"error": "rate limited"}, status=429)

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", rate_limited)

    async with make_harness(upstream_app=upstream, upstream_state=state) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="m"),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 429
        assert (await resp.json())["error"] == "rate limited"


@pytest.mark.asyncio
async def test_model_load_failure_returns_503_without_upstream_proxy():
    state = UpstreamState()

    async def chat(request):
        _record(state, request, await request.read())
        return web.json_response({"ok": True})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)

    async with make_harness(
        upstream_app=upstream,
        upstream_state=state,
        adapter=_ModelLoadErrorAdapter(),
    ) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="m"),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 503
        assert (await resp.json())["error"] == "model unavailable; load did not complete"

    assert state.requests == []


@pytest.mark.asyncio
async def test_edge_b_upstream_dies_midstream_no_synthesized_done():
    state = UpstreamState()

    async def die_midstream(request):
        _record(state, request, await request.read())
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"i": 0}\n\n')
        await resp.write(b'data: {"i": 1}\n\n')
        request.transport.close()  # abrupt death, no [DONE]
        return resp

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", die_midstream)

    async with make_harness(upstream_app=upstream, upstream_state=state) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="m", stream=True),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        received = b""
        try:
            while True:
                piece = await resp.content.readany()
                if not piece:
                    break
                received += piece
        except aiohttp.ClientError:
            pass

    assert received.startswith(b'data: {"i": 0}\n\ndata: {"i": 1}\n\n')
    assert b"[DONE]" not in received  # daemon never synthesizes a terminator


@pytest.mark.asyncio
async def test_edge_c_status_and_headers_before_first_chunk():
    state = UpstreamState()

    async def delayed_body(request):
        _record(state, request, await request.read())
        resp = web.StreamResponse(status=200, headers={"X-Probe": "present"})
        await resp.prepare(request)  # headers flushed now
        await asyncio.sleep(0.15)  # body withheld
        await resp.write(b"payload-bytes")
        await resp.write_eof()
        return resp

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", delayed_body)

    async with make_harness(upstream_app=upstream, upstream_state=state) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="m", stream=True),
            headers={"Content-Type": "application/json"},
        )
        # Response object (status + headers) arrives before the withheld body.
        assert resp.status == 200
        assert resp.headers["X-Probe"] == "present"
        body = await resp.read()
        assert body == b"payload-bytes"


@pytest.mark.asyncio
async def test_ungoverned_models_passthrough_strips_prefix():
    state = UpstreamState()

    async def models(request):
        _record(state, request, await request.read())
        return web.json_response({"data": [{"id": "m"}]})

    upstream = web.Application()
    upstream.router.add_get("/v1/models", models)

    async with make_harness(upstream_app=upstream, upstream_state=state) as h:
        resp = await h.client.get("/hermes/v1/models")
        assert resp.status == 200
        bare = await h.client.get("/v1/models")
        assert bare.status == 200

    assert [r["path"] for r in state.requests] == ["/v1/models", "/v1/models"]


@pytest.mark.asyncio
async def test_upstream_connect_failure_returns_502():
    upstream = web.Application()  # unused; the proxy points at a dead port
    async with make_harness(
        upstream_app=upstream,
        upstream_state=UpstreamState(),
        upstream="http://127.0.0.1:1",
    ) as h:
        resp = await h.client.post(
            "/hermes/v1/chat/completions",
            data=_json_body(model="m"),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 502


@pytest.mark.asyncio
async def test_upstream_session_has_no_total_timeout_ceiling():
    # Regression pin (plan-0017 gate FINDING 1): the aiohttp default of
    # total=300s covers the ENTIRE streamed body read, killing any proxied
    # generation longer than 5 minutes mid-stream. The startup session must
    # build with total=None (uncapped stream) and only a connect-phase bound.
    # A real >300s stream test is impractical, so pin the construction.
    from laplace_serve.app import SESSION

    upstream = web.Application()
    async with make_harness(upstream_app=upstream, upstream_state=UpstreamState()) as h:
        session = h.client.app[SESSION]
        assert session.timeout.total is None  # no 5-minute stream ceiling
        assert session.timeout.sock_connect == 30  # connect phase still fails fast


@pytest.mark.asyncio
async def test_healthz_is_cheap_and_ok():
    upstream = web.Application()
    async with make_harness(upstream_app=upstream, upstream_state=UpstreamState()) as h:
        resp = await h.client.get("/healthz")
        assert resp.status == 200
        assert (await resp.json())["status"] == "ok"


@pytest.mark.asyncio
async def test_readyz_ready_when_models_and_ps_ok():
    async def models(request):
        return web.json_response({"data": []})

    upstream = web.Application()
    upstream.router.add_get("/v1/models", models)

    async def ps_ok():
        return True

    async with make_harness(
        upstream_app=upstream, upstream_state=UpstreamState(), readyz_ps_check=ps_ok
    ) as h:
        resp = await h.client.get("/readyz")
        assert resp.status == 200
        assert (await resp.json())["status"] == "ready"


@pytest.mark.asyncio
async def test_readyz_ready_when_upstream_returns_401():
    # LM Studio answers unauthenticated /v1/models probes with 401. Any HTTP
    # status proves the upstream is reachable and alive, so readyz is ready.
    async def models(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    upstream = web.Application()
    upstream.router.add_get("/v1/models", models)

    async def ps_ok():
        return True

    async with make_harness(
        upstream_app=upstream, upstream_state=UpstreamState(), readyz_ps_check=ps_ok
    ) as h:
        resp = await h.client.get("/readyz")
        assert resp.status == 200
        assert (await resp.json())["status"] == "ready"


@pytest.mark.asyncio
async def test_readyz_not_ready_when_upstream_unreachable():
    # A connect error/timeout (nothing listening) is the only upstream failure
    # that keeps readyz not-ready.
    async def models(request):
        return web.json_response({})

    upstream = web.Application()
    upstream.router.add_get("/v1/models", models)

    async def ps_ok():
        return True

    async with make_harness(
        upstream_app=upstream,
        upstream_state=UpstreamState(),
        readyz_ps_check=ps_ok,
        upstream="http://127.0.0.1:1",
    ) as h:
        resp = await h.client.get("/readyz")
        assert resp.status == 503
        assert "upstream" in (await resp.json())["reason"]


@pytest.mark.asyncio
async def test_readyz_not_ready_when_ps_fails():
    async def models(request):
        return web.json_response({"data": []})

    upstream = web.Application()
    upstream.router.add_get("/v1/models", models)

    async def ps_bad():
        return False

    async with make_harness(
        upstream_app=upstream, upstream_state=UpstreamState(), readyz_ps_check=ps_bad
    ) as h:
        resp = await h.client.get("/readyz")
        assert resp.status == 503
        assert "lms ps" in (await resp.json())["reason"]
