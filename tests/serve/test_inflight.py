from __future__ import annotations

import asyncio
import json
import logging
from unittest import mock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from laplace_serve import app as app_module
from laplace_serve.config import InflightConfig, LaplacedConfig
from laplace_serve.inflight import MAX_LINE_BYTES, InflightRecord, InflightRegistry
from tests.serve.harness import UpstreamState, make_harness


def _sse(**delta) -> bytes:
    return b"data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}).encode() + b"\n\n"


def _config(origins=(), tail_chars=400) -> LaplacedConfig:
    return LaplacedConfig(
        reaper_sweep=False,
        inflight=InflightConfig(origins=tuple(origins), tail_chars=tail_chars),
    )


def _gated_upstream(
    state: UpstreamState, chunks: list[bytes], gates: list[asyncio.Event], delay: float = 0.0
):
    """Stream chunks[i], then wait on gates[i] (if any) or delay before the next chunk."""

    async def handler(request: web.Request) -> web.StreamResponse:
        state.requests.append({"body": await request.read()})
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for index, chunk in enumerate(chunks):
            await resp.write(chunk)
            if index < len(gates):
                await gates[index].wait()
            elif delay:
                await asyncio.sleep(delay)
        await resp.write_eof()
        return resp

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", handler)
    return upstream


async def _post(h, origin="hermes", model="m", stream=True):
    return await h.client.post(
        f"/{origin}/v1/chat/completions",
        data=json.dumps({"model": model, "stream": stream}).encode(),
        headers={"Content-Type": "application/json"},
    )


async def _inflight(h) -> dict:
    resp = await h.client.get("/inflight")
    assert resp.status == 200
    return await resp.json()


async def _wait_empty(h) -> dict:
    for _ in range(200):
        body = await _inflight(h)
        if not body["inflight"]:
            return body
        await asyncio.sleep(0.01)
    return body


async def _read_all(resp) -> list[bytes]:
    pieces = []
    while True:
        piece = await resp.content.readany()
        if not piece:
            return pieces
        pieces.append(piece)


_CHAT_CHUNKS = [
    _sse(role="assistant"),
    _sse(reasoning_content="think "),
    _sse(content="Hello"),
    # A line split across two upstream chunks, then a garbage data line.
    _sse(content=", world")[:12],
    _sse(content=", world")[12:] + b"data: {not json\n\n",
    _sse(reasoning="!"),
    b"data: [DONE]\n\n",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("origins", [(), ("hermes",)])
async def test_stream_byte_identical_with_feature_on_and_off(origins):
    state = UpstreamState()
    upstream = _gated_upstream(state, _CHAT_CHUNKS, [], delay=0.03)
    async with make_harness(upstream_app=upstream, upstream_state=state, config=_config(origins)) as h:
        resp = await _post(h)
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "text/event-stream"
        pieces = await _read_all(resp)
        assert (await _wait_empty(h))["inflight"] == []

    assert b"".join(pieces) == b"".join(_CHAT_CHUNKS)
    assert len(pieces) >= 2  # still relayed incrementally


@pytest.mark.asyncio
async def test_record_grows_during_stream_then_is_removed(caplog):
    caplog.set_level(logging.DEBUG)
    state = UpstreamState()
    first = _sse(role="assistant") + _sse(content="Hello")
    second = _sse(reasoning_content=" there")
    gates = [asyncio.Event(), asyncio.Event()]
    upstream = _gated_upstream(state, [first, second, b"data: [DONE]\n\n"], gates)

    async with make_harness(
        upstream_app=upstream, upstream_state=state, config=_config(["hermes"])
    ) as h:
        resp = await _post(h)
        assert await resp.content.readexactly(len(first)) == first

        body = await _inflight(h)
        assert len(body["inflight"]) == 1
        record = body["inflight"][0]
        assert record["origin"] == "hermes"
        assert record["model"] == "m"
        assert record["stream"] is True
        assert record["bytes"] == len(first)
        assert record["chunks"] >= 1
        assert record["tail"] == "Hello"
        assert record["started_at"] <= record["admitted_at"] <= record["first_byte_at"]
        assert record["first_byte_at"] <= record["last_chunk_at"] <= body["now"]

        gates[0].set()
        assert await resp.content.readexactly(len(second)) == second
        grown = (await _inflight(h))["inflight"][0]
        assert grown["bytes"] == len(first) + len(second)
        assert grown["tail"] == "Hello there"
        assert grown["last_chunk_at"] >= record["last_chunk_at"]

        gates[1].set()
        await _read_all(resp)
        assert (await _wait_empty(h))["inflight"] == []

    assert "Hello" not in caplog.text  # tail content is never logged


@pytest.mark.asyncio
async def test_record_removed_after_client_disconnect():
    state = UpstreamState()

    async def stream_forever(request):
        await request.read()
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            for _ in range(500):
                await resp.write(_sse(content="x"))
                await asyncio.sleep(0.02)
        except Exception:  # noqa: BLE001 - client gone
            state.aborted = True
            raise
        return resp

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", stream_forever)
    async with make_harness(
        upstream_app=upstream, upstream_state=state, config=_config(["hermes"])
    ) as h:
        resp = await _post(h)
        assert await resp.content.readany()
        assert len((await _inflight(h))["inflight"]) == 1
        resp.close()
        assert (await _wait_empty(h))["inflight"] == []


@pytest.mark.asyncio
async def test_record_removed_after_upstream_error():
    async def die_midstream(request):
        await request.read()
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(_sse(content="partial"))
        request.transport.close()
        return resp

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", die_midstream)
    async with make_harness(upstream_app=upstream, config=_config(["hermes"])) as h:
        resp = await _post(h)
        try:
            await _read_all(resp)
        except Exception:  # noqa: BLE001 - truncated stream is expected
            pass
        assert (await _wait_empty(h))["inflight"] == []


@pytest.mark.asyncio
async def test_record_removed_after_upstream_connect_failure():
    async with make_harness(
        upstream_app=web.Application(), config=_config(["hermes"]), upstream="http://127.0.0.1:1"
    ) as h:
        resp = await _post(h, stream=False)
        assert resp.status == 502
        assert (await _inflight(h))["inflight"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("origins", "caller"), [(["hermes"], "bot"), ([], "hermes")])
async def test_untracked_origin_or_feature_off_leaves_no_record(origins, caller):
    state = UpstreamState()
    gates = [asyncio.Event()]
    upstream = _gated_upstream(state, [_sse(content="secret"), b"data: [DONE]\n\n"], gates)
    async with make_harness(upstream_app=upstream, upstream_state=state, config=_config(origins)) as h:
        resp = await _post(h, origin=caller)
        assert await resp.content.readany()
        assert await _inflight(h) == {"inflight": [], "now": mock.ANY}
        gates[0].set()
        await _read_all(resp)


@pytest.mark.asyncio
async def test_tail_is_bounded_at_tail_chars():
    state = UpstreamState()
    gates = [asyncio.Event()]
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    chunks = [_sse(content=text[:600]) + _sse(content=text[600:]), b"data: [DONE]\n\n"]
    upstream = _gated_upstream(state, chunks, gates)
    async with make_harness(
        upstream_app=upstream, upstream_state=state, config=_config(["hermes"], tail_chars=10)
    ) as h:
        resp = await _post(h)
        await resp.content.readexactly(len(chunks[0]))
        record = (await _inflight(h))["inflight"][0]
        assert record["tail"] == text[-10:]
        gates[0].set()
        await _read_all(resp)


def test_record_ignores_parse_failures_and_bounds_partial_lines():
    record = InflightRecord("hermes", "m", True, tail_chars=400)
    record.observe(b"data: {broken\n\n")
    record.observe(b"data: " + b"x" * (MAX_LINE_BYTES + 1))
    assert record._pending == b""  # oversized partial line dropped
    record.observe(b"still the same line\n" + _sse(content="ok"))
    assert record.tail == "ok"
    assert record.chunks == 3
    assert record.bytes == (
        len(b"data: {broken\n\n")
        + 6 + MAX_LINE_BYTES + 1
        + len(b"still the same line\n" + _sse(content="ok"))
    )


def test_non_stream_record_counts_bytes_without_tail():
    record = InflightRecord("hermes", "m", False, tail_chars=400)
    record.observe(_sse(content="ignored"))
    assert record.bytes == len(_sse(content="ignored"))
    assert record.tail == ""


@pytest.mark.asyncio
async def test_inflight_endpoint_rejects_non_loopback_peer():
    transport = mock.Mock()
    transport.get_extra_info.return_value = ("10.0.0.5", 40000)
    app = web.Application()
    app[app_module.CONFIG] = LaplacedConfig()
    app[app_module.INFLIGHT] = InflightRegistry(frozenset({"hermes"}))
    request = make_mocked_request("GET", "/inflight", app=app, transport=transport)
    resp = await app_module._inflight(request)
    assert resp.status == 403


def test_tail_concatenates_content_and_reasoning_across_split_lines():
    record = InflightRecord("hermes", "m", True, tail_chars=400)
    for chunk in _CHAT_CHUNKS:
        record.observe(chunk)
    assert record.tail == "think Hello, world!"
    assert record.bytes == len(b"".join(_CHAT_CHUNKS))
    assert record.chunks == len(_CHAT_CHUNKS)


_NESTED = b"data: " + b"[" * 10_000 + b"]" * 10_000 + b"\n"  # 20,007 bytes + newline


def test_deeply_nested_json_line_is_ignored_not_raised():
    record = InflightRecord("hermes", "m", True, tail_chars=400)
    record.observe(_NESTED)
    record.observe(_sse(content="after"))
    assert record.tail == "after"
    assert record.observer_failed is False


def test_oversized_complete_line_is_not_parsed():
    record = InflightRecord("hermes", "m", True, tail_chars=400)
    big = _sse(content="x" * (MAX_LINE_BYTES + 1))
    record.observe(big + _sse(content="ok"))
    assert record.tail == "ok"


async def _stream_and_collect(h, chunks):
    resp = await _post(h)
    assert resp.status == 200
    pieces = await _read_all(resp)  # returns only on EOF
    assert resp.content.at_eof()
    return b"".join(pieces)


@pytest.mark.asyncio
async def test_nested_brackets_line_stream_is_byte_identical_with_eof():
    chunks = [_sse(content="a"), _NESTED + b"\n", _sse(content="b"), b"data: [DONE]\n\n"]
    upstream = _gated_upstream(UpstreamState(), chunks, [], delay=0.02)
    async with make_harness(upstream_app=upstream, config=_config(["hermes"])) as h:
        received = await _stream_and_collect(h, chunks)
        assert (await _wait_empty(h))["inflight"] == []
    assert received == b"".join(chunks)


@pytest.mark.asyncio
async def test_observer_exception_is_contained_and_stream_is_byte_identical(
    monkeypatch, caplog
):
    calls = []

    def boom(self, chunk):
        calls.append(chunk)
        raise Exception("observer bug")

    monkeypatch.setattr(InflightRecord, "observe", boom)
    chunks = [_sse(content="secret-a"), _sse(content="b"), b"data: [DONE]\n\n"]
    gates = [asyncio.Event()]
    upstream = _gated_upstream(UpstreamState(), chunks, gates, delay=0.02)
    async with make_harness(upstream_app=upstream, config=_config(["hermes"])) as h:
        resp = await _post(h)
        first = await resp.content.readexactly(len(chunks[0]))
        record = (await _inflight(h))["inflight"][0]
        assert record["observer_failed"] is True
        gates[0].set()
        rest = await _read_all(resp)
        assert resp.content.at_eof()
        assert (await _wait_empty(h))["inflight"] == []

    assert first + b"".join(rest) == b"".join(chunks)
    assert len(calls) == 1  # observation stopped after the first failure
    assert "observer failed (Exception)" in caplog.text
    assert "secret-a" not in caplog.text
