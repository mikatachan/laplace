"""Test harness: fake LM Studio upstream + a live laplaced app under a client."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from aiohttp.web_runner import AppRunner

from laplace.broker import ContentionBroker
from laplace.reaper import Reaper

from laplace_serve.app import build_app
from laplace_serve.config import LaplacedConfig
from tests.fakes import FakeAdapter


class ProductionServer(TestServer):
    """TestServer with laplaced's production handler_cancellation.

    aiohttp's TestServer hard-codes handler_cancellation=True, but web.run_app
    (laplace_serve.__main__) runs with aiohttp's default False: a handler is
    not cancelled when its client disconnects. Tests must see the same.
    """

    async def _make_runner(self, **kwargs):
        kwargs["handler_cancellation"] = False
        return AppRunner(self.app, **kwargs)


class Harness:
    def __init__(self, client, adapter, reaper, broker, upstream_state):
        self.client = client
        self.adapter = adapter
        self.reaper = reaper
        self.broker = broker
        self.upstream = upstream_state


class UpstreamState:
    """Records what the fake upstream observed for assertions."""

    def __init__(self):
        self.requests: list[dict] = []
        self.aborted = False


@asynccontextmanager
async def make_harness(
    *,
    upstream_app: web.Application,
    upstream_state: UpstreamState | None = None,
    config: LaplacedConfig | None = None,
    adapter: FakeAdapter | None = None,
    reaper: Reaper | None = None,
    broker: ContentionBroker | None = None,
    readyz_ps_check: Callable[[], Awaitable[bool]] | None = None,
    upstream_listener_check: Callable[[], Awaitable[bool]] | None = None,
    catalog_check: Callable[[str], Awaitable[str]] | None = None,
    upstream: str | None = None,
):
    upstream_server = TestServer(upstream_app)
    await upstream_server.start_server()
    upstream_url = upstream or str(upstream_server.make_url("")).rstrip("/")

    adapter = adapter or FakeAdapter()
    reaper = reaper or Reaper(adapter, delay_seconds=300.0)
    broker = broker or ContentionBroker(adapter, reaper, budget_mb=90000)

    base = config or LaplacedConfig(reaper_sweep=False)
    config = _with_upstream(base, upstream_url)

    app = build_app(
        config,
        adapter=adapter,
        reaper=reaper,
        broker=broker,
        readyz_ps_check=readyz_ps_check,
        upstream_listener_check=upstream_listener_check,
        catalog_check=catalog_check,
    )
    client = TestClient(ProductionServer(app))
    await client.start_server()

    harness = Harness(client, adapter, reaper, broker, upstream_state or UpstreamState())
    try:
        yield harness
    finally:
        await client.close()
        await upstream_server.close()


def _with_upstream(config: LaplacedConfig, upstream: str) -> LaplacedConfig:
    from dataclasses import replace

    return replace(config, upstream=upstream)
