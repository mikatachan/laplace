"""aiohttp application: governed routes, ungoverned passthrough, ops surface."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Awaitable, Callable

import aiohttp
from aiohttp import web

from laplace.adapter import InferenceAdapter, ModelLoadError
from laplace.broker import ContentionBroker
from laplace.reaper import Reaper

from laplace_serve import origins
from laplace_serve.admission import govern
from laplace_serve.config import LaplacedConfig
from laplace_serve.proxy import UpstreamProxy

log = logging.getLogger(__name__)

_DEFAULT_LMS_CANDIDATES = ("~/.lmstudio/bin/lms", "lms")

CONFIG = web.AppKey("config", LaplacedConfig)
ADAPTER = web.AppKey("adapter", InferenceAdapter)
REAPER = web.AppKey("reaper", Reaper)
BROKER = web.AppKey("broker", ContentionBroker)
PROXY = web.AppKey("proxy", UpstreamProxy)
SESSION = web.AppKey("session", aiohttp.ClientSession)
READYZ_PS_CHECK = web.AppKey("readyz_ps_check", object)


def build_app(
    config: LaplacedConfig,
    *,
    adapter: InferenceAdapter,
    reaper: Reaper,
    broker: ContentionBroker,
    readyz_ps_check: Callable[[], Awaitable[bool]] | None = None,
) -> web.Application:
    app = web.Application()
    app[CONFIG] = config
    app[ADAPTER] = adapter
    app[REAPER] = reaper
    app[BROKER] = broker
    app[READYZ_PS_CHECK] = readyz_ps_check or (lambda: _lms_ps_ok(10.0))

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    app.router.add_post("/{origin}/v1/chat/completions", _governed)
    app.router.add_post("/{origin}/v1/responses", _governed)
    app.router.add_post("/{origin}/v1/embeddings", _governed)
    app.router.add_post("/v1/chat/completions", _governed)
    app.router.add_post("/v1/responses", _governed)
    app.router.add_post("/v1/embeddings", _governed)

    app.router.add_get("/{origin}/v1/models", _ungoverned)
    app.router.add_get("/v1/{tail:.*}", _ungoverned)

    app.router.add_get("/healthz", _healthz)
    app.router.add_get("/readyz", _readyz)
    return app


async def _on_startup(app: web.Application) -> None:
    config = app[CONFIG]
    # total=None: the default aiohttp total=300s covers the ENTIRE streamed body
    # read, so any proxied generation longer than 5 minutes would die mid-stream
    # with a TimeoutError. Streams must be uncapped end to end; keep only a
    # connect-phase bound (sock_connect) so a dead upstream still fails fast.
    # probe_models passes its own explicit per-call timeout, and forward's 502
    # path still catches connect-phase timeouts.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    session = aiohttp.ClientSession(auto_decompress=False, timeout=timeout)
    app[SESSION] = session
    app[PROXY] = UpstreamProxy(session, config.upstream)
    if config.reaper_sweep:
        app[REAPER].start_sweep_loop(config.sweep_interval_s)


async def _on_cleanup(app: web.Application) -> None:
    app[REAPER].stop_sweep_loop()
    session = app.get(SESSION)
    if session is not None:
        await session.close()


async def _governed(request: web.Request) -> web.StreamResponse:
    config = request.app[CONFIG]
    origin, tier = origins.resolve(request.match_info.get("origin"), request.headers, config)

    body = await request.read()
    requested_model, stream = _extract_model_stream(body)
    model = config.canonical_model_id(requested_model)
    forwarded_body = _canonicalize_forwarded_model(body, requested_model, model)
    ctx = config.model_context.get(model, config.default_context_length) if model else None

    try:
        async with govern(
            request.app[BROKER],
            request.app[REAPER],
            request.app[ADAPTER],
            model=model,
            ctx=ctx,
            tier=tier,
            fail_open=config.fail_open,
        ) as admission:
            log.info(
                "admission origin=%s tier=%s model=%s decision=%s wait_ms=%s stream=%s",
                origin,
                tier,
                model or "-",
                admission.decision,
                admission.wait_ms,
                stream,
            )
            if not admission.proxy:
                return web.json_response(
                    {"error": "inference pool at capacity, retry later"},
                    status=503,
                    headers={"Retry-After": "30"},
                )
            return await request.app[PROXY].forward(request, _upstream_path(request), forwarded_body)
    except ModelLoadError as exc:
        log.warning(
            "admission origin=%s tier=%s model=%s decision=load-failed detail=%s",
            origin,
            tier,
            model or "-",
            str(exc)[:200],
        )
        return web.json_response(
            {"error": "model unavailable; load did not complete"},
            status=503,
            headers={"Retry-After": "5"},
        )


async def _ungoverned(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    return await request.app[PROXY].forward(request, _upstream_path(request), body)


async def _healthz(request: web.Request) -> web.Response:
    # Liveness only: the process is up and serving. No upstream calls.
    return web.json_response({"status": "ok"})


async def _readyz(request: web.Request) -> web.Response:
    config = request.app[CONFIG]
    proxy = request.app[PROXY]
    models_ok = await proxy.probe_models(timeout=5.0)
    ps_ok = await request.app[READYZ_PS_CHECK]()
    sweep_state = (
        request.app[REAPER].sweep_health(config.sweep_interval_s * 2)
        if config.reaper_sweep
        else "disabled"
    )
    if models_ok and ps_ok and sweep_state in {"healthy", "disabled"}:
        return web.json_response({"status": "ready", "reaper_sweep": sweep_state})
    reasons = []
    if not models_ok:
        reasons.append("upstream /v1/models unreachable within 5s")
    if not ps_ok:
        reasons.append("lms ps not rc 0 within 10s")
    if sweep_state not in {"healthy", "disabled"}:
        reasons.append(f"reaper sweep {sweep_state}")
    return web.json_response(
        {"status": "not ready", "reason": "; ".join(reasons), "reaper_sweep": sweep_state},
        status=503,
    )


def _upstream_path(request: web.Request) -> str:
    origin = request.match_info.get("origin")
    if origin:
        # Strip the /{origin} prefix so the upstream sees a plain /v1/... path.
        return request.path[len("/" + origin):]
    return request.path


def _extract_model_stream(body: bytes) -> tuple[str | None, bool]:
    if not body:
        return None, False
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None, False
    if not isinstance(payload, dict):
        return None, False
    model = payload.get("model")
    model = model if isinstance(model, str) and model else None
    return model, bool(payload.get("stream"))


def _canonicalize_forwarded_model(
    body: bytes, requested_model: str | None, canonical_model: str | None
) -> bytes:
    """Replace a normalized request model without altering no-op bodies.

    Admission uses canonical configured IDs.  When normalization changed the
    caller's bare ID, the upstream must receive that same configured key rather
    than relying on LM Studio to resolve the bare spelling independently.
    """
    if not requested_model or requested_model == canonical_model:
        return body
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict) or payload.get("model") != requested_model:
        return body
    payload["model"] = canonical_model
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


async def _lms_ps_ok(timeout: float) -> bool:
    for cli in (os.path.expanduser(_DEFAULT_LMS_CANDIDATES[0]), _DEFAULT_LMS_CANDIDATES[1]):
        try:
            proc = await asyncio.create_subprocess_exec(
                cli,
                "ps",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            continue
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return False
        return proc.returncode == 0
    return False
