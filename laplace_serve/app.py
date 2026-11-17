"""aiohttp application: governed routes, ungoverned passthrough, ops surface."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import os
import re
from urllib.parse import unquote
from typing import Awaitable, Callable, Optional

import aiohttp
from aiohttp import web

from laplace.adapter import InferenceAdapter, ModelLoadError
from laplace.adapters.lmstudio import LMStudioAdapter
from laplace.broker import ContentionBroker
from laplace.reaper import Reaper

from laplace_serve import origins
from laplace_serve._token_words import COST1, COST2
from laplace_serve.admission import govern
from laplace_serve.auth import CallerAuthenticator
from laplace_serve.config import LaplacedConfig
from laplace_serve.inflight import RECORD, InflightRegistry
from laplace_serve.proxy import UpstreamProxy
from laplace_serve.remote import RemoteFleet

log = logging.getLogger(__name__)

_DEFAULT_LMS_CANDIDATES = ("~/.lmstudio/bin/lms", "lms")

# Wait before each startup-validation retry after the catalog was unavailable
# at boot (laplaced can start before LM Studio is ready). Module-level so tests
# can monkeypatch it to zeros; production waits 10..60s, six retries max.
STARTUP_VALIDATION_RETRY_DELAYS: tuple[float, ...] = (10.0, 20.0, 40.0, 60.0, 60.0, 60.0)

CONFIG = web.AppKey("config", LaplacedConfig)
ADAPTER = web.AppKey("adapter", InferenceAdapter)
REAPER = web.AppKey("reaper", Reaper)
BROKER = web.AppKey("broker", ContentionBroker)
PROXY = web.AppKey("proxy", UpstreamProxy)
SESSION = web.AppKey("session", aiohttp.ClientSession)
FLEET = web.AppKey("fleet", RemoteFleet)
CALLER_AUTH = web.AppKey("caller_auth", Optional[CallerAuthenticator])
INFLIGHT = web.AppKey("inflight", InflightRegistry)
READYZ_PS_CHECK = web.AppKey("readyz_ps_check", object)
UPSTREAM_LISTENER_CHECK = web.AppKey("upstream_listener_check", object)
CATALOG_CHECK = web.AppKey("catalog_check", object)
STARTUP_VALIDATION_RETRY_TASK = web.AppKey(
    "startup_validation_retry_task", Optional[asyncio.Task]
)


def build_app(
    config: LaplacedConfig,
    *,
    adapter: InferenceAdapter,
    reaper: Reaper,
    broker: ContentionBroker,
    readyz_ps_check: Callable[[], Awaitable[bool]] | None = None,
    upstream_listener_check: Callable[[], Awaitable[bool]] | None = None,
    catalog_check: Callable[[str], Awaitable[str]] | None = None,
) -> web.Application:
    config.validate_governed_rest_auth()
    app = web.Application()
    app[CONFIG] = config
    app[ADAPTER] = adapter
    app[REAPER] = reaper
    app[BROKER] = broker
    app[FLEET] = RemoteFleet(config.backends, config.placement)
    app[INFLIGHT] = InflightRegistry(
        frozenset(config.inflight.origins), config.inflight.tail_chars
    )
    app[CALLER_AUTH] = (
        CallerAuthenticator(config.caller_verifiers) if config.governed_rest_auth else None
    )
    app[READYZ_PS_CHECK] = readyz_ps_check or (lambda: _lms_ps_ok(10.0))
    app[UPSTREAM_LISTENER_CHECK] = upstream_listener_check or (
        lambda: _upstream_listener_ok(2.0)
    )
    app[CATALOG_CHECK] = catalog_check or (lambda model: _lms_catalog_status(model, 10.0))

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    app.router.add_post("/{origin}/v1/chat/completions", _governed)
    app.router.add_post("/{origin}/v1/completions", _governed)
    app.router.add_post("/{origin}/v1/responses", _governed)
    app.router.add_post("/{origin}/v1/embeddings", _governed)
    if not config.governed_rest_auth:
        app.router.add_post("/v1/chat/completions", _governed)
        app.router.add_post("/v1/completions", _governed)
        app.router.add_post("/v1/responses", _governed)
        app.router.add_post("/v1/embeddings", _governed)
        app.router.add_get("/v1/{tail:.*}", _ungoverned)

    app.router.add_get("/{origin}/v1/models", _models)
    # LM Studio native model list (Hermes reads capabilities from it). Exact path only.
    app.router.add_get("/{origin}/api/v1/models", _models)

    app.router.add_get("/healthz", _healthz)
    app.router.add_get("/readyz", _readyz)
    app.router.add_get("/fleet/status", _fleet_status)
    app.router.add_get("/inflight", _inflight)
    if config.governed_rest_auth:
        app.router.add_get("/ops/upstream-listener", _upstream_listener)
        app.router.add_get("/ops/upstream-api", _upstream_api)
        app.router.add_get("/ops/catalog/{model:.*}", _catalog)
    return app


async def _on_startup(app: web.Application) -> None:
    config = app[CONFIG]
    adapter = app[ADAPTER]
    if isinstance(adapter, LMStudioAdapter):
        model_ids = (
            set(config.model_context)
            | set(config.model_footprint_mb)
            | set(config.model_parallel)
        )
        if not await adapter.validate_model_ids(model_ids):
            # Boot raced LM Studio startup: the catalog was unreachable, so
            # stale configured ids were never reported. Retry in the background
            # (never blocking or failing startup) until it succeeds or exhausts.
            app[STARTUP_VALIDATION_RETRY_TASK] = asyncio.create_task(
                _retry_startup_validation(adapter, model_ids),
                name="laplace:startup-validation-retry",
            )
    # total=None: the default aiohttp total=300s covers the ENTIRE streamed body
    # read, so any proxied generation longer than 5 minutes would die mid-stream
    # with a TimeoutError. Streams must be uncapped end to end; keep only a
    # connect-phase bound (sock_connect) so a dead upstream still fails fast.
    # probe_models passes its own explicit per-call timeout, and forward's 502
    # path still catches connect-phase timeouts.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    session = aiohttp.ClientSession(auto_decompress=False, timeout=timeout)
    app[SESSION] = session
    app[PROXY] = UpstreamProxy(
        session,
        config.upstream,
        config.upstream_auth_command if config.governed_rest_auth else (),
    )
    try:
        await app[PROXY].start()
    except Exception:
        await session.close()
        raise
    await app[FLEET].start()
    if config.adopt_resident_on_start:
        await _adopt_resident(app)
    if config.reaper_sweep:
        app[REAPER].start_sweep_loop(config.sweep_interval_s)


async def _adopt_resident(app: web.Application) -> None:
    """Adopt already-resident models as managed. Never fails startup."""
    try:
        entries = await app[ADAPTER].list_loaded()
        adopted = app[REAPER].adopt_resident(entries)
    except Exception:  # noqa: BLE001 - adoption is best-effort, startup must proceed
        log.exception("laplaced: adopting resident models at startup failed")
        return
    log.info("laplaced: adopted %d resident model(s) as managed: %s", len(adopted), adopted)


async def _retry_startup_validation(
    adapter: LMStudioAdapter, model_ids: set[str]
) -> None:
    """Retry startup model validation in the background after an unavailable catalog.

    Waits STARTUP_VALIDATION_RETRY_DELAYS between attempts (six retries max)
    and stops at the first successful validation. Never raises: a boot race
    must not take the daemon down, and _on_cleanup cancels us if we outlive it.
    """
    for attempt, delay in enumerate(STARTUP_VALIDATION_RETRY_DELAYS, start=1):
        await asyncio.sleep(delay)
        try:
            ok = await adapter.validate_model_ids(model_ids)
        except Exception:  # noqa: BLE001 - a background task must not die on a bug
            log.exception("lmstudio startup model validation retry %d raised", attempt)
            ok = False
        if ok:
            log.info("lmstudio startup model validation retry %d succeeded", attempt)
            return
    log.error(
        "lmstudio startup model validation gave up after %d retries",
        len(STARTUP_VALIDATION_RETRY_DELAYS),
    )


async def _on_cleanup(app: web.Application) -> None:
    app[REAPER].stop_sweep_loop()
    await app[FLEET].close()
    # Cancel the retry task BEFORE closing the session: a failure while
    # closing must not skip the cancel and leave the task running.
    task = app.get(STARTUP_VALIDATION_RETRY_TASK)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    session = app.get(SESSION)
    if session is not None:
        await session.close()


async def _governed(request: web.Request) -> web.StreamResponse:
    config = request.app[CONFIG]
    path_origin = request.match_info.get("origin")
    if config.governed_rest_auth:
        denied = request.app[CALLER_AUTH].authenticate(request, path_origin)
        if denied is not None:
            return denied
        origin, tier = path_origin, config.tier_for(path_origin)
    else:
        origin, tier = origins.resolve(path_origin, request.headers, config)

    body = await request.read()
    requested_model, stream = _extract_model_stream(body)
    model = config.canonical_model_id(requested_model)
    forwarded_body = _canonicalize_forwarded_model(body, requested_model, model)
    ctx = config.model_context.get(model, config.default_context_length) if model else None
    allowance = _output_allowance(config, ctx)
    need_ctx = (
        _estimate_need_tokens(
            body, allowance, config.fit_context_word_estimate
        )
        if model and config.fit_context_admission
        else None
    )

    tracked = None
    if _upstream_path(request) == "/v1/chat/completions":
        tracked = request.app[INFLIGHT].start(origin, model, stream)
    if tracked is None:
        return await _route(request, origin, tier, model, stream, ctx, forwarded_body, need_ctx)
    key, record = tracked
    request[RECORD] = record
    try:
        return await _route(request, origin, tier, model, stream, ctx, forwarded_body, need_ctx)
    finally:
        request.app[INFLIGHT].finish(key)


async def _route(
    request: web.Request,
    origin: str,
    tier: str,
    model: str | None,
    stream: bool,
    ctx: int | None,
    forwarded_body: bytes,
    need_ctx: int | None = None,
) -> web.StreamResponse:
    for backend_name in request.app[FLEET].route(origin, model):
        if backend_name == "axis":
            return await _forward_axis(
                request, origin, tier, model, stream, ctx, forwarded_body, need_ctx
            )
        result = await request.app[FLEET].backends[backend_name].forward(
            request, _upstream_path(request), forwarded_body
        )
        if result.response is not None:
            log.info(
                "route origin=%s model=%s backend=%s reason=served",
                origin,
                model or "-",
                backend_name,
            )
            return result.response
        log.info(
            "route origin=%s model=%s backend=%s reason=fallback:%s",
            origin,
            model or "-",
            backend_name,
            result.failure_reason,
        )
    return web.json_response({"error": "no backend available"}, status=502)


def _client_gone(request: web.Request) -> bool:
    """The proxy's disconnect test (proxy._unless_disconnected), used while queued."""
    transport = request.transport
    return transport is None or transport.is_closing()


async def _forward_axis(
    request: web.Request,
    origin: str,
    tier: str,
    model: str | None,
    stream: bool,
    ctx: int | None,
    forwarded_body: bytes,
    need_ctx: int | None = None,
) -> web.StreamResponse:
    config = request.app[CONFIG]

    try:
        async with govern(
            request.app[BROKER],
            request.app[REAPER],
            request.app[ADAPTER],
            model=model,
            ctx=ctx,
            tier=tier,
            fail_open=config.fail_open,
            need_ctx=need_ctx,
            abandoned=(
                (lambda: _client_gone(request))
                if config.cancel_admission_on_disconnect
                else None
            ),
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
            if admission.decision == "abandoned":
                # aiohttp treats a ConnectionError from a handler as a premature
                # client disconnection and drops the connection quietly.
                raise ConnectionResetError("downstream client disconnected")
            if not admission.proxy:
                return web.json_response(
                    {"error": "inference pool at capacity, retry later"},
                    status=503,
                    headers={"Retry-After": "30"},
                )
            record = request.get(RECORD)
            if record is not None:
                record.mark_admitted()
            return await request.app[PROXY].forward(request, _upstream_path(request), forwarded_body)
    except ModelLoadError as exc:
        remaining = request.app[ADAPTER].guard_window_remaining_s()
        # The OpenAI Python SDK ignores Retry-After values above 60 seconds.
        retry_after = str(min(60, max(1, math.ceil(remaining)))) if remaining is not None else "5"
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
            headers={"Retry-After": retry_after},
        )


async def _ungoverned(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    return await request.app[PROXY].forward(request, _upstream_path(request), body)


async def _models(request: web.Request) -> web.StreamResponse:
    config = request.app[CONFIG]
    if config.governed_rest_auth:
        denied = request.app[CALLER_AUTH].authenticate(
            request, request.match_info["origin"]
        )
        if denied is not None:
            return denied
    return await _ungoverned(request)


async def _healthz(request: web.Request) -> web.Response:
    # Liveness only: the process is up and serving. No upstream calls.
    return web.json_response({"status": "ok"})


async def _fleet_status(request: web.Request) -> web.Response:
    denied = _authenticate_status(request)
    if denied is not None:
        return denied
    return web.json_response(request.app[FLEET].snapshot())


async def _inflight(request: web.Request) -> web.Response:
    # Tail text is model output: serve it to loopback callers only, whatever
    # the listener's bind address, and behind the same auth as /fleet/status.
    try:
        loopback = ipaddress.ip_address(request.remote or "").is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        return web.json_response({"error": "forbidden"}, status=403)
    denied = _authenticate_status(request)
    if denied is not None:
        return denied
    return web.json_response(request.app[INFLIGHT].snapshot())


async def _readyz(request: web.Request) -> web.Response:
    denied = _authenticate_status(request)
    if denied is not None:
        return denied
    config = request.app[CONFIG]
    proxy = request.app[PROXY]
    models_ok = (
        await proxy.probe_api(timeout=5.0)
        if config.governed_rest_auth
        else await proxy.probe_models(timeout=5.0)
    )
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
        reasons.append(
            "upstream API unavailable within 5s"
            if config.governed_rest_auth
            else "upstream /v1/models unreachable within 5s"
        )
    if not ps_ok:
        reasons.append("lms ps not rc 0 within 10s")
    if sweep_state not in {"healthy", "disabled"}:
        reasons.append(f"reaper sweep {sweep_state}")
    return web.json_response(
        {"status": "not ready", "reason": "; ".join(reasons), "reaper_sweep": sweep_state},
        status=503,
    )


def _authenticate_status(request: web.Request) -> web.Response | None:
    config = request.app[CONFIG]
    if not config.governed_rest_auth:
        return None
    return request.app[CALLER_AUTH].authenticate(request, None)


async def _upstream_listener(request: web.Request) -> web.Response:
    denied = _authenticate_status(request)
    if denied is not None:
        return denied
    if await request.app[UPSTREAM_LISTENER_CHECK]():
        return web.json_response({"status": "listening"})
    return web.json_response({"status": "not-listening"}, status=503)


async def _upstream_api(request: web.Request) -> web.Response:
    denied = _authenticate_status(request)
    if denied is not None:
        return denied
    if await request.app[PROXY].probe_api(timeout=5.0):
        return web.json_response({"status": "ready"})
    return web.json_response({"status": "unavailable"}, status=503)


async def _catalog(request: web.Request) -> web.Response:
    denied = _authenticate_status(request)
    if denied is not None:
        return denied
    model = unquote(request.match_info["model"])
    if not model or len(model) > 300 or any(ord(char) < 0x20 for char in model):
        return web.json_response({"error": "invalid_model"}, status=400)
    status = await request.app[CATALOG_CHECK](model)
    if status == "available":
        return web.json_response({"status": status, "model": model})
    if status == "missing":
        return web.json_response({"status": status, "model": model}, status=404)
    return web.json_response(
        {"status": "unavailable", "reason": "control-plane"}, status=503
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


# Output-limit fields across the governed routes: chat/completions, the legacy
# completions route, and responses.
_OUTPUT_LIMIT_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


# Chat-template tokens that have no bytes in the request body: default system
# prompts and tool-calling instructions. Worst seen: Qwen3-Coder rendered a
# 222-byte request with one tool to 257 tokens (see _estimate_need_tokens).
_TEMPLATE_ALLOWANCE_TOKENS = 1024


def _output_allowance(config: LaplacedConfig, ctx: int | None) -> int:
    """Output allowance for a request that sends no max_tokens.

    With ``fit_context_output_from_window`` on and a configured context
    available, the model's whole window is allowed; otherwise the fixed
    ``fit_context_min_output_tokens`` applies. A caller's own max_tokens always
    wins (see ``_estimate_need_tokens``).
    """
    if config.fit_context_output_from_window and ctx:
        return ctx
    return config.fit_context_min_output_tokens


def _estimate_need_tokens(
    body: bytes, min_output_tokens: int, word_estimate: bool = False
) -> int:
    """Upper-bound context a request needs: prompt plus output allowance.

    Prompt: the request body's UTF-8 byte count plus
    ``_TEMPLATE_ALLOWANCE_TOKENS``. Without a tokenizer (the core is
    stdlib-only) bytes are the bound: byte-level BPE and byte-fallback
    SentencePiece emit at most one token per byte of text, so 12,000 random
    digits cost about 12,000 tokens on Qwen and Gemma. JSON escaping makes the
    body longer than the strings the template renders, and each message's JSON
    syntax (about 25 bytes) outweighs its template tokens. Offline, five local
    tokenizers (GLM-4.5-Air, gpt-oss, Qwen3-Coder, Qwen3-Next, Gemma 4) on 15
    adversarial bodies (digits, punctuation, CJK, emoji, 400 empty messages,
    300 tools, 200-parameter schemas) never rendered more than 35 tokens above
    body bytes. An empirical bound, not a proof for every template. For
    English it is about 4.3x the real count; a need above the configured
    context is treated as the configured context.
    ``word_estimate`` lowers the prompt part by ``_text_savings`` of each
    message text; everything else (JSON syntax, tools, tool-call arguments)
    stays counted by bytes.
    Output: the caller's limit when it sends one, else ``min_output_tokens``.
    """
    prompt = len(body) + _TEMPLATE_ALLOWANCE_TOKENS
    output = min_output_tokens
    try:
        payload = json.loads(body) if body else None
    except (ValueError, UnicodeDecodeError):
        payload = None
    if isinstance(payload, dict):
        for key in _OUTPUT_LIMIT_KEYS:
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                output = value
                break
        if word_estimate:
            prompt -= sum(_text_savings(text) for text in _message_texts(payload))
    return prompt + output


# Message text is rendered verbatim by chat templates, so only it is
# discounted. Tool schemas and arguments are not: Qwen3-Coder expands each
# tool parameter into XML, and its byte count is what covers that expansion.
def _message_texts(payload: dict):
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    yield part["text"]


# A word is a run of ASCII letters with no letter, digit or non-ASCII character
# on either side, so base64, hex and identifiers glued to digits are never
# discounted. Runs of two or more spaces are matched separately.
_TEXT_RUN = re.compile(r" {2,}|[A-Za-z0-9\x80-\U0010ffff]+")
_WORD_SHAPE = re.compile(r"[A-Z]?[a-z]+")


def _text_savings(text: str) -> int:
    """How many tokens fewer than its UTF-8 bytes a message text can cost.

    Per-class costs, checked offline on the five local tokenizers named in
    ``_estimate_need_tokens``: a word in ``COST1`` or ``COST2`` costs 1 or 2
    tokens (lowercase or capitalised, with or without a leading space,
    regenerated and checked by scripts/gen-token-words.py), and a single
    space before such a word is free. A run of k spaces costs ceil(k / 16)
    (one token holds 31 spaces on Gemma 4, at least 64 on the others). Every
    other character keeps its byte count. The text's JSON escaping in the body
    is not discounted, so the saving never exceeds the bytes the text adds.
    """
    saved = 0
    for match in _TEXT_RUN.finditer(text):
        run = match.group()
        if run[0] == " ":
            saved += len(run) - -(-len(run) // 16)
            continue
        if not _WORD_SHAPE.fullmatch(run):
            continue
        word = run.lower()
        if word in COST1:
            cost = 1
        elif word in COST2:
            cost = 2
        else:
            continue
        start = match.start()
        lead = start > 0 and text[start - 1] == " " and (start < 2 or text[start - 2] != " ")
        saved += len(run) - cost + lead
    return saved


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


async def _upstream_listener_ok(timeout: float) -> bool:
    writer = None
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", 1234), timeout=timeout
        )
        return True
    except (OSError, asyncio.TimeoutError):
        return False
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()


async def _lms_catalog_status(model: str, timeout: float) -> str:
    for cli in (os.path.expanduser(_DEFAULT_LMS_CANDIDATES[0]), _DEFAULT_LMS_CANDIDATES[1]):
        try:
            proc = await asyncio.create_subprocess_exec(
                cli,
                "ls",
                "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            continue
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "unavailable"
        if proc.returncode != 0:
            return "unavailable"
        try:
            entries = json.loads(stdout.decode() or "[]")
        except (UnicodeDecodeError, ValueError):
            return "unavailable"
        if not isinstance(entries, list):
            return "unavailable"
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            identifiers = (
                entry.get("modelKey"),
                entry.get("path"),
                entry.get("identifier"),
            )
            if model in identifiers:
                return "available"
        return "missing"
    return "unavailable"
