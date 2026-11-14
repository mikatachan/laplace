"""Remote backend credentials, health, and pre-commit routing."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Mapping

import aiohttp
from aiohttp import web

from laplace_serve.config import RemoteBackendConfig
from laplace_serve.proxy import filter_request_headers, relay_after_commit

log = logging.getLogger(__name__)

PROBE_INTERVAL_S = 120.0
PROBE_TIMEOUT_S = 10.0
# A non-UP backend may need a cold JIT model load (~41 s measured on shinobi).
PROBE_COLD_TIMEOUT_S = 60.0
QUEUE_TIMEOUT_S = 10.0
FIRST_BYTE_TIMEOUT_S = 120.0
FALLBACK_HTTP_STATUSES = frozenset({400, 404, 429})
PRECOMMIT_EVENT_LIMIT = 64 * 1024
UPSTREAM_ERROR_READ_LIMIT = 8 * 1024
UPSTREAM_ERROR_LOG_LIMIT = 160


@dataclass(frozen=True)
class AttemptResult:
    response: web.StreamResponse | None = None
    failure_reason: str | None = None
    failure_message: str | None = None


@dataclass
class ModelHealth:
    state: str = "UP"
    reason: str | None = None
    consecutive_failures: int = 0


class RemoteBackend:
    """One remote backend. Credentials never leave this object except in headers."""

    def __init__(self, name: str, config: RemoteBackendConfig, *, active: bool):
        self.name = name
        self.config = config
        self.state = "DISABLED" if not active else "DOWN"
        self.reason = "unplaced" if not active else "not-probed"
        self.active_url: str | None = None
        self.last_probe_at: float | None = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.in_flight = 0
        self.queued = 0
        self._active = active
        self._credential: str | None = None
        self._session: aiohttp.ClientSession | None = None
        self._probe_task: asyncio.Task | None = None
        self._slots = asyncio.Semaphore(config.slots)
        self._model_health: dict[str, ModelHealth] = {}

    def _set_state(self, state: str, reason: str | None) -> None:
        previous = self.state
        self.state = state
        self.reason = reason
        if state != previous:
            log.info(
                "backend %s state %s->%s reason=%s", self.name, previous, state, reason
            )

    def ensure_model(self, model: str) -> ModelHealth:
        return self._model_health.setdefault(model, ModelHealth())

    def _set_model_state(self, model: str, state: str, reason: str | None) -> None:
        health = self.ensure_model(model)
        previous = health.state
        health.state = state
        health.reason = reason
        if state != previous:
            log.info(
                "backend %s model %s state %s->%s reason=%s",
                self.name,
                model,
                previous,
                state,
                reason,
            )

    async def start(self) -> None:
        if not self._active:
            return
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=3)
        self._session = aiohttp.ClientSession(auto_decompress=False, timeout=timeout)
        credential_ready = await self._refresh_credential()
        self._probe_task = asyncio.create_task(
            self._probe_loop(delay_first=not credential_ready),
            name=f"laplace:probe:{self.name}",
        )

    async def close(self) -> None:
        if self._probe_task is not None and not self._probe_task.done():
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
        if self._session is not None:
            await self._session.close()

    async def _run_auth_command(self) -> str | None:
        """Run argv without ever exposing argv, stdout, or stderr."""
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.config.auth_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except asyncio.TimeoutError:
            if proc is not None:
                proc.kill()
                await proc.wait()
            return None
        except Exception:  # noqa: BLE001 - details might contain sensitive argv
            return None
        if proc.returncode != 0:
            return None
        try:
            value = stdout.decode().strip()
        except UnicodeDecodeError:
            return None
        return value or None

    async def _refresh_credential(self) -> bool:
        credential = await self._run_auth_command()
        if credential is None:
            self._credential = None
            self._set_state("DISABLED", "no-credential")
            return False
        if not _is_header_safe_token(credential):
            self._credential = None
            self._set_state("DISABLED", "bad-credential")
            return False
        self._credential = credential
        if self.state == "DISABLED":
            self._set_state("DOWN", "not-probed")
        return True

    async def _probe_loop(self, *, delay_first: bool = False) -> None:
        if delay_first:
            await asyncio.sleep(PROBE_INTERVAL_S)
        while True:
            try:
                await self.probe_once()
            except Exception:  # noqa: BLE001 - keep the background probe alive
                self.last_probe_at = time.time()
                self.record_probe_failure("probe-error")
                log.exception("backend %s probe failed unexpectedly", self.name)
            await asyncio.sleep(PROBE_INTERVAL_S)

    async def probe_once(self) -> bool:
        if self._session is None:
            return False
        if self._credential is None and not await self._refresh_credential():
            self.last_probe_at = time.time()
            return False
        payload = json.dumps(
            {
                "model": self.config.probe_model,
                "messages": [{"role": "user", "content": "Reply with OK"}],
                "max_tokens": 4,
                "temperature": 0,
            },
            separators=(",", ":"),
        ).encode()
        probe_health = self.ensure_model(self.config.probe_model)
        timeout_s = (
            PROBE_TIMEOUT_S
            if self.state == "UP" and probe_health.state == "UP"
            else PROBE_COLD_TIMEOUT_S
        )
        success_url = None
        failure_reason = "probe-failed"
        for base_url in self._ordered_urls():
            try:
                async with self._session.post(
                    base_url + "/v1/chat/completions",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {self._credential}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout_s),
                ) as response:
                    if response.status == 401:
                        await self._refresh_credential()
                        failure_reason = "auth"
                        break
                    if response.status != 200:
                        failure_reason = "probe-http"
                        continue
                    try:
                        response_body = await response.json()
                    except (ValueError, aiohttp.ClientError):
                        failure_reason = "probe-body"
                        continue
                    choices = (
                        response_body.get("choices")
                        if isinstance(response_body, dict)
                        else None
                    )
                    if (
                        isinstance(choices, list)
                        and choices
                        and isinstance(choices[0], dict)
                        and isinstance(choices[0].get("message"), dict)
                    ):
                        success_url = base_url
                        break
                    failure_reason = "probe-body"
            except asyncio.TimeoutError:
                failure_reason = "probe-timeout"
            except aiohttp.ClientError:
                failure_reason = "probe-connect"
        self.last_probe_at = time.time()
        if success_url is not None:
            self.active_url = success_url
            self.record_probe_success()
            return True
        self.record_probe_failure(failure_reason)
        return False

    def record_probe_success(self) -> None:
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        if self.consecutive_successes >= 2:
            self._set_state("UP", None)
            for model, health in self._model_health.items():
                health.consecutive_failures = 0
                self._set_model_state(model, "UP", None)

    def record_probe_failure(self, reason: str = "probe-failed") -> None:
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        if self.state == "DISABLED":
            return
        if self.consecutive_failures >= 2:
            self._set_state("DOWN", reason)

    def record_request_failure(
        self, reason: str, model: str | None = None, *, host_wide: bool = False
    ) -> None:
        self.consecutive_successes = 0
        if self.state == "DISABLED":
            return
        if host_wide or model is None:
            self.consecutive_failures += 1
            self._set_state("DOWN", reason)
            return
        health = self.ensure_model(model)
        health.consecutive_failures += 1
        self._set_model_state(model, "DOWN", reason)

    async def forward(
        self,
        request: web.Request,
        upstream_path: str,
        body: bytes,
        model: str,
        stream: bool,
    ) -> AttemptResult:
        if self.state != "UP" or self._session is None or self._credential is None:
            return AttemptResult(failure_reason=f"state:{self.state.lower()}")
        if self.ensure_model(model).state != "UP":
            return AttemptResult(failure_reason="state:model-down")

        self.queued += 1
        try:
            try:
                await asyncio.wait_for(
                    self._slots.acquire(), timeout=QUEUE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                return AttemptResult(failure_reason="queue")
        finally:
            self.queued -= 1
        self.in_flight += 1
        try:
            if self.state != "UP":
                return AttemptResult(failure_reason="state:down")
            return await self._forward_acquired(
                request, upstream_path, body, model, stream
            )
        finally:
            self.in_flight -= 1
            self._slots.release()

    async def _forward_acquired(
        self,
        request: web.Request,
        upstream_path: str,
        body: bytes,
        model: str,
        stream: bool,
    ) -> AttemptResult:
        base_url = self.active_url or self.config.url
        headers = {
            key: value
            for key, value in filter_request_headers(request.headers).items()
            if key.lower() != "authorization"
        }
        headers["Authorization"] = f"Bearer {self._credential}"
        response = None
        try:
            response = await asyncio.wait_for(
                self._session.request(
                    request.method,
                    base_url + upstream_path,
                    headers=headers,
                    params=request.query.copy() if request.query_string else None,
                    data=body or None,
                ),
                timeout=FIRST_BYTE_TIMEOUT_S,
            )
            if response.status == 401:
                message = await _bounded_upstream_error_message(response)
                response.close()
                await self._refresh_credential()
                self.record_request_failure("auth", model)
                return AttemptResult(failure_reason="auth", failure_message=message)
            if response.status in FALLBACK_HTTP_STATUSES:
                status = response.status
                message = await _bounded_upstream_error_message(response)
                response.close()
                reason = f"upstream-{status}"
                return AttemptResult(failure_reason=reason, failure_message=message)
            if 500 <= response.status <= 599:
                message = await _bounded_upstream_error_message(response)
                response.close()
                self.record_request_failure("upstream-5xx", model)
                return AttemptResult(
                    failure_reason="upstream-5xx", failure_message=message
                )
            try:
                first_chunk, error_message = await asyncio.wait_for(
                    _inspect_precommit_response(response, stream),
                    timeout=FIRST_BYTE_TIMEOUT_S,
                )
            except (aiohttp.ClientError, asyncio.TimeoutError):
                response.close()
                self.record_request_failure("first-byte", model)
                return AttemptResult(failure_reason="first-byte")
            if error_message is not None:
                response.close()
                return AttemptResult(
                    failure_reason="upstream-inband-error",
                    failure_message=error_message,
                )
        except asyncio.TimeoutError:
            if response is not None:
                response.close()
            self.record_request_failure("connect-timeout", host_wide=True)
            return AttemptResult(failure_reason="connect-timeout")
        except aiohttp.ClientError:
            if response is not None:
                response.close()
            self.record_request_failure("connect", host_wide=True)
            return AttemptResult(failure_reason="connect")

        committed = await relay_after_commit(request, response, first_chunk)
        return AttemptResult(response=committed)

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "reason": self.reason,
            "active_url": self.active_url,
            "last_probe_at": self.last_probe_at,
            "consecutive_failures": self.consecutive_failures,
            "in_flight": self.in_flight,
            "queued": self.queued,
            "models": {
                model: {
                    "state": health.state,
                    "reason": health.reason,
                    "consecutive_failures": health.consecutive_failures,
                }
                for model, health in sorted(self._model_health.items())
            },
        }

    def _ordered_urls(self) -> tuple[str, ...]:
        urls = (self.config.url,) + self.config.alt_urls
        if self.active_url is None or self.active_url not in urls:
            return urls
        return (self.active_url,) + tuple(url for url in urls if url != self.active_url)


def _is_header_safe_token(value: str) -> bool:
    """Accept non-empty visible ASCII without whitespace or control bytes."""
    return bool(value) and all(0x21 <= ord(char) <= 0x7E for char in value)


async def _inspect_precommit_response(
    response: aiohttp.ClientResponse, stream: bool
) -> tuple[bytes, str | None]:
    if not stream:
        buffered = bytearray()
        while len(buffered) < PRECOMMIT_EVENT_LIMIT:
            chunk = await response.content.read(
                PRECOMMIT_EVENT_LIMIT - len(buffered)
            )
            if not chunk:
                body = bytes(buffered)
                return body, _error_message_from_bytes(body)
            buffered.extend(chunk)
        return bytes(buffered), None

    content_type = response.headers.get("Content-Type", "").lower()
    buffered = bytearray()
    while len(buffered) < PRECOMMIT_EVENT_LIMIT:
        chunk = await response.content.read(PRECOMMIT_EVENT_LIMIT - len(buffered))
        if not chunk:
            break
        buffered.extend(chunk)
        candidate = bytes(buffered)
        if "text/event-stream" in content_type:
            event = _first_sse_event(candidate)
            if event is None:
                continue
            return candidate, _error_message_from_bytes(event)
        if "json" in content_type:
            complete, message = _first_json_value(candidate)
            if not complete:
                continue
            return candidate, message
        return candidate, _error_message_from_bytes(candidate)
    candidate = bytes(buffered)
    if "text/event-stream" in content_type:
        event = _first_sse_event(candidate)
        return candidate, _error_message_from_bytes(event) if event else None
    return candidate, _error_message_from_bytes(candidate)


def _first_sse_event(body: bytes) -> bytes | None:
    normalized = body.replace(b"\r\n", b"\n")
    boundary = normalized.find(b"\n\n")
    if boundary < 0:
        return None
    data = []
    for line in normalized[:boundary].splitlines():
        if line.startswith(b"data:"):
            data.append(line[5:].lstrip())
    return b"\n".join(data)


def _first_json_value(body: bytes) -> tuple[bool, str | None]:
    try:
        text = body.decode("utf-8")
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, None
    return True, _error_message_from_value(value)


async def _upstream_error_message(response: aiohttp.ClientResponse) -> str | None:
    body = await response.content.read(UPSTREAM_ERROR_READ_LIMIT + 1)
    content_type = response.headers.get("Content-Type", "").lower()
    return _error_message_from_bytes(
        body[:UPSTREAM_ERROR_READ_LIMIT], text_fallback="json" not in content_type
    )


async def _bounded_upstream_error_message(
    response: aiohttp.ClientResponse,
) -> str | None:
    try:
        return await asyncio.wait_for(
            _upstream_error_message(response), timeout=FIRST_BYTE_TIMEOUT_S
        )
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


def _error_message_from_bytes(
    body: bytes | None, *, text_fallback: bool = False
) -> str | None:
    if not body:
        return None
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, RecursionError, ValueError):
        if not text_fallback:
            return None
        return _sanitize_error_message(body.decode("utf-8", errors="replace"))
    return _error_message_from_value(value)


def _error_message_from_value(value: object) -> str | None:
    if not isinstance(value, dict) or "error" not in value:
        return None
    error = value["error"]
    if not error:
        return None
    if isinstance(error, str):
        return _sanitize_error_message(error)
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return _sanitize_error_message(error["message"])
    return "upstream returned an error object"


def _sanitize_error_message(message: str) -> str:
    message = re.sub(
        r"\x1b(?:\][^\x07]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])",
        "",
        message,
    )
    message = re.sub(
        r"(?i)\b(authorization|api[-_ ]?key|token|password)\b"
        r"\s*[:=]?\s*(?:bearer\s+)?\S+",
        r"\1=[redacted]",
        message,
    )
    message = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [redacted]", message)
    message = re.sub(r"\b[A-Za-z0-9_-]{32,}\b", "[redacted]", message)
    message = " ".join(message.split())
    if len(message) > UPSTREAM_ERROR_LOG_LIMIT:
        return message[: UPSTREAM_ERROR_LOG_LIMIT - 3] + "..."
    return message


class RemoteFleet:
    """Configured remote backends plus the placement lookup."""

    def __init__(
        self,
        configs: Mapping[str, RemoteBackendConfig],
        placement: Mapping[str, tuple[str, ...]],
    ):
        active = {
            name for names in placement.values() for name in names if name != "axis"
        }
        self.backends = {
            name: RemoteBackend(name, config, active=name in active)
            for name, config in configs.items()
        }
        self.placement = dict(placement)
        for route, names in self.placement.items():
            _, _, model = route.partition(":")
            for name in names:
                if name in self.backends:
                    self.backends[name].ensure_model(model)

    async def start(self) -> None:
        for backend in self.backends.values():
            await backend.start()

    async def close(self) -> None:
        for backend in self.backends.values():
            await backend.close()

    def route(self, origin: str, model: str | None) -> tuple[str, ...]:
        if not model:
            return ("axis",)
        return self.placement.get(f"{origin}:{model}", ("axis",))

    def snapshot(self) -> dict:
        return {
            "backends": {
                name: backend.snapshot() for name, backend in self.backends.items()
            }
        }
