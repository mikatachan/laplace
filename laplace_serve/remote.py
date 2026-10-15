"""Remote backend credentials, health, and pre-commit routing."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Mapping

import aiohttp
from aiohttp import web

from laplace_serve.config import RemoteBackendConfig
from laplace_serve.proxy import filter_request_headers, relay_after_commit

PROBE_INTERVAL_S = 120.0
PROBE_TIMEOUT_S = 10.0
PRECOMMIT_TIMEOUT_S = 10.0
QUEUE_TIMEOUT_S = 10.0
FIRST_BYTE_TIMEOUT_S = 120.0
FALLBACK_HTTP_STATUSES = frozenset({400, 404, 429})


@dataclass(frozen=True)
class AttemptResult:
    response: web.StreamResponse | None = None
    failure_reason: str | None = None


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
            self.state = "DISABLED"
            self.reason = "no-credential"
            return False
        if not _is_header_safe_token(credential):
            self._credential = None
            self.state = "DISABLED"
            self.reason = "bad-credential"
            return False
        self._credential = credential
        if self.state == "DISABLED":
            self.state = "DOWN"
            self.reason = "not-probed"
        return True

    async def _probe_loop(self, *, delay_first: bool = False) -> None:
        if delay_first:
            await asyncio.sleep(PROBE_INTERVAL_S)
        while True:
            await self.probe_once()
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
                    timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT_S),
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
            except (aiohttp.ClientError, asyncio.TimeoutError):
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
            self.state = "UP"
            self.reason = None

    def record_probe_failure(self, reason: str = "probe-failed") -> None:
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        if self.state == "DISABLED":
            return
        if self.consecutive_failures >= 2:
            self.state = "DOWN"
            self.reason = reason

    def record_request_failure(self, reason: str) -> None:
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        if self.state == "DISABLED":
            return
        self.state = "DOWN"
        self.reason = reason

    async def forward(
        self,
        request: web.Request,
        upstream_path: str,
        body: bytes,
    ) -> AttemptResult:
        if self.state != "UP" or self._session is None or self._credential is None:
            return AttemptResult(failure_reason=f"state:{self.state.lower()}")

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
            return await self._forward_acquired(request, upstream_path, body)
        finally:
            self.in_flight -= 1
            self._slots.release()

    async def _forward_acquired(
        self,
        request: web.Request,
        upstream_path: str,
        body: bytes,
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
                timeout=PRECOMMIT_TIMEOUT_S,
            )
            if response.status == 401:
                response.close()
                await self._refresh_credential()
                self.record_request_failure("auth")
                return AttemptResult(failure_reason="auth")
            if response.status in FALLBACK_HTTP_STATUSES:
                status = response.status
                response.close()
                reason = f"upstream-{status}"
                self.record_request_failure(reason)
                return AttemptResult(failure_reason=reason)
            if 500 <= response.status <= 599:
                response.close()
                self.record_request_failure("upstream-5xx")
                return AttemptResult(failure_reason="upstream-5xx")
            try:
                first_chunk = await asyncio.wait_for(
                    response.content.readany(), timeout=FIRST_BYTE_TIMEOUT_S
                )
            except (aiohttp.ClientError, asyncio.TimeoutError):
                response.close()
                self.record_request_failure("first-byte")
                return AttemptResult(failure_reason="first-byte")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if response is not None:
                response.close()
            self.record_request_failure("connect")
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
        }

    def _ordered_urls(self) -> tuple[str, ...]:
        urls = (self.config.url,) + self.config.alt_urls
        if self.active_url is None or self.active_url not in urls:
            return urls
        return (self.active_url,) + tuple(url for url in urls if url != self.active_url)


def _is_header_safe_token(value: str) -> bool:
    """Accept non-empty visible ASCII without whitespace or control bytes."""
    return bool(value) and all(0x21 <= ord(char) <= 0x7E for char in value)


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
