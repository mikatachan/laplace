"""LM Studio adapter for the standalone Laplace core."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from laplace.adapter import LoadedModel

log = logging.getLogger(__name__)

_MIB = 1024 * 1024
_DEFAULT_LMS_CLI = "~/.lmstudio/bin/lms"
_FOOTPRINT_OVERHEAD = 1.3


class LMStudioAdapter:
    """InferenceAdapter backed by the `lms` CLI."""

    def __init__(
        self,
        lms_cli: str | None = None,
        unload_verify_delay_s: float = 2.0,
    ):
        self._explicit_cli = lms_cli
        self._lms_cli = self._expand_home(lms_cli or _DEFAULT_LMS_CLI)
        self._unload_verify_delay_s = unload_verify_delay_s
        self._footprint_cache: dict[str, int | None] = {}

    def base_id(self, model_id: str) -> str:
        if not model_id:
            return model_id
        head, sep, tail = model_id.rpartition(":")
        return head if (sep and tail.isdigit()) else model_id

    async def list_loaded(self) -> list[LoadedModel]:
        try:
            rc, out, err = await self._run(["ps", "--json"], timeout=15.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("lmstudio: failed to list loaded models: %s", exc)
            return []
        if rc != 0:
            log.warning(
                "lmstudio: 'lms ps --json' failed (rc=%s): %s",
                rc,
                (err or out).strip()[:200],
            )
            return []
        try:
            entries = json.loads(out or "[]")
        except Exception as exc:  # noqa: BLE001
            log.warning("lmstudio: invalid JSON from 'lms ps --json': %s", exc)
            return []

        loaded: list[LoadedModel] = []
        for entry in entries:
            model_id = self._entry_identifier(entry)
            if not model_id:
                continue
            loaded.append(
                LoadedModel(
                    id=model_id,
                    base_id=self.base_id(model_id),
                    size_bytes=self._entry_size_bytes(entry),
                    status=self._entry_status(entry),
                    last_used_ms=self._entry_last_used_ms(entry),
                    queued=self._entry_queued(entry),
                    context_length=self._entry_context_length(entry),
                )
            )
        return loaded

    async def footprint_mb(self, model_id: str) -> int | None:
        if model_id in self._footprint_cache:
            return self._footprint_cache[model_id]

        for entry in await self.list_loaded():
            if self.base_id(entry.id) == self.base_id(model_id) and entry.size_bytes:
                mb = int(entry.size_bytes * _FOOTPRINT_OVERHEAD / _MIB)
                self._footprint_cache[model_id] = mb
                return mb

        try:
            rc, out, err = await self._run(["ls", "--json"], timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("lmstudio: footprint catalog lookup failed for %s: %s", model_id, exc)
            return None
        if rc == 0:
            try:
                entries = json.loads(out or "[]")
            except Exception:  # noqa: BLE001
                entries = []
            for entry in entries:
                size = entry.get("sizeBytes")
                if isinstance(size, bool) or not isinstance(size, (int, float)):
                    continue
                keys = [
                    entry.get("modelKey"),
                    entry.get("path"),
                    entry.get("identifier"),
                ]
                if any(
                    isinstance(key, str) and self.base_id(key) == self.base_id(model_id)
                    for key in keys
                ):
                    mb = int(size * _FOOTPRINT_OVERHEAD / _MIB)
                    self._footprint_cache[model_id] = mb
                    return mb
        else:
            log.warning(
                "lmstudio: 'lms ls --json' failed (rc=%s): %s",
                rc,
                (err or out).strip()[:200],
            )

        try:
            rc, out, err = await self._run([model_id, "--estimate-only"], timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("lmstudio: estimate-only failed for %s: %s", model_id, exc)
            return None
        if rc != 0:
            return None
        estimate = self._parse_estimate_mb(f"{out}\n{err}")
        self._footprint_cache[model_id] = estimate
        return estimate

    async def ensure_loaded(self, model_id: str, context_length: int | None) -> None:
        if not model_id or not context_length or context_length <= 0:
            return
        current = None
        for entry in await self.list_loaded():
            if self.base_id(entry.id) == self.base_id(model_id):
                current = entry.context_length
                if current is not None and current >= context_length:
                    return
                break
        try:
            rc, out, err = await self._run(
                ["load", model_id, "--context-length", str(context_length), "-y"],
                timeout=180.0,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("lmstudio: failed to load %s at ctx=%s: %s", model_id, context_length, exc)
            return
        if rc != 0:
            log.warning(
                "lmstudio: 'lms load %s --context-length %s -y' failed (rc=%s): %s",
                model_id,
                context_length,
                rc,
                (err or out).strip()[:200],
            )

    async def force_unload(self, model_id: str) -> bool:
        if not model_id:
            return False
        base = self.base_id(model_id)
        for attempt in range(2):
            try:
                rc, out, err = await self._run(["unload", model_id], timeout=30.0)
            except Exception as exc:  # noqa: BLE001
                log.warning("lmstudio: unload failed for %s: %s", model_id, exc)
                return False
            if rc != 0:
                message = (err or out).strip().lower()
                if "not loaded" not in message and "not found" not in message:
                    log.warning(
                        "lmstudio: 'lms unload %s' failed (rc=%s): %s",
                        model_id,
                        rc,
                        (err or out).strip()[:200],
                    )
            await asyncio.sleep(self._unload_verify_delay_s)
            still_loaded = any(
                entry.id == model_id
                or (model_id == base and entry.base_id == base)
                for entry in await self.list_loaded()
            )
            if not still_loaded:
                return True
            if attempt == 0:
                log.warning("lmstudio: %s still resident after unload; retrying once", model_id)
        log.warning("lmstudio: %s still resident after retry", model_id)
        return False

    @staticmethod
    def _expand_home(path: str) -> str:
        if path.startswith("~/"):
            home = os.environ.get("HOME")
            if home:
                return f"{home}/{path[2:]}"
        return path

    def _cli_candidates(self) -> list[str]:
        if self._explicit_cli:
            return [self._lms_cli]
        return [self._lms_cli, "lms"]

    async def _run(self, args: list[str], timeout: float) -> tuple[int, str, str]:
        last_error: Exception | None = None
        for cli in self._cli_candidates():
            try:
                proc = await asyncio.create_subprocess_exec(
                    cli,
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                last_error = exc
                continue
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode, (out or b"").decode(), (err or b"").decode()
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(self._lms_cli)

    @staticmethod
    def _entry_identifier(entry: dict) -> str:
        for key in ("identifier", "modelKey", "path"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    @staticmethod
    def _entry_size_bytes(entry: dict) -> int | None:
        for key in ("sizeBytes", "size_bytes", "loadedSizeBytes", "loaded_size_bytes"):
            value = entry.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    @staticmethod
    def _entry_status(entry: dict) -> str | None:
        value = entry.get("status")
        if isinstance(value, str) and value:
            return value.lower()
        return None

    @staticmethod
    def _entry_last_used_ms(entry: dict) -> float | None:
        for key in ("lastUsedTime", "last_used_time", "lastUsed"):
            value = entry.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
            if isinstance(value, str) and value.isdigit():
                return float(value)
        return None

    @staticmethod
    def _entry_queued(entry: dict) -> int | None:
        for key in ("queued", "queuedRequests", "queued_requests"):
            value = entry.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= 0:
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    @staticmethod
    def _entry_context_length(entry: dict) -> int | None:
        for key in ("contextLength", "context_length", "maxContextLength", "loadedContextLength"):
            value = entry.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    @staticmethod
    def _parse_estimate_mb(text: str) -> int | None:
        marker = "estimated total memory:"
        for line in text.splitlines():
            lower = line.lower()
            if marker not in lower:
                continue
            suffix = line[lower.index(marker) + len(marker):].strip()
            if not suffix:
                return None
            number = suffix.split()[0]
            try:
                return int(float(number) * 1024)
            except ValueError:
                return None
        return None
