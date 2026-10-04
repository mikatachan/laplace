"""LM Studio adapter for the standalone Laplace core."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import time
from dataclasses import dataclass

from laplace.adapter import LoadedModel, ModelLoadError

log = logging.getLogger(__name__)

_MIB = 1024 * 1024
_DEFAULT_LMS_CLI = "~/.lmstudio/bin/lms"
_FOOTPRINT_OVERHEAD = 1.3


@dataclass
class _UncertainLoad:
    previous: set[str]
    started_at: float
    catalog_keys: set[str]


class LMStudioAdapter:
    """InferenceAdapter backed by the `lms` CLI."""

    def __init__(
        self,
        lms_cli: str | None = None,
        unload_verify_delay_s: float = 2.0,
        load_timeout_s: float = 180.0,
        footprint_overrides: dict[str, int] | None = None,
        load_grace_s: float = 90.0,
    ):
        self._explicit_cli = lms_cli
        self._lms_cli = self._expand_home(lms_cli or _DEFAULT_LMS_CLI)
        self._unload_verify_delay_s = unload_verify_delay_s
        self._load_timeout_s = load_timeout_s
        self._load_grace_s = load_grace_s
        self._footprint_overrides = self._normalize_overrides(footprint_overrides)
        self._footprint_cache: dict[str, int | None] = {}
        # Serialize mutations; resident admissions never wait for this lock.
        self._load_lock = asyncio.Lock()
        self._active_load: str | None = None
        self._load_failures: dict[str, str] = {}
        # Preserve the original deadline across retries and inventory failures.
        self._load_uncertain: dict[str, _UncertainLoad] = {}

    def _normalize_overrides(self, overrides: dict[str, int] | None) -> dict[str, int]:
        if not overrides:
            return {}
        return {self.base_id(key): int(value) for key, value in overrides.items()}

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
        override = self._footprint_overrides.get(self.base_id(model_id))
        if override is not None:
            return override
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
        base = self.base_id(model_id)
        before = await self._resident_entries()
        if self._admit_resident(before, model_id, context_length):
            # Do not clear state belonging to a command still in progress.
            if self._active_load != base:
                self._clear_load_state(base)
            return
        async with self._load_lock:
            # Another admission may have loaded it while we waited.
            before = await self._resident_entries()
            if self._admit_resident(before, model_id, context_length):
                self._clear_load_state(base)
                return
            failure = self._load_failures.get(base)
            if failure:
                raise ModelLoadError(failure)
            self._reconcile_uncertain(before)
            keys = await self._catalog_keys()
            if model_id not in keys:
                closest = difflib.get_close_matches(model_id, sorted(keys), n=1, cutoff=0)
                message = (
                    f"lmstudio refusing fuzzy load: requested {model_id} has no exact catalog key; "
                    f"closest key: {closest[0] if closest else '<empty catalog>'}"
                )
                log.error("%s", message)
                raise ModelLoadError(message)

            previous = {self._entry_identifier(entry) for entry in before}
            self._load_uncertain[base] = _UncertainLoad(previous, time.monotonic(), keys)
            self._active_load = base
            try:
                rc, out, err = await self._run(
                    ["load", model_id, "--context-length", str(context_length), "-y"],
                    timeout=self._load_timeout_s,
                )
                if rc != 0:
                    raise ModelLoadError(
                        f"lmstudio load failed for {model_id} at ctx={context_length} "
                        f"(rc={rc}): {(err or out).strip()[:200]}"
                    )
                await self._verify_load(model_id, context_length, before, keys)
            except TimeoutError as exc:
                message = (
                    f"lms load timed out after {self._load_timeout_s:g}s "
                    f"for {model_id} at ctx={context_length}"
                )
                log.error("%s", message)
                raise ModelLoadError(message) from exc
            except Exception as exc:
                log.error("%s", exc)
                raise ModelLoadError(str(exc)) from exc
            finally:
                self._active_load = None
            self._clear_load_state(base)

    def _reconcile_uncertain(self, entries: list[dict]) -> None:
        """Use the fresh locked snapshot before permitting any cold load."""
        for base, pending in list(self._load_uncertain.items()):
            if any(self._matches(entry, base) for entry in entries):
                # Visible instances are accounted for by the broker's inventory.
                self._clear_load_state(base)
                continue
            deadline = pending.started_at + self._load_timeout_s + self._load_grace_s
            if time.monotonic() < deadline:
                raise ModelLoadError(
                    f"lmstudio load outcome unresolved for requested {base}; "
                    "cold loads blocked during timeout plus grace window"
                )
            # A known other catalog model is not evidence of a fuzzy result.
            # Unknown new identities cannot safely be attributed or ignored.
            possible = [
                entry for entry in entries
                if self._entry_identifier(entry) not in pending.previous
                and not any(self._matches(entry, key) for key in pending.catalog_keys - {base})
            ]
            if possible:
                raise ModelLoadError(
                    f"lmstudio load outcome unresolved for requested {base}; "
                    f"new unidentified instances={[self._entry_identifier(e) for e in possible]}; "
                    "operator reconciliation required"
                )
            log.warning(
                "lmstudio load outcome failed for %s after timeout plus grace; "
                "fresh inventory has no new matching instance; allowing retry", base,
            )
            self._clear_load_state(base)

    def _clear_load_state(self, base: str) -> None:
        self._load_uncertain.pop(base, None)
        self._load_failures.pop(base, None)

    def _admit_resident(self, entries: list[dict], model_id: str, context_length: int) -> bool:
        matches = [entry for entry in entries if self._matches(entry, model_id)]
        if not matches:
            return False
        if len(matches) == 1 and (self._entry_context_length(matches[0]) or 0) >= context_length:
            return True
        raise ModelLoadError(
            f"lmstudio refusing duplicate load for {model_id}: already resident as "
            f"{[self._entry_identifier(entry) for entry in matches]} with "
            f"insufficient/unknown context or duplicate instances (wanted ctx={context_length})"
        )

    async def _resident_entries(self) -> list[dict]:
        """Inventory used for mutations must fail closed, never mean 'empty'."""
        try:
            rc, out, err = await self._run(["ps", "--json"], timeout=15.0)
            if rc != 0:
                raise ValueError(f"rc={rc}: {(err or out).strip()[:200]}")
            entries = json.loads(out)
            if not isinstance(entries, list) or any(
                not isinstance(entry, dict) or not self._entry_identifier(entry)
                for entry in entries
            ):
                raise ValueError("expected an array of identified model instances")
            identifiers = [self._entry_identifier(entry) for entry in entries]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("ambiguous duplicate instance identifiers")
            return entries
        except Exception as exc:
            raise ModelLoadError(f"lmstudio cannot establish residency from lms ps: {exc}") from exc

    def _matches(self, entry: dict, model_id: str) -> bool:
        return any(
            isinstance(entry.get(key), str)
            and self.base_id(entry[key]) == self.base_id(model_id)
            for key in ("identifier", "modelKey", "path")
        )

    async def _verify_load(
        self, model_id: str, context_length: int, before: list[dict], keys: set[str]
    ) -> None:
        previous = {self._entry_identifier(entry) for entry in before}
        deadline = time.monotonic() + min(10.0, self._load_timeout_s)
        while True:
            entries = await self._resident_entries()
            created = [entry for entry in entries if self._entry_identifier(entry) not in previous]
            exact = [entry for entry in created if self._entry_identifier(entry) == model_id]
            if exact and self._admit_resident(entries, model_id, context_length):
                return
            # Known catalog identities may be another client's concurrent load.
            # Even an unknown identity is only a candidate, never proof of ownership.
            candidates = [
                entry for entry in created
                if not any(self._matches(entry, key) for key in keys - {model_id})
            ]
            fuzzy = [entry for entry in candidates if self._entry_identifier(entry) != model_id]
            if fuzzy:
                message = (
                    f"lmstudio requested {model_id}, observed possible fuzzy load "
                    f"{[self._entry_identifier(entry) for entry in fuzzy]}; "
                    "ownership unknown; no cleanup; automatic retry disabled until "
                    "requested model is verified resident"
                )
                self._load_failures[self.base_id(model_id)] = message
                self._load_uncertain.pop(self.base_id(model_id), None)
                raise ModelLoadError(message)
            if time.monotonic() >= deadline:
                raise ModelLoadError(
                    f"lmstudio load for {model_id} did not become resident at "
                    f"ctx={context_length} after a successful load command; "
                    f"new identifiers={[self._entry_identifier(entry) for entry in created]}; "
                    "cannot safely attribute cleanup"
                )
            await asyncio.sleep(0.1)

    async def validate_model_ids(self, model_ids: set[str]) -> None:
        """Report stale configured keys without preventing daemon startup."""
        if not model_ids:
            return
        try:
            keys = await self._catalog_keys()
        except ModelLoadError as exc:
            log.error("lmstudio startup model validation unavailable: %s", exc)
            return
        for model_id in sorted(model_ids - keys):
            closest = difflib.get_close_matches(model_id, sorted(keys), n=1, cutoff=0)
            log.error(
                "lmstudio configured model id %s has no exact catalog match; closest key: %s",
                model_id, closest[0] if closest else "<empty catalog>",
            )

    async def _catalog_keys(self) -> set[str]:
        try:
            rc, out, err = await self._run(["ls", "--json"], timeout=30.0)
            if rc != 0:
                raise ValueError(f"rc={rc}: {(err or out).strip()[:200]}")
            entries = json.loads(out)
            if not isinstance(entries, list) or any(
                not isinstance(entry, dict) or not isinstance(entry.get("modelKey"), str)
                for entry in entries
            ):
                raise ValueError("expected a catalog array with exact modelKey values")
            return {entry["modelKey"] for entry in entries}
        except Exception as exc:
            raise ModelLoadError(f"lmstudio cannot validate catalog keys: {exc}") from exc

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
            try:
                entries = await self._resident_entries()
            except ModelLoadError as exc:
                log.error("lmstudio cannot verify unload of %s: %s", model_id, exc)
                return False
            still_loaded = any(
                self._entry_identifier(entry) == model_id
                or (model_id == base and self._matches(entry, base))
                for entry in entries
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
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except (TimeoutError, asyncio.CancelledError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass  # The child may have exited between timeout and kill.
                await proc.wait()
                raise
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
