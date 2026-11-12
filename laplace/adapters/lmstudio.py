"""LM Studio adapter for the standalone Laplace core."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import math
import os
import time
from dataclasses import dataclass

from laplace.adapter import LoadCost, LoadedModel, ModelLoadError

log = logging.getLogger(__name__)

_MIB = 1024 * 1024
_DEFAULT_LMS_CLI = "~/.lmstudio/bin/lms"
_FOOTPRINT_OVERHEAD = 1.3
_DEFAULT_MODELS_DIR = "~/.lmstudio/models"
# fp16/bf16 KV entries. A quantized KV cache is smaller, so this over-reserves.
_KV_DTYPE_BYTES = 2
# Used only when no architecture metadata is readable. 256 KiB/token/sequence
# exceeds the per-token KV of every model in the local MLX catalog (largest:
# gemma-4-26B at 220 KiB inside its sliding window, GLM-4.5-Air at 184 KiB),
# so it errs toward refusing a load.
_FALLBACK_KV_BYTES_PER_TOKEN = 256 * 1024
# Parallel used for sizing when [model_parallel] does not pin it: LM Studio
# then applies its own default, which laplace cannot read.
_FALLBACK_PARALLEL = 4
# Runtime buffers beyond weights and KV, as a fraction of weights.
_RUNTIME_OVERHEAD = 0.10


@dataclass
class _UncertainLoad:
    previous: set[str]
    started_at: float
    catalog_keys: set[str]
    reason: str
    window_s: float
    wedge_logged: bool = False


class LMStudioAdapter:
    """InferenceAdapter backed by the `lms` CLI."""

    def __init__(
        self,
        lms_cli: str | None = None,
        unload_verify_delay_s: float = 2.0,
        load_timeout_s: float = 180.0,
        footprint_overrides: dict[str, int] | None = None,
        load_grace_s: float = 90.0,
        parallel_overrides: dict[str, int] | None = None,
        models_dir: str | None = None,
    ):
        self._explicit_cli = lms_cli
        self._lms_cli = self._expand_home(lms_cli or _DEFAULT_LMS_CLI)
        self._unload_verify_delay_s = unload_verify_delay_s
        self._load_timeout_s = load_timeout_s
        self._load_grace_s = load_grace_s
        self._footprint_overrides = self._normalize_overrides(footprint_overrides)
        self._parallel_overrides = self._normalize_overrides(parallel_overrides)
        self._footprint_cache: dict[str, int | None] = {}
        self._models_dir = self._expand_home(models_dir or _DEFAULT_MODELS_DIR)
        self._catalog_entries: dict[str, dict] = {}
        self._kv_specs_cache: dict[str, list[tuple[int, int | None]] | None] = {}
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

    async def load_cost(self, model_id: str, context_length: int | None) -> LoadCost | None:
        """Peak memory for ``model_id`` loaded at ``context_length`` x parallel.

        weights: catalog ``sizeBytes``, else the configured footprint table.
        kv: per-layer K+V bytes from the model's config.json, times context
        (capped at the window for sliding layers) and parallel.
        Returns None when no weight figure is known at all.
        """
        base = self.base_id(model_id)
        estimate: list[str] = []
        entry = await self._catalog_entry(model_id)
        size = self._entry_size_bytes(entry) if entry else None
        if size:
            weights_mb = math.ceil(size / _MIB)
        else:
            override = self._footprint_overrides.get(base)
            if override is None:
                return None
            weights_mb = override
            estimate.append("weights=footprint-table")
        parallel = self._parallel_overrides.get(base)
        if parallel is None:
            parallel = _FALLBACK_PARALLEL
            estimate.append(f"parallel=unconfigured({parallel})")
        if not context_length or context_length <= 0:
            kv_mb = 0
            estimate.append("context=unknown")
        else:
            specs = await self._kv_specs(base, entry)
            if specs is None:
                per_sequence = _FALLBACK_KV_BYTES_PER_TOKEN * context_length
                estimate.append("kv=no-architecture")
            else:
                per_sequence = kv_cache_bytes(specs, context_length)
            kv_mb = math.ceil(per_sequence * parallel / _MIB)
        return LoadCost(
            weights_mb=weights_mb,
            kv_mb=kv_mb,
            overhead_mb=math.ceil(weights_mb * _RUNTIME_OVERHEAD),
            context_length=context_length,
            parallel=parallel,
            estimate=tuple(estimate),
        )

    async def _catalog_entry(self, model_id: str) -> dict | None:
        base = self.base_id(model_id)
        if base in self._catalog_entries:
            return self._catalog_entries[base]
        try:
            rc, out, err = await self._run(["ls", "--json"], timeout=30.0)
            entries = json.loads(out or "[]") if rc == 0 else []
        except Exception as exc:  # noqa: BLE001
            log.warning("lmstudio: catalog lookup failed for %s: %s", model_id, exc)
            return None
        if not isinstance(entries, list):
            return None
        for entry in entries:
            if isinstance(entry, dict) and self._matches(entry, base):
                self._catalog_entries[base] = entry
                return entry
        return None

    async def _variant_dir(self, entry: dict) -> str | None:
        """On-disk subpath of a hub model's selected variant.

        ``lms ls --json`` reports hub models by hub id only; ``lms ls <path>
        --json`` lists each variant's ``indexedModelIdentifier`` as
        ``<hub id>@<publisher>/<directory under the models dir>``.
        """
        path, selected = entry.get("path"), entry.get("selectedVariant")
        if not isinstance(path, str) or not entry.get("variants"):
            return None
        try:
            rc, out, _err = await self._run(["ls", path, "--json"], timeout=30.0)
            variants = json.loads(out or "[]") if rc == 0 else []
        except Exception as exc:  # noqa: BLE001
            log.warning("lmstudio: variant lookup failed for %s: %s", path, exc)
            return None
        variants = [v for v in variants if isinstance(v, dict)] if isinstance(variants, list) else []
        chosen = [v for v in variants if v.get("modelKey") == selected] or (
            variants if len(variants) == 1 else []
        )
        identity = chosen[0].get("indexedModelIdentifier") if chosen else None
        if isinstance(identity, str) and "@" in identity:
            return identity.split("@", 1)[1] or None
        return None

    async def _kv_specs(self, base: str, entry: dict | None) -> list[tuple[int, int | None]] | None:
        if base in self._kv_specs_cache:
            return self._kv_specs_cache[base]
        if not entry:
            return None
        # Hub models report a ``path`` that is not the on-disk directory. Try the
        # selected variant's real directory first, then each identity LM Studio
        # reports, and use the first with a config.
        candidates: list[tuple[str, str]] = []
        variant_dir = await self._variant_dir(entry)
        if variant_dir:
            candidates.append(("variant", variant_dir))
        for key in ("indexedModelIdentifier", "path"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                candidates.append((key, value))
        specs = None
        for source, candidate in candidates:
            root = (
                candidate if os.path.isabs(candidate)
                else os.path.join(self._models_dir, candidate)
            )
            if os.path.isfile(root):
                root = os.path.dirname(root)
            try:
                with open(os.path.join(root, "config.json"), encoding="utf-8") as handle:
                    specs = kv_layer_specs(json.load(handle))
                log.info("lmstudio: architecture metadata for %s from %s: %s", base, source, root)
                break
            except (OSError, ValueError) as exc:
                log.info("lmstudio: no architecture metadata for %s via %s at %s: %s",
                         base, source, root, exc)
        self._kv_specs_cache[base] = specs
        return specs

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
            keys = await self._catalog_keys()
            self._reconcile_uncertain(base, before, keys)
            if model_id not in keys:
                closest = difflib.get_close_matches(model_id, sorted(keys), n=1, cutoff=0)
                message = (
                    f"lmstudio refusing fuzzy load: requested {model_id} has no exact catalog key; "
                    f"closest key: {closest[0] if closest else '<empty catalog>'}"
                )
                log.error("%s", message)
                raise ModelLoadError(message)

            previous = {self._entry_identifier(entry) for entry in before}
            self._load_uncertain[base] = _UncertainLoad(
                previous=previous,
                started_at=time.monotonic(),
                catalog_keys=keys,
                reason="unknown",
                window_s=self._load_timeout_s + self._load_grace_s,
            )
            self._active_load = base
            try:
                load_args = ["load", model_id, "--context-length", str(context_length)]
                parallel = self._parallel_overrides.get(base)
                if parallel is not None:
                    load_args.extend(["--parallel", str(parallel)])
                load_args.append("-y")
                log.info("lmstudio: exec lms %s", " ".join(load_args))
                rc, out, err = await self._run(load_args, timeout=self._load_timeout_s)
                if rc != 0:
                    pending = self._load_uncertain[base]
                    pending.started_at = time.monotonic()
                    pending.reason = "rc"
                    pending.window_s = self._load_grace_s
                    self._log_uncertain_window(base, pending)
                    raise ModelLoadError(
                        f"lmstudio load failed for {model_id} at ctx={context_length} "
                        f"(rc={rc}): {(err or out).strip()[:200]}"
                    )
                await self._verify_load(model_id, context_length, before, keys)
            except TimeoutError as exc:
                pending = self._load_uncertain[base]
                pending.reason = "timeout"
                self._log_uncertain_window(base, pending)
                message = (
                    f"lms load timed out after {self._load_timeout_s:g}s "
                    f"for {model_id} at ctx={context_length}"
                )
                log.error("%s", message)
                raise ModelLoadError(message) from exc
            except Exception as exc:
                # Handle the case where we have an unknown reason and need to log the window
                pending = self._load_uncertain.get(base)
                if pending and pending.reason == "unknown":
                    pending.reason = "verify"
                    self._log_uncertain_window(base, pending)
                log.error("%s", exc)
                raise ModelLoadError(str(exc)) from exc
            finally:
                self._active_load = None
            self._clear_load_state(base)

    def _log_uncertain_window(self, base: str, pending: _UncertainLoad) -> None:
        log.error(
            "lmstudio uncertain load window opened model=%s reason=%s window=%gs",
            base,
            pending.reason,
            pending.window_s,
        )

    def guard_window_remaining_s(self) -> float | None:
        """Return the longest active uncertain-load guard duration."""
        now = time.monotonic()
        remaining = [
            pending.started_at + pending.window_s - now
            for pending in self._load_uncertain.values()
        ]
        active = [duration for duration in remaining if duration > 0]
        return max(active, default=None)

    def _reconcile_uncertain(
        self, requested_base: str, entries: list[dict], catalog_keys: set[str]
    ) -> None:
        """Reconcile visible loads, globally guarding only client timeouts."""
        for base, pending in list(self._load_uncertain.items()):
            if any(self._matches(entry, base) for entry in entries):
                # Visible instances are accounted for by the broker's inventory.
                self._clear_load_state(base)
                continue
            if base != requested_base and pending.reason != "timeout":
                continue
            deadline = pending.started_at + pending.window_s
            if time.monotonic() < deadline:
                raise ModelLoadError(
                    f"lmstudio load outcome unresolved for requested {base}; "
                    f"cold loads blocked during {pending.reason} window "
                    f"({pending.window_s:g}s)"
                )
            # A known other catalog model is not evidence of a fuzzy result.
            # Unknown new identities cannot safely be attributed or ignored.
            known_keys = catalog_keys | pending.catalog_keys
            possible = [
                entry for entry in entries
                if self._entry_identifier(entry) not in pending.previous
                and not any(
                    self._matches(entry, key)
                    for key in known_keys - {base}
                )
            ]
            if possible:
                message = (
                    f"lmstudio load outcome unresolved for requested {base}; "
                    f"new unidentified instances={[self._entry_identifier(e) for e in possible]}; "
                    "operator reconciliation required"
                )
                if not pending.wedge_logged:
                    log.error("%s", message)
                    pending.wedge_logged = True
                raise ModelLoadError(message)
            log.error(
                "lmstudio uncertain load window expired model=%s reason=%s window=%gs; "
                "fresh inventory has no new matching instance; allowing retry",
                base,
                pending.reason,
                pending.window_s,
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

    async def validate_model_ids(self, model_ids: set[str]) -> bool:
        """Report stale configured keys without preventing daemon startup.

        Returns True when the catalog was reachable (validation ran, including
        the no-op for an empty id set) and False when it was unavailable.
        """
        if not model_ids:
            return True
        try:
            keys = await self._catalog_keys()
        except ModelLoadError as exc:
            log.error("lmstudio startup model validation unavailable: %s", exc)
            return False
        for model_id in sorted(model_ids - keys):
            closest = difflib.get_close_matches(model_id, sorted(keys), n=1, cutoff=0)
            log.error(
                "lmstudio configured model id %s has no exact catalog match; closest key: %s",
                model_id, closest[0] if closest else "<empty catalog>",
            )
        return True

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

    async def force_unload(self, model_id: str, *, reason: str = "direct") -> bool:
        if not model_id:
            return False
        log.info("lmstudio: unload requested model=%s reason=%s", model_id, reason)
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
            # A same-base sibling (identifier "<base>:N") is never the victim:
            # matching it here would retry the unload against a kept instance.
            still_loaded = any(
                self._entry_identifier(entry) == model_id
                or (
                    model_id == base
                    and self._matches(entry, base)
                    and self.base_id(self._entry_identifier(entry)) != base
                )
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


def kv_layer_specs(config: dict) -> list[tuple[int, int | None]] | None:
    """Per attention layer: (K+V bytes per token, sliding window or None).

    Reads Hugging Face / MLX config.json fields. Linear-attention layers keep a
    fixed-size state, not a per-token cache, and are omitted. Returns None for
    layouts this does not model (e.g. MLA) so callers fall back conservatively.
    """
    cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    try:
        layers = int(cfg["num_hidden_layers"])
        heads = int(cfg["num_attention_heads"])
        kv_heads = int(cfg.get("num_key_value_heads") or heads)
        head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // heads)
        global_kv_heads = int(cfg.get("num_global_key_value_heads") or kv_heads)
        global_head_dim = int(cfg.get("global_head_dim") or head_dim)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if cfg.get("kv_lora_rank") or layers <= 0:
        return None
    window = cfg.get("sliding_window")
    window = window if isinstance(window, int) and window > 0 else None
    types = cfg.get("layer_types")
    if not (isinstance(types, list) and len(types) == layers):
        interval = cfg.get("full_attention_interval")
        if isinstance(interval, int) and interval > 0:
            types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(layers)
            ]
        else:
            # Without per-layer types, size every layer as full attention.
            types = ["full_attention"] * layers
    specs: list[tuple[int, int | None]] = []
    for layer_type in types:
        if layer_type == "full_attention":
            specs.append((2 * global_kv_heads * global_head_dim * _KV_DTYPE_BYTES, None))
        elif layer_type == "sliding_attention":
            specs.append((2 * kv_heads * head_dim * _KV_DTYPE_BYTES, window))
        elif layer_type == "linear_attention":
            continue
        else:
            return None
    return specs


def kv_cache_bytes(specs: list[tuple[int, int | None]], context_length: int) -> int:
    """KV bytes one sequence holds at ``context_length`` tokens."""
    return sum(
        per_token * (min(context_length, window) if window else context_length)
        for per_token, window in specs
    )
