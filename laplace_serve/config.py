"""Daemon configuration: TOML load, dataclass model, startup validation."""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields

from laplace.priority import from_legacy_string

log = logging.getLogger(__name__)

_DEFAULT_ORIGINS: dict[str, str] = {
    "hermes": "interactive",
    "openclaw": "interactive",
    "bot": "interactive",
    "default": "scheduled",
}

# The catch-all origin name used for bare /v1 traffic and for any origin whose
# path prefix has no explicit [origins] tier. Matches the ratified [origins]
# table key (the plan's D4 prose says "other"; the ratified config keys it
# "default", which is the source of truth).
DEFAULT_ORIGIN = "default"


@dataclass(frozen=True)
class RemoteBackendConfig:
    """Configuration for an OpenAI-compatible unmanaged backend."""

    url: str
    alt_urls: tuple[str, ...] = ()
    slots: int = 1
    probe_model: str = ""
    auth_command: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaplacedConfig:
    """Validated daemon configuration.

    Every field mirrors a key in ~/.laplace/laplaced.toml. ``budget_mb`` of 0
    means auto-detect (sysctl wired limit, 90112 MB fallback in the broker).
    """

    bind: str = "127.0.0.1"
    port: int = 4242
    upstream: str = "http://127.0.0.1:1234"
    budget_mb: int = 0
    max_concurrent: int = 2
    admit_timeout_s: float = 120.0
    fail_open: bool = True
    reaper_sweep: bool = True
    sweep_interval_s: float = 60.0
    load_timeout_s: float = 300.0
    load_grace_s: float = 90.0
    keep_loaded: tuple[str, ...] = ("nomic-embed-text",)
    default_context_length: int = 64000
    log_rotate_mb: int = 10
    log_backups: int = 5
    log_path: str = "~/.laplace/logs/laplaced.log"
    origins: Mapping[str, str] = field(default_factory=lambda: dict(_DEFAULT_ORIGINS))
    model_context: Mapping[str, int] = field(default_factory=dict)
    model_footprint_mb: Mapping[str, int] = field(default_factory=dict)
    model_parallel: Mapping[str, int] = field(default_factory=dict)
    backends: Mapping[str, RemoteBackendConfig] = field(default_factory=dict)
    placement: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def budget_mb_or_none(self) -> int | None:
        """Broker budget: None triggers auto-detection, a positive value pins it."""
        return self.budget_mb if self.budget_mb and self.budget_mb > 0 else None

    def resolved_log_path(self) -> str:
        return os.path.expanduser(self.log_path)

    def tier_for(self, origin: str) -> str:
        return self.origins.get(origin) or self.origins.get(DEFAULT_ORIGIN) or "scheduled"

    def canonical_model_id(self, model_id: str | None) -> str | None:
        """Resolve an unambiguous bare request id to a configured LM Studio key.

        LM Studio reports catalog and resident identities as canonical keys such
        as ``qwen/qwen3-coder-30b`` while callers may send the bare trailing
        segment.  Do not guess when more than one configured key shares that
        segment: preserving the incoming id is safer than admitting a request
        against the wrong model.
        """
        if not model_id:
            return model_id
        configured = (
            set(self.model_context)
            | set(self.model_footprint_mb)
            | set(self.model_parallel)
        )
        if model_id in configured:
            return model_id
        matches = [key for key in configured if key.rsplit("/", 1)[-1] == model_id]
        if len(matches) == 1:
            return matches[0]
        if matches:
            log.warning(
                "laplaced config: ambiguous bare model id %r matches %s; using it unchanged",
                model_id,
                sorted(matches),
            )
        else:
            log.warning(
                "laplaced config: unconfigured model id %r; using default context and unchanged id",
                model_id,
            )
        return model_id

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "LaplacedConfig":
        raw = dict(data or {})

        origins = dict(raw.pop("origins", None) or dict(_DEFAULT_ORIGINS))
        model_context = _coerce_int_map(raw.pop("model_context", None), "model_context")
        model_footprint_mb = _coerce_int_map(
            raw.pop("model_footprint_mb", None), "model_footprint_mb"
        )
        model_parallel = _coerce_int_map(
            raw.pop("model_parallel", None), "model_parallel"
        )
        backends = _coerce_backends(raw.pop("backends", None))
        placement = _coerce_placement(raw.pop("placement", None), backends)
        for model_id, parallel in model_parallel.items():
            if parallel <= 0:
                raise ValueError(
                    f"laplaced config: [model_parallel] value for {model_id!r} "
                    f"must be positive, got {parallel!r}"
                )

        for origin, tier in origins.items():
            try:
                from_legacy_string(tier)
            except (ValueError, AttributeError, TypeError):
                raise ValueError(
                    f"laplaced config: origin {origin!r} has invalid tier {tier!r}; "
                    "expected one of interactive/scheduled/dreaming"
                ) from None
        origins.setdefault("bot", "interactive")
        origins.setdefault(DEFAULT_ORIGIN, "scheduled")

        known = {f.name for f in fields(cls)}
        unknown = [key for key in raw if key not in known]
        if unknown:
            log.warning("laplaced config: ignoring unknown keys %s", sorted(unknown))
        kwargs = {key: value for key, value in raw.items() if key in known}
        if "keep_loaded" in kwargs and kwargs["keep_loaded"] is not None:
            kwargs["keep_loaded"] = tuple(kwargs["keep_loaded"])

        return cls(
            origins=origins,
            model_context=model_context,
            model_footprint_mb=model_footprint_mb,
            model_parallel=model_parallel,
            backends=backends,
            placement=placement,
            **kwargs,
        )

    @classmethod
    def from_toml_file(cls, path: str) -> "LaplacedConfig":
        with open(os.path.expanduser(path), "rb") as handle:
            data = tomllib.load(handle)
        return cls.from_dict(data)


def _coerce_int_map(value: Mapping | None, label: str) -> dict[str, int]:
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"laplaced config: [{label}] must be a table")
    result: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(
                f"laplaced config: [{label}] value for {key!r} must be an integer, got {raw!r}"
            )
        result[str(key)] = raw
    return result


def _coerce_backends(value: Mapping | None) -> dict[str, RemoteBackendConfig]:
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("laplaced config: [backends] must be a table")
    result: dict[str, RemoteBackendConfig] = {}
    for raw_name, raw_backend in value.items():
        name = str(raw_name)
        if name == "axis":
            raise ValueError("laplaced config: backend name 'axis' is reserved")
        if not isinstance(raw_backend, Mapping):
            raise ValueError(f"laplaced config: [backends.{name}] must be a table")
        unknown = set(raw_backend) - {
            "url", "alt_urls", "slots", "probe_model", "auth_command"
        }
        if unknown:
            raise ValueError(
                f"laplaced config: [backends.{name}] has unknown keys {sorted(unknown)}"
            )
        url = raw_backend.get("url")
        probe_model = raw_backend.get("probe_model")
        alt_urls = raw_backend.get("alt_urls", ())
        slots = raw_backend.get("slots", 1)
        auth_command = raw_backend.get("auth_command", ())
        if not isinstance(url, str) or not url:
            raise ValueError(f"laplaced config: [backends.{name}].url must be a string")
        if not isinstance(probe_model, str) or not probe_model:
            raise ValueError(
                f"laplaced config: [backends.{name}].probe_model must be a string"
            )
        if (
            not isinstance(alt_urls, (list, tuple))
            or not all(isinstance(item, str) and item for item in alt_urls)
        ):
            raise ValueError(
                f"laplaced config: [backends.{name}].alt_urls must be a string list"
            )
        if isinstance(slots, bool) or not isinstance(slots, int) or slots <= 0:
            raise ValueError(
                f"laplaced config: [backends.{name}].slots must be a positive integer"
            )
        if (
            not isinstance(auth_command, (list, tuple))
            or not auth_command
            or not all(isinstance(item, str) and item for item in auth_command)
        ):
            # Never include the command or its arguments in an exception.
            raise ValueError(
                f"laplaced config: [backends.{name}].auth_command must be a non-empty argv list"
            )
        result[name] = RemoteBackendConfig(
            url=url.rstrip("/"),
            alt_urls=tuple(item.rstrip("/") for item in alt_urls),
            slots=slots,
            probe_model=probe_model,
            auth_command=tuple(auth_command),
        )
    return result


def _coerce_placement(
    value: Mapping | None,
    backends: Mapping[str, RemoteBackendConfig],
) -> dict[str, tuple[str, ...]]:
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("laplaced config: [placement] must be a table")
    result: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_names in value.items():
        key = str(raw_key)
        if ":" not in key:
            raise ValueError(
                f"laplaced config: placement key {key!r} must be '<origin>:<model>'"
            )
        if (
            not isinstance(raw_names, (list, tuple))
            or not raw_names
            or not all(isinstance(name, str) and name for name in raw_names)
        ):
            raise ValueError(
                f"laplaced config: placement {key!r} must be a non-empty backend list"
            )
        missing = [name for name in raw_names if name != "axis" and name not in backends]
        if missing:
            raise ValueError(
                f"laplaced config: placement {key!r} names unknown backends {missing}"
            )
        result[key] = tuple(raw_names)
    return result


def missing_context_ids(roster_ids: Iterable[str], config: LaplacedConfig) -> list[str]:
    """Roster model ids that have no [model_context] entry.

    Used by the roster-completeness check: a live ``lms ls`` roster is compared
    against the configured operating-context map so gaps are visible before a
    consumer cutover picks the fallback context length by accident.
    """
    mapped = set(config.model_context)
    return sorted({mid for mid in roster_ids if mid and mid not in mapped})
