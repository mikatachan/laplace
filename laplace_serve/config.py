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
    "default": "scheduled",
}

# The catch-all origin name used for bare /v1 traffic and for any origin whose
# path prefix has no explicit [origins] tier. Matches the ratified [origins]
# table key (the plan's D4 prose says "other"; the ratified config keys it
# "default", which is the source of truth).
DEFAULT_ORIGIN = "default"


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
    keep_loaded: tuple[str, ...] = ("nomic-embed-text",)
    default_context_length: int = 64000
    log_rotate_mb: int = 10
    log_backups: int = 5
    log_path: str = "~/.laplace/logs/laplaced.log"
    origins: Mapping[str, str] = field(default_factory=lambda: dict(_DEFAULT_ORIGINS))
    model_context: Mapping[str, int] = field(default_factory=dict)
    model_footprint_mb: Mapping[str, int] = field(default_factory=dict)

    def budget_mb_or_none(self) -> int | None:
        """Broker budget: None triggers auto-detection, a positive value pins it."""
        return self.budget_mb if self.budget_mb and self.budget_mb > 0 else None

    def resolved_log_path(self) -> str:
        return os.path.expanduser(self.log_path)

    def tier_for(self, origin: str) -> str:
        return self.origins.get(origin) or self.origins.get(DEFAULT_ORIGIN) or "scheduled"

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "LaplacedConfig":
        raw = dict(data or {})

        origins = dict(raw.pop("origins", None) or dict(_DEFAULT_ORIGINS))
        model_context = _coerce_int_map(raw.pop("model_context", None), "model_context")
        model_footprint_mb = _coerce_int_map(
            raw.pop("model_footprint_mb", None), "model_footprint_mb"
        )

        for origin, tier in origins.items():
            try:
                from_legacy_string(tier)
            except (ValueError, AttributeError, TypeError):
                raise ValueError(
                    f"laplaced config: origin {origin!r} has invalid tier {tier!r}; "
                    "expected one of interactive/scheduled/dreaming"
                ) from None
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


def missing_context_ids(roster_ids: Iterable[str], config: LaplacedConfig) -> list[str]:
    """Roster model ids that have no [model_context] entry.

    Used by the roster-completeness check: a live ``lms ls`` roster is compared
    against the configured operating-context map so gaps are visible before a
    consumer cutover picks the fallback context length by accident.
    """
    mapped = set(config.model_context)
    return sorted({mid for mid in roster_ids if mid and mid not in mapped})
