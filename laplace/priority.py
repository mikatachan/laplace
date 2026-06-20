"""Server-derived request priority resolver for the standalone Laplace core."""

from __future__ import annotations

import contextvars
import logging
from enum import IntEnum

log = logging.getLogger(__name__)


class PriorityTier(IntEnum):
    """Request priority. Lower value = higher priority."""

    INTERACTIVE = 0
    SCHEDULED = 1
    DREAMING = 2


_TIER_TO_STRING: dict[PriorityTier, str] = {
    PriorityTier.INTERACTIVE: "interactive",
    PriorityTier.SCHEDULED: "scheduled",
    PriorityTier.DREAMING: "dreaming",
}
_STRING_TO_TIER: dict[str, PriorityTier] = {value: key for key, value in _TIER_TO_STRING.items()}
_WARNED_BAD_OVERRIDES: set[tuple[str, object]] = set()


def to_legacy_string(tier: PriorityTier) -> str:
    return _TIER_TO_STRING[tier]


def from_legacy_string(value: str) -> PriorityTier:
    try:
        return _STRING_TO_TIER[value.strip().lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown priority string {value!r}; expected one of {sorted(_STRING_TO_TIER)}"
        ) from None


def validate_priority_config(config: dict | None) -> list[str]:
    errors: list[str] = []
    if not isinstance(config, dict):
        return errors
    priority = config.get("priority")
    if priority is None:
        return errors
    if not isinstance(priority, dict):
        errors.append(
            f"config['priority'] must be a mapping, got {type(priority).__name__}"
        )
        return errors
    overrides = priority.get("agent_overrides")
    if overrides is None:
        return errors
    if not isinstance(overrides, dict):
        errors.append(
            "config['priority']['agent_overrides'] must be a mapping, got "
            f"{type(overrides).__name__}"
        )
        return errors
    for agent_id, raw in overrides.items():
        try:
            from_legacy_string(raw)
        except (ValueError, TypeError, AttributeError):
            errors.append(
                f"agent_overrides[{agent_id!r}]={raw!r} is not a valid priority tier; "
                f"expected one of {sorted(_STRING_TO_TIER)}"
            )
    return errors


def _override_for(agent_id: str | None, config: dict | None) -> PriorityTier | None:
    if not agent_id or not isinstance(config, dict):
        return None
    overrides = (config.get("priority") or {}).get("agent_overrides")
    if not isinstance(overrides, dict):
        return None
    raw = overrides.get(agent_id)
    if raw is None:
        return None
    try:
        return from_legacy_string(raw)
    except ValueError:
        if (agent_id, raw) not in _WARNED_BAD_OVERRIDES:
            _WARNED_BAD_OVERRIDES.add((agent_id, raw))
            log.warning(
                "priority: invalid agent_overrides[%r]=%r in config; ignoring override. "
                "Valid tiers: %s",
                agent_id,
                raw,
                sorted(_STRING_TO_TIER),
            )
        return None


def resolve_priority(
    *,
    from_cron: bool = False,
    is_heartbeat: bool = False,
    agent_id: str | None = None,
    config: dict | None = None,
) -> PriorityTier:
    override = _override_for(agent_id, config)
    if override is not None:
        return override
    if from_cron:
        return PriorityTier.DREAMING if is_heartbeat else PriorityTier.SCHEDULED
    return PriorityTier.INTERACTIVE


_REQUEST_PRIORITY: contextvars.ContextVar[PriorityTier | None] = contextvars.ContextVar(
    "request_priority",
    default=None,
)


def set_request_priority(tier: PriorityTier | None) -> contextvars.Token:
    return _REQUEST_PRIORITY.set(tier)


def reset_request_priority(token: contextvars.Token) -> None:
    _REQUEST_PRIORITY.reset(token)


def get_request_priority() -> PriorityTier | None:
    return _REQUEST_PRIORITY.get()


class _RequestPriorityBinding:
    def __init__(self, tier: PriorityTier | None):
        self._tier = tier
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "_RequestPriorityBinding":
        self._token = _REQUEST_PRIORITY.set(self._tier)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            _REQUEST_PRIORITY.reset(self._token)
        return False


def request_priority(tier: PriorityTier | None) -> _RequestPriorityBinding:
    return _RequestPriorityBinding(tier)


def current_priority_legacy(default: str = "interactive") -> str:
    tier = _REQUEST_PRIORITY.get()
    if tier is None:
        return default
    return to_legacy_string(tier)
