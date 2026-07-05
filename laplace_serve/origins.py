"""Map an incoming path prefix plus headers onto (origin, priority tier)."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from laplace.priority import from_legacy_string

from laplace_serve.config import DEFAULT_ORIGIN, LaplacedConfig

log = logging.getLogger(__name__)

_ORIGIN_HEADER = "x-laplace-origin"
_PRIORITY_HEADER = "x-laplace-priority"


def resolve(
    origin_from_path: str | None,
    headers: Mapping[str, str],
    config: LaplacedConfig,
) -> tuple[str, str]:
    """Resolve (origin, tier) for a request.

    The path prefix /{origin}/v1/... selects the origin; a bare /v1 request maps
    to the catch-all origin. ``x-laplace-origin`` overrides the origin (and thus
    its tier lookup); ``x-laplace-priority`` overrides the final tier directly.
    An invalid priority header is ignored with a warning, keeping the resolved
    tier.
    """
    origin = (origin_from_path or DEFAULT_ORIGIN).strip() or DEFAULT_ORIGIN

    header_origin = _clean(headers.get(_ORIGIN_HEADER))
    if header_origin:
        origin = header_origin

    tier = config.tier_for(origin)

    header_priority = _clean(headers.get(_PRIORITY_HEADER))
    if header_priority:
        try:
            from_legacy_string(header_priority)
            tier = header_priority.lower()
        except ValueError:
            log.warning(
                "origins: ignoring invalid %s=%r for origin %s",
                _PRIORITY_HEADER,
                header_priority,
                origin,
            )

    return origin, tier


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    trimmed = value.strip()
    return trimmed or None
