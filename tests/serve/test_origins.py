from __future__ import annotations

from laplace_serve import origins
from laplace_serve.config import LaplacedConfig

_CONFIG = LaplacedConfig.from_dict(
    {"origins": {"hermes": "interactive", "openclaw": "interactive", "default": "scheduled"}}
)


def test_path_prefix_tags_origin_and_tier():
    assert origins.resolve("hermes", {}, _CONFIG) == ("hermes", "interactive")
    assert origins.resolve("openclaw", {}, _CONFIG) == ("openclaw", "interactive")


def test_bare_v1_maps_to_default_origin():
    assert origins.resolve(None, {}, _CONFIG) == ("default", "scheduled")


def test_unknown_origin_falls_back_to_default_tier():
    assert origins.resolve("mystery", {}, _CONFIG) == ("mystery", "scheduled")


def test_origin_header_override_reresolves_tier():
    # Override origin from a bare request; tier follows the overridden origin.
    assert origins.resolve(None, {"x-laplace-origin": "hermes"}, _CONFIG) == (
        "hermes",
        "interactive",
    )


def test_priority_header_override_wins():
    assert origins.resolve("hermes", {"x-laplace-priority": "scheduled"}, _CONFIG) == (
        "hermes",
        "scheduled",
    )


def test_invalid_priority_header_ignored():
    assert origins.resolve("hermes", {"x-laplace-priority": "asap"}, _CONFIG) == (
        "hermes",
        "interactive",
    )
