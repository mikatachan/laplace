from __future__ import annotations

import pytest

from laplace.priority import (
    PriorityTier,
    current_priority_legacy,
    from_legacy_string,
    get_request_priority,
    request_priority,
    reset_request_priority,
    resolve_priority,
    set_request_priority,
    to_legacy_string,
    validate_priority_config,
)


def test_cron_heartbeat_is_dreaming():
    assert resolve_priority(from_cron=True, is_heartbeat=True) == PriorityTier.DREAMING


def test_cron_non_heartbeat_is_scheduled():
    assert resolve_priority(from_cron=True) == PriorityTier.SCHEDULED
    assert resolve_priority(from_cron=True, is_heartbeat=False) == PriorityTier.SCHEDULED


def test_default_is_interactive():
    assert resolve_priority() == PriorityTier.INTERACTIVE
    assert resolve_priority(is_heartbeat=True) == PriorityTier.INTERACTIVE


def test_config_override_takes_precedence():
    config = {"priority": {"agent_overrides": {"soma": "scheduled"}}}
    assert resolve_priority(agent_id="soma", config=config) == PriorityTier.SCHEDULED
    assert (
        resolve_priority(
            agent_id="soma",
            config=config,
            from_cron=True,
            is_heartbeat=True,
        )
        == PriorityTier.SCHEDULED
    )


def test_legacy_string_round_trip():
    for tier in PriorityTier:
        assert from_legacy_string(to_legacy_string(tier)) == tier


def test_from_legacy_string_rejects_unknown():
    with pytest.raises(ValueError):
        from_legacy_string("urgent")


def test_validate_priority_config_reports_errors():
    config = {"priority": {"agent_overrides": {"soma": "urgent"}}}
    errors = validate_priority_config(config)
    assert len(errors) == 1
    assert "soma" in errors[0]


def test_validate_priority_config_structural_errors():
    assert validate_priority_config({"priority": "interactive"})
    assert validate_priority_config({"priority": {"agent_overrides": ["soma"]}})


def test_resolve_priority_bad_override_falls_back():
    config = {"priority": {"agent_overrides": {"soma": "urgent"}}}
    assert resolve_priority(agent_id="soma", config=config) == PriorityTier.INTERACTIVE
    assert (
        resolve_priority(
            agent_id="soma",
            config=config,
            from_cron=True,
            is_heartbeat=True,
        )
        == PriorityTier.DREAMING
    )


def test_request_priority_unset_is_none_and_legacy_default():
    assert get_request_priority() is None
    assert current_priority_legacy() == "interactive"
    assert current_priority_legacy("scheduled") == "scheduled"


def test_set_and_reset_request_priority_round_trip():
    token = set_request_priority(PriorityTier.SCHEDULED)
    try:
        assert get_request_priority() == PriorityTier.SCHEDULED
        assert current_priority_legacy() == "scheduled"
    finally:
        reset_request_priority(token)
    assert get_request_priority() is None


def test_request_priority_context_manager_sets_and_resets():
    with request_priority(PriorityTier.DREAMING):
        assert get_request_priority() == PriorityTier.DREAMING
        assert current_priority_legacy() == "dreaming"
    assert get_request_priority() is None


def test_request_priority_context_manager_resets_on_exception():
    with pytest.raises(RuntimeError, match="boom"):
        with request_priority(PriorityTier.SCHEDULED):
            raise RuntimeError("boom")
    assert get_request_priority() is None


def test_request_priority_context_manager_nests():
    with request_priority(PriorityTier.SCHEDULED):
        assert current_priority_legacy() == "scheduled"
        with request_priority(PriorityTier.DREAMING):
            assert current_priority_legacy() == "dreaming"
        assert current_priority_legacy() == "scheduled"
