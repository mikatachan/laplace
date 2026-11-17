"""fit_context_output_from_window: no-max_tokens requests may use the model's window."""

from __future__ import annotations

import json

import pytest

from laplace_serve.app import _estimate_need_tokens, _output_allowance
from laplace_serve.config import LaplacedConfig


def test_default_is_off() -> None:
    assert LaplacedConfig().fit_context_output_from_window is False


def test_from_dict_true() -> None:
    config = LaplacedConfig.from_dict({"fit_context_output_from_window": True})
    assert config.fit_context_output_from_window is True


def test_from_dict_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="fit_context_output_from_window"):
        LaplacedConfig.from_dict({"fit_context_output_from_window": "yes"})


def test_no_max_tokens_allowance_shifts_need() -> None:
    body = json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    ).encode()
    assert (
        _estimate_need_tokens(body, 262144) - _estimate_need_tokens(body, 8192)
        == 262144 - 8192
    )


def test_caller_max_tokens_wins_over_allowance() -> None:
    body = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
        }
    ).encode()
    assert _estimate_need_tokens(body, 8192) == _estimate_need_tokens(body, 262144)


def test_output_allowance_flag_off() -> None:
    config = LaplacedConfig(fit_context_min_output_tokens=8192)
    assert _output_allowance(config, 262144) == 8192


def test_output_allowance_flag_on_uses_window() -> None:
    config = LaplacedConfig(
        fit_context_min_output_tokens=8192, fit_context_output_from_window=True
    )
    assert _output_allowance(config, 262144) == 262144


def test_output_allowance_flag_on_no_ctx_falls_back() -> None:
    config = LaplacedConfig(
        fit_context_min_output_tokens=8192, fit_context_output_from_window=True
    )
    assert _output_allowance(config, None) == 8192
