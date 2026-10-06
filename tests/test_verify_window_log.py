"""Tests for verify window logging in LM Studio adapter."""
import json
import logging
from types import SimpleNamespace

import pytest

from laplace.adapter import ModelLoadError
from laplace.adapters import lmstudio
from tests.test_load_recovery import A, Runtime, resident


@pytest.mark.asyncio
async def test_verify_window_logged_when_load_succeeds_but_model_not_resident(caplog):
    """Test that window is logged when load command succeeds but model never appears in lms ps."""
    rt = Runtime()
    # Make the load command succeed (rc=0) but don't add model to resident list
    # This simulates the case where the load command succeeds but the model never becomes resident
    original_run = rt.run
    
    async def run(args, timeout):
        if args[0] == 'load' and not rt.loads:
            rt.loads.append(args[1])
            # Return success (rc=0) but don't add the model to rows
            return 0, '', ''
        return await original_run(args, timeout)
    
    rt.adapter._run = run
    
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)
    
    # Check that there's exactly one ERROR log containing "window opened" and "reason=verify"
    window_logs = [
        record for record in caplog.records
        if record.levelno == logging.ERROR
        and "window opened" in record.message
        and "reason=verify" in record.message
    ]
    
    assert len(window_logs) == 1
    assert "model=" in window_logs[0].message
    assert A in window_logs[0].message


@pytest.mark.asyncio
async def test_rc_window_logged_when_load_fails(caplog):
    """Test that window is logged when load command fails with rc != 0."""
    rt = Runtime()
    original_run = rt.run
    
    async def run(args, timeout):
        if args[0] == 'load' and not rt.loads:
            rt.loads.append(args[1])
            # Return failure (rc=1) 
            return 1, '', 'Operation canceled'
        return await original_run(args, timeout)
    
    rt.adapter._run = run
    
    with pytest.raises(ModelLoadError):
        await rt.adapter.ensure_loaded(A, 4096)

    # Check that there's exactly one ERROR log containing "window opened" (regardless of reason)
    window_logs = [
        record for record in caplog.records
        if record.levelno == logging.ERROR
        and "window opened" in record.message
    ]

    assert len(window_logs) == 1
    assert "reason=rc" in window_logs[0].message
    assert "model=" in window_logs[0].message
    assert A in window_logs[0].message