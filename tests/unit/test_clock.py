"""Unit tests for ezmsg.simbiophys.clock module."""

import time

import ezmsg.core as ez
import numpy as np
import pytest

from ezmsg.simbiophys import ClockProducer, ClockSettings


@pytest.mark.parametrize("dispatch_rate", [None, 1.0, 2.0, 5.0, 10.0, 20.0])
def test_clock_producer_sync(dispatch_rate: float | None):
    """Test synchronous ClockProducer via __call__."""
    run_time = 1.0
    n_target = int(np.ceil(dispatch_rate * run_time)) if dispatch_rate else 100

    producer = ClockProducer(ClockSettings(dispatch_rate=dispatch_rate))

    result = []
    t_start = time.time()
    while len(result) < n_target:
        result.append(producer())
    t_elapsed = time.time() - t_start

    assert all([_ == ez.Flag() for _ in result])
    if dispatch_rate is not None:
        assert (run_time - 1 / dispatch_rate) < t_elapsed < (run_time + 0.2)
    else:
        # 100 usec per iteration is pretty generous
        assert t_elapsed < (n_target * 1e-4)


@pytest.mark.parametrize("dispatch_rate", [None, 2.0, 20.0])
@pytest.mark.asyncio
async def test_clock_producer_async(dispatch_rate: float | None):
    """Test asynchronous ClockProducer via __acall__."""
    run_time = 1.0
    n_target = int(np.ceil(dispatch_rate * run_time)) if dispatch_rate else 100

    producer = ClockProducer(ClockSettings(dispatch_rate=dispatch_rate))

    result = []
    t_start = time.time()
    while len(result) < n_target:
        result.append(await producer.__acall__())
    t_elapsed = time.time() - t_start

    assert all([_ == ez.Flag() for _ in result])
    if dispatch_rate:
        assert (run_time - 1.1 / dispatch_rate) < t_elapsed < (run_time + 0.1)
    else:
        # 100 usec per iteration is pretty generous
        assert t_elapsed < (n_target * 1e-4)
