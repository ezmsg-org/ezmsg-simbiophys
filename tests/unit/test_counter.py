"""Unit tests for ezmsg.simbiophys.counter module."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys import CounterProducer, CounterSettings


@pytest.mark.parametrize("block_size", [1, 20])
@pytest.mark.parametrize("fs", [10.0, 1000.0])
@pytest.mark.parametrize("n_ch", [3])
@pytest.mark.parametrize(
    "dispatch_rate", [None, "realtime", "ext_clock", 2.0, 20.0]
)  # "ext_clock" needs a separate test
@pytest.mark.parametrize("mod", [2**3, None])
@pytest.mark.asyncio
async def test_counter_producer_async(
    block_size: int,
    fs: float,
    n_ch: int,
    dispatch_rate: float | str | None,
    mod: int | None,
):
    """Test asynchronous CounterProducer via __acall__."""
    target_dur = 2.6  # 2.6 seconds per test
    if dispatch_rate is None:
        # No sleep / wait
        chunk_dur = 0.1
    elif isinstance(dispatch_rate, str):
        if dispatch_rate == "realtime":
            chunk_dur = block_size / fs
        elif dispatch_rate == "ext_clock":
            # No sleep / wait
            chunk_dur = 0.1
    else:
        # Note: float dispatch_rate will yield different number of samples than expected by target_dur and fs
        chunk_dur = 1.0 / dispatch_rate
    target_messages = int(target_dur / chunk_dur)

    # Create producer
    producer = CounterProducer(
        CounterSettings(
            n_time=block_size,
            fs=fs,
            n_ch=n_ch,
            dispatch_rate=dispatch_rate,
            mod=mod,
        )
    )

    # Run producer
    messages = [await producer.__acall__() for _ in range(target_messages)]

    # Test contents of individual messages
    for msg in messages:
        assert type(msg) is AxisArray
        assert msg.data.shape == (block_size, n_ch)
        assert "time" in msg.axes
        assert msg.axes["time"].gain == 1 / fs
        assert "ch" in msg.axes
        assert np.array_equal(msg.axes["ch"].data, np.array([f"Ch{_}" for _ in range(n_ch)]))

    agg = AxisArray.concatenate(*messages, dim="time")

    target_samples = block_size * target_messages
    expected_data = np.arange(target_samples)
    if mod is not None:
        expected_data = expected_data % mod
    assert np.array_equal(agg.data[:, 0], expected_data)

    offsets = np.array([m.axes["time"].offset for m in messages])
    expected_offsets = np.arange(target_messages) * block_size / fs
    if dispatch_rate == "realtime":
        expected_offsets += offsets[0]  # offsets are in real-time
        atol = 0.002
        assert np.allclose(offsets[2:], expected_offsets[2:], atol=atol)
    elif dispatch_rate == "ext_clock":
        # In ext_clock mode without an actual external clock, offsets are
        # time.monotonic() at call time (nearly identical for fast calls).
        # The real use case is with Clock providing synchronized timestamps.
        # Just verify offsets are monotonically non-decreasing.
        assert np.all(np.diff(offsets) >= 0)
    else:
        # Offsets are synthetic.
        atol = 1.0e-8
        assert np.allclose(offsets[2:], expected_offsets[2:], atol=atol)


@pytest.mark.asyncio
async def test_counter_ext_clock_with_timestamps():
    """Test ext_clock mode with explicit timestamps from external clock."""
    block_size = 100
    fs = 1000.0
    n_ch = 2

    producer = CounterProducer(
        CounterSettings(
            n_time=block_size,
            fs=fs,
            n_ch=n_ch,
            dispatch_rate="ext_clock",
        )
    )

    # Simulate external clock providing timestamps
    base_time = 1000.0  # Arbitrary base time
    timestamps = [base_time + i * 0.1 for i in range(10)]  # 100ms apart

    messages = []
    for ts in timestamps:
        producer.set_clock_offset(ts)
        msg = await producer.__acall__()
        messages.append(msg)

    # Verify offsets match the provided timestamps
    offsets = [m.axes["time"].offset for m in messages]
    np.testing.assert_array_equal(offsets, timestamps)

    # Verify data is still correct
    agg = AxisArray.concatenate(*messages, dim="time")
    expected_data = np.arange(block_size * len(timestamps))
    assert np.array_equal(agg.data[:, 0], expected_data)
