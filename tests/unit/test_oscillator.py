"""Unit tests for ezmsg.simbiophys.oscillator module."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys import SinGeneratorSettings, SinProducer


def test_sin_generator_basic(freq: float = 1.0, amp: float = 1.0, phase: float = 0.0):
    """Test SinProducer via __call__."""
    n_ch = 1
    srate = max(4.0 * freq, 1000.0)
    sim_dur = 30.0
    n_samples = int(srate * sim_dur)
    n_msgs = min(n_samples, 10)
    samples_per_msg = n_samples // n_msgs

    def f_test(t):
        return amp * np.sin(2 * np.pi * freq * t + phase)

    # Create producer with clock-driven settings
    producer = SinProducer(
        SinGeneratorSettings(fs=srate, n_time=samples_per_msg, n_ch=n_ch, freq=freq, amp=amp, phase=phase)
    )

    # Process clock ticks
    results = []
    for i in range(n_msgs):
        offset = i * samples_per_msg / srate
        clock_tick = AxisArray.TimeAxis(fs=srate, offset=offset)
        res = producer(clock_tick)

        # Check output shape
        expected_n = samples_per_msg if i < n_msgs - 1 else n_samples - i * samples_per_msg
        # Note: With fixed n_time, all chunks have the same size
        assert res.data.shape == (expected_n, n_ch)
        results.append(res)

    # Verify concatenated output
    concat_ax_arr = AxisArray.concatenate(*results, dim="time")
    total_samples = concat_ax_arr.data.shape[0]
    t = np.arange(total_samples) / srate
    expected = f_test(t)[:, np.newaxis]
    np.testing.assert_allclose(concat_ax_arr.data, expected, rtol=1e-10)


def test_sin_generator_multi_channel():
    """Test SinProducer with multiple channels."""
    n_ch = 4
    freq = 10.0
    srate = 1000.0
    n_samples = 100

    # Create producer
    producer = SinProducer(SinGeneratorSettings(fs=srate, n_time=n_samples, n_ch=n_ch, freq=freq))

    # Process single clock tick
    clock_tick = AxisArray.TimeAxis(fs=srate, offset=0.0)
    result = producer(clock_tick)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)
    assert result.dims == ["time", "ch"]

    # All channels should have identical values
    for ch in range(1, n_ch):
        np.testing.assert_allclose(result.data[:, 0], result.data[:, ch])


def test_sin_generator_per_channel_freq():
    """Test SinProducer with per-channel frequencies."""
    n_ch = 3
    freqs = [5.0, 10.0, 20.0]
    amp = 1.0
    phase = 0.0
    srate = 1000.0
    n_samples = 100

    # Create producer with per-channel freqs
    producer = SinProducer(
        SinGeneratorSettings(fs=srate, n_time=n_samples, n_ch=n_ch, freq=freqs, amp=amp, phase=phase)
    )

    # Process
    clock_tick = AxisArray.TimeAxis(fs=srate, offset=0.0)
    result = producer(clock_tick)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)

    # Verify each channel has correct frequency
    t = np.arange(n_samples) / srate
    for ch, freq in enumerate(freqs):
        expected = amp * np.sin(2 * np.pi * freq * t + phase)
        np.testing.assert_allclose(result.data[:, ch], expected, rtol=1e-10)


def test_sin_generator_per_channel_all_params():
    """Test SinProducer with per-channel freq, amp, and phase."""
    n_ch = 4
    freqs = [5.0, 10.0, 15.0, 20.0]
    amps = [1.0, 2.0, 0.5, 1.5]
    phases = [0.0, np.pi / 4, np.pi / 2, np.pi]
    srate = 1000.0
    n_samples = 200

    # Create producer with all per-channel params
    producer = SinProducer(
        SinGeneratorSettings(fs=srate, n_time=n_samples, n_ch=n_ch, freq=freqs, amp=amps, phase=phases)
    )

    # Process
    clock_tick = AxisArray.TimeAxis(fs=srate, offset=0.0)
    result = producer(clock_tick)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)

    # Verify each channel (use atol for values near zero)
    t = np.arange(n_samples) / srate
    for ch in range(n_ch):
        expected = amps[ch] * np.sin(2 * np.pi * freqs[ch] * t + phases[ch])
        np.testing.assert_allclose(result.data[:, ch], expected, rtol=1e-10, atol=1e-14)


def test_sin_generator_mixed_scalar_array():
    """Test SinProducer with mixed scalar and array params."""
    n_ch = 3
    freqs = [5.0, 10.0, 20.0]  # per-channel
    amp = 2.0  # scalar - same for all channels
    phase = 0.0  # scalar
    srate = 1000.0
    n_samples = 100

    # Create producer
    producer = SinProducer(
        SinGeneratorSettings(fs=srate, n_time=n_samples, n_ch=n_ch, freq=freqs, amp=amp, phase=phase)
    )

    # Process
    clock_tick = AxisArray.TimeAxis(fs=srate, offset=0.0)
    result = producer(clock_tick)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)

    # Verify each channel
    t = np.arange(n_samples) / srate
    for ch, freq in enumerate(freqs):
        expected = amp * np.sin(2 * np.pi * freq * t + phase)
        np.testing.assert_allclose(result.data[:, ch], expected, rtol=1e-10)


def test_sin_generator_array_length_mismatch():
    """Test SinProducer raises error when array length doesn't match n_ch."""
    n_ch = 4
    freqs = [5.0, 10.0, 20.0]  # length 3, but n_ch is 4
    srate = 1000.0
    n_samples = 100

    # Create producer - should not raise here
    producer = SinProducer(SinGeneratorSettings(fs=srate, n_time=n_samples, n_ch=n_ch, freq=freqs))

    # Should raise ValueError when processing (during _reset_state)
    clock_tick = AxisArray.TimeAxis(fs=srate, offset=0.0)
    with pytest.raises(ValueError, match="freq has length 3 but n_ch is 4"):
        producer(clock_tick)


def test_sin_generator_numpy_array_input():
    """Test SinProducer accepts numpy arrays for per-channel params."""
    n_ch = 3
    freqs = np.array([5.0, 10.0, 20.0])
    amps = np.array([1.0, 2.0, 0.5])
    phases = np.array([0.0, np.pi / 4, np.pi / 2])
    srate = 1000.0
    n_samples = 100

    # Create producer with numpy arrays
    producer = SinProducer(
        SinGeneratorSettings(fs=srate, n_time=n_samples, n_ch=n_ch, freq=freqs, amp=amps, phase=phases)
    )

    # Process
    clock_tick = AxisArray.TimeAxis(fs=srate, offset=0.0)
    result = producer(clock_tick)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)

    # Verify each channel (use atol for values near zero)
    t = np.arange(n_samples) / srate
    for ch in range(n_ch):
        expected = amps[ch] * np.sin(2 * np.pi * freqs[ch] * t + phases[ch])
        np.testing.assert_allclose(result.data[:, ch], expected, rtol=1e-10, atol=1e-14)


def test_sin_generator_continuity_across_chunks():
    """Test that sine wave is continuous across multiple clock ticks."""
    n_ch = 1
    freq = 10.0
    srate = 1000.0
    n_samples_per_chunk = 50
    n_chunks = 4

    # Create producer
    producer = SinProducer(SinGeneratorSettings(fs=srate, n_time=n_samples_per_chunk, n_ch=n_ch, freq=freq))

    # Process multiple clock ticks
    results = []
    for i in range(n_chunks):
        offset = i * n_samples_per_chunk / srate
        clock_tick = AxisArray.TimeAxis(fs=srate, offset=offset)
        results.append(producer(clock_tick))

    # Concatenate and verify continuity
    concat = AxisArray.concatenate(*results, dim="time")
    total_samples = n_samples_per_chunk * n_chunks
    t = np.arange(total_samples) / srate
    expected = np.sin(2 * np.pi * freq * t)[:, np.newaxis]
    np.testing.assert_allclose(concat.data, expected, rtol=1e-10)


def test_sin_generator_variable_chunk_mode():
    """Test SinProducer with variable chunk sizes (n_time=None)."""
    n_ch = 1
    freq = 10.0
    srate = 1000.0

    # Create producer without fixed n_time
    producer = SinProducer(SinGeneratorSettings(fs=srate, n_time=None, n_ch=n_ch, freq=freq))

    # Clock ticks with different gains -> different chunk sizes
    clock_ticks = [
        AxisArray.LinearAxis(gain=0.1, offset=0.0),  # 100 samples
        AxisArray.LinearAxis(gain=0.05, offset=0.1),  # 50 samples
        AxisArray.LinearAxis(gain=0.2, offset=0.15),  # 200 samples
    ]

    results = []
    for tick in clock_ticks:
        results.append(producer(tick))

    assert results[0].data.shape == (100, n_ch)
    assert results[1].data.shape == (50, n_ch)
    assert results[2].data.shape == (200, n_ch)

    # Verify continuity
    concat = AxisArray.concatenate(*results, dim="time")
    t = np.arange(350) / srate
    expected = np.sin(2 * np.pi * freq * t)[:, np.newaxis]
    np.testing.assert_allclose(concat.data, expected, rtol=1e-10)
