"""Unit tests for ezmsg.simbiophys.oscillator module."""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys import SinGeneratorSettings, SinTransformer


def test_sin_transformer(freq: float = 1.0, amp: float = 1.0, phase: float = 0.0):
    """Test SinTransformer via __call__."""
    n_ch = 1
    srate = max(4.0 * freq, 1000.0)
    sim_dur = 30.0
    n_samples = int(srate * sim_dur)
    n_msgs = min(n_samples, 10)

    # Create input messages with counter data (integer sample counts)
    messages = []
    counter = 0
    samples_per_msg = n_samples // n_msgs
    for i in range(n_msgs):
        n = samples_per_msg if i < n_msgs - 1 else n_samples - counter
        sample_indices = np.arange(counter, counter + n)
        _time_axis = AxisArray.TimeAxis(fs=srate, offset=counter / srate)
        messages.append(AxisArray(sample_indices, dims=["time"], axes={"time": _time_axis}))
        counter += n

    def f_test(t):
        return amp * np.sin(2 * np.pi * freq * t + phase)

    # Create transformer
    transformer = SinTransformer(SinGeneratorSettings(n_ch=n_ch, freq=freq, amp=amp, phase=phase))

    # Process messages
    results = []
    for msg in messages:
        res = transformer(msg)
        # Check output shape
        assert res.data.shape == (len(msg.data), n_ch)
        # Check values
        t = msg.data / srate
        expected = f_test(t)[:, np.newaxis]
        assert np.allclose(res.data, expected)
        results.append(res)

    # Verify concatenated output
    concat_ax_arr = AxisArray.concatenate(*results, dim="time")
    assert np.allclose(concat_ax_arr.data, f_test(np.arange(n_samples) / srate)[:, np.newaxis])


def test_sin_transformer_multi_channel():
    """Test SinTransformer with multiple channels."""
    n_ch = 4
    freq = 10.0
    srate = 1000.0
    n_samples = 100

    # Create input with counter data
    sample_indices = np.arange(n_samples)
    _time_axis = AxisArray.TimeAxis(fs=srate, offset=0.0)
    msg = AxisArray(sample_indices, dims=["time"], axes={"time": _time_axis})

    # Create transformer
    transformer = SinTransformer(SinGeneratorSettings(n_ch=n_ch, freq=freq))

    # Process
    result = transformer(msg)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)
    assert result.dims == ["time", "ch"]

    # All channels should have identical values
    for ch in range(1, n_ch):
        np.testing.assert_allclose(result.data[:, 0], result.data[:, ch])


def test_sin_transformer_per_channel_freq():
    """Test SinTransformer with per-channel frequencies."""
    n_ch = 3
    freqs = [5.0, 10.0, 20.0]
    amp = 1.0
    phase = 0.0
    srate = 1000.0
    n_samples = 100

    # Create input with counter data
    sample_indices = np.arange(n_samples)
    _time_axis = AxisArray.TimeAxis(fs=srate, offset=0.0)
    msg = AxisArray(sample_indices, dims=["time"], axes={"time": _time_axis})

    # Create transformer with per-channel freqs
    transformer = SinTransformer(SinGeneratorSettings(n_ch=n_ch, freq=freqs, amp=amp, phase=phase))

    # Process
    result = transformer(msg)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)

    # Verify each channel has correct frequency
    t = sample_indices / srate
    for ch, freq in enumerate(freqs):
        expected = amp * np.sin(2 * np.pi * freq * t + phase)
        np.testing.assert_allclose(result.data[:, ch], expected, rtol=1e-10)


def test_sin_transformer_per_channel_all_params():
    """Test SinTransformer with per-channel freq, amp, and phase."""
    n_ch = 4
    freqs = [5.0, 10.0, 15.0, 20.0]
    amps = [1.0, 2.0, 0.5, 1.5]
    phases = [0.0, np.pi / 4, np.pi / 2, np.pi]
    srate = 1000.0
    n_samples = 200

    # Create input with counter data
    sample_indices = np.arange(n_samples)
    _time_axis = AxisArray.TimeAxis(fs=srate, offset=0.0)
    msg = AxisArray(sample_indices, dims=["time"], axes={"time": _time_axis})

    # Create transformer with all per-channel params
    transformer = SinTransformer(SinGeneratorSettings(n_ch=n_ch, freq=freqs, amp=amps, phase=phases))

    # Process
    result = transformer(msg)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)

    # Verify each channel (use atol for values near zero)
    t = sample_indices / srate
    for ch in range(n_ch):
        expected = amps[ch] * np.sin(2 * np.pi * freqs[ch] * t + phases[ch])
        np.testing.assert_allclose(result.data[:, ch], expected, rtol=1e-10, atol=1e-14)


def test_sin_transformer_mixed_scalar_array():
    """Test SinTransformer with mixed scalar and array params."""
    n_ch = 3
    freqs = [5.0, 10.0, 20.0]  # per-channel
    amp = 2.0  # scalar - same for all channels
    phase = 0.0  # scalar
    srate = 1000.0
    n_samples = 100

    # Create input with counter data
    sample_indices = np.arange(n_samples)
    _time_axis = AxisArray.TimeAxis(fs=srate, offset=0.0)
    msg = AxisArray(sample_indices, dims=["time"], axes={"time": _time_axis})

    # Create transformer
    transformer = SinTransformer(SinGeneratorSettings(n_ch=n_ch, freq=freqs, amp=amp, phase=phase))

    # Process
    result = transformer(msg)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)

    # Verify each channel
    t = sample_indices / srate
    for ch, freq in enumerate(freqs):
        expected = amp * np.sin(2 * np.pi * freq * t + phase)
        np.testing.assert_allclose(result.data[:, ch], expected, rtol=1e-10)


def test_sin_transformer_array_length_mismatch():
    """Test SinTransformer raises error when array length doesn't match n_ch."""
    import pytest

    n_ch = 4
    freqs = [5.0, 10.0, 20.0]  # length 3, but n_ch is 4
    srate = 1000.0

    # Create input
    sample_indices = np.arange(100)
    _time_axis = AxisArray.TimeAxis(fs=srate, offset=0.0)
    msg = AxisArray(sample_indices, dims=["time"], axes={"time": _time_axis})

    # Create transformer - should not raise here
    transformer = SinTransformer(SinGeneratorSettings(n_ch=n_ch, freq=freqs))

    # Should raise ValueError when processing
    with pytest.raises(ValueError, match="freq has length 3 but n_ch is 4"):
        transformer(msg)


def test_sin_transformer_numpy_array_input():
    """Test SinTransformer accepts numpy arrays for per-channel params."""
    n_ch = 3
    freqs = np.array([5.0, 10.0, 20.0])
    amps = np.array([1.0, 2.0, 0.5])
    phases = np.array([0.0, np.pi / 4, np.pi / 2])
    srate = 1000.0
    n_samples = 100

    # Create input with counter data
    sample_indices = np.arange(n_samples)
    _time_axis = AxisArray.TimeAxis(fs=srate, offset=0.0)
    msg = AxisArray(sample_indices, dims=["time"], axes={"time": _time_axis})

    # Create transformer with numpy arrays
    transformer = SinTransformer(SinGeneratorSettings(n_ch=n_ch, freq=freqs, amp=amps, phase=phases))

    # Process
    result = transformer(msg)

    # Check output shape
    assert result.data.shape == (n_samples, n_ch)

    # Verify each channel (use atol for values near zero)
    t = sample_indices / srate
    for ch in range(n_ch):
        expected = amps[ch] * np.sin(2 * np.pi * freqs[ch] * t + phases[ch])
        np.testing.assert_allclose(result.data[:, ch], expected, rtol=1e-10, atol=1e-14)
