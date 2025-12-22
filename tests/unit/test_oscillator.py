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
