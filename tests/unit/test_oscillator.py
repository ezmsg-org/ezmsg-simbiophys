"""Unit tests for ezmsg.simbiophys.oscillator module."""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys import SinGeneratorSettings, SinTransformer


def test_sin_transformer(freq: float = 1.0, amp: float = 1.0, phase: float = 0.0):
    """Test SinTransformer via __call__."""
    axis: str | None = "time"
    srate = max(4.0 * freq, 1000.0)
    sim_dur = 30.0
    n_samples = int(srate * sim_dur)
    n_msgs = min(n_samples, 10)
    axis_idx = 0

    # Create input messages with counter data
    messages = []
    for split_dat in np.array_split(np.arange(n_samples)[:, None], n_msgs, axis=axis_idx):
        _time_axis = AxisArray.TimeAxis(fs=srate, offset=float(split_dat[0, 0]))
        messages.append(AxisArray(split_dat, dims=["time", "ch"], axes={"time": _time_axis}))

    def f_test(t):
        return amp * np.sin(2 * np.pi * freq * t + phase)

    # Create transformer
    transformer = SinTransformer(SinGeneratorSettings(axis=axis, freq=freq, amp=amp, phase=phase))

    # Process messages
    results = []
    for msg in messages:
        res = transformer(msg)
        assert np.allclose(res.data, f_test(msg.data / srate))
        results.append(res)

    concat_ax_arr = AxisArray.concatenate(*results, dim="time")
    assert np.allclose(concat_ax_arr.data, f_test(np.arange(n_samples) / srate)[:, None])
