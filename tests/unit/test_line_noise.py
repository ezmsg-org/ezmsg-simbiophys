"""Unit tests for ezmsg.simbiophys.line_noise module."""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from numpy.fft import rfft, rfftfreq

from ezmsg.simbiophys import LineNoiseSettings, LineNoiseTransformer


def _msg(n, fs, offset, n_ch=4, val=0.0):
    return AxisArray(
        np.full((n, n_ch), val), dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=fs, offset=offset)}
    )


class TestLineNoiseTransformer:
    def test_disabled_is_passthrough(self):
        tr = LineNoiseTransformer(LineNoiseSettings(freq=None))
        m = _msg(1000, 30000.0, 0.0, val=3.0)
        out = tr(m)
        np.testing.assert_array_equal(out.data, m.data)

    def test_added_to_all_channels_equally(self):
        tr = LineNoiseTransformer(LineNoiseSettings(freq=60.0, amp=5.0, seed=1))
        out = tr(_msg(3000, 30000.0, 0.0, n_ch=4)).data
        for ch in range(1, 4):
            np.testing.assert_allclose(out[:, 0], out[:, ch])

    def test_amplitude_and_peak_frequency(self):
        fs = 30000.0
        tr = LineNoiseTransformer(LineNoiseSettings(freq=60.0, amp=7.0, drift_rate=0.0, seed=1))
        # No drift -> pure 60 Hz tone of amplitude 7.
        y = tr(_msg(int(fs), fs, 0.0, n_ch=1)).data[:, 0]
        assert abs(y.max() - 7.0) < 0.1
        f = rfftfreq(len(y), 1 / fs)
        p = np.abs(rfft(y - y.mean())) ** 2
        assert abs(f[np.argmax(p)] - 60.0) < 0.5

    def test_continuity_across_chunks(self):
        fs = 30000.0
        tr = LineNoiseTransformer(LineNoiseSettings(freq=60.0, amp=1.0, drift_rate=0.1, seed=3))
        parts, off = [], 0.0
        for _ in range(20):
            parts.append(tr(_msg(int(fs), fs, off, n_ch=1)).data[:, 0])
            off += 1.0
        y = np.concatenate(parts)
        d = np.abs(np.diff(y))
        boundaries = [int(fs) * k - 1 for k in range(1, 20)]
        # Boundary steps should be no worse than interior steps (phase is continuous).
        assert d[boundaries].max() < 1.5 * np.delete(d, boundaries).max()

    def test_frequency_bounded(self):
        fs = 30000.0
        # Aggressive drift so the bound is actually reached.
        tr = LineNoiseTransformer(LineNoiseSettings(freq=60.0, amp=1.0, drift_rate=2.0, drift_bound=1.5, seed=5))
        off = 0.0
        for _ in range(1200):  # 20 min
            tr(_msg(int(fs), fs, off, n_ch=1))
            assert abs(tr._state.freq_off[0, 0]) <= 1.5 + 1e-9
            off += 1.0

    def test_drift_rate_magnitude(self):
        fs = 30000.0
        tr = LineNoiseTransformer(LineNoiseSettings(freq=60.0, amp=1.0, drift_rate=0.1, drift_bound=99.0, seed=7))
        offs, off = [], 0.0
        for _ in range(600):  # 10 min, unbounded to measure the walk itself
            tr(_msg(int(fs), fs, off, n_ch=1))
            offs.append(tr._state.freq_off[0, 0])
            off += 1.0
        offs = np.array(offs)
        one_min_drift = np.std(offs[60:] - offs[:-60])
        assert 0.05 < one_min_drift < 0.2  # ~0.1 Hz/min
