"""Unit tests for ezmsg.simbiophys.line_noise module."""

import platform

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from numpy.fft import rfft, rfftfreq

from ezmsg.simbiophys import LineNoiseSettings, LineNoiseTransformer

requires_apple_silicon = pytest.mark.skipif(
    platform.machine() != "arm64" or platform.system() != "Darwin",
    reason="Requires Apple Silicon for MLX",
)


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
        rate = 0.05  # Hz per second
        tr = LineNoiseTransformer(LineNoiseSettings(freq=60.0, amp=1.0, drift_rate=rate, drift_bound=99.0, seed=7))
        offs, off = [], 0.0
        for _ in range(600):  # 600 one-second chunks, unbounded to measure the walk itself
            tr(_msg(int(fs), fs, off, n_ch=1))
            offs.append(tr._state.freq_off[0, 0])
            off += 1.0
        offs = np.array(offs)
        one_sec_drift = np.std(np.diff(offs))  # std of 1 s increments ~ drift_rate
        assert 0.5 * rate < one_sec_drift < 1.5 * rate

    @requires_apple_silicon
    @pytest.mark.parametrize("shape", [(300, 256), (1000, 256), (15000, 256), (250000,)])
    def test_mlx_matches_numpy_on_both_sides_of_dispatch_crossover(self, shape):
        import mlx.core as mx

        rng = np.random.default_rng(20)
        data = rng.standard_normal(shape).astype(np.float32)
        settings = LineNoiseSettings(freq=60.0, amp=10.0, drift_rate=0.002, seed=42)
        numpy_tr = LineNoiseTransformer(settings)
        mlx_tr = LineNoiseTransformer(settings)

        for chunk_idx in range(2):
            axes = {"time": AxisArray.TimeAxis(fs=30000.0, offset=chunk_idx * shape[0] / 30000.0)}
            dims = ["time"] if data.ndim == 1 else ["time", "ch"]
            out_np = numpy_tr(AxisArray(data, dims=dims, axes=axes))
            out_mx = mlx_tr(AxisArray(mx.array(data), dims=dims, axes=axes))
            mx.eval(out_mx.data)

            assert isinstance(out_mx.data, mx.array)
            np.testing.assert_allclose(np.asarray(out_mx.data), out_np.data, rtol=1e-5, atol=2e-6)
