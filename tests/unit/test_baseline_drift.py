"""Unit tests for ezmsg.simbiophys.baseline_drift module."""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from numpy.fft import rfft, rfftfreq

from ezmsg.simbiophys import (
    BaselineDriftSettings,
    BaselineDriftTransformer,
)


def _msg(n, fs, offset, n_ch=1):
    data = np.zeros((n, n_ch)) if n_ch else np.zeros(n)
    return AxisArray(data, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=fs, offset=offset)})


class TestBaselineDriftTransformer:
    def test_output_shape_and_addition(self):
        """Drift is added to the input, preserving shape."""
        tr = BaselineDriftTransformer(BaselineDriftSettings(scale=1.0, drift_fs=50.0, seed=0))
        base = np.full((3000, 4), 7.0)
        msg = AxisArray(base, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=30000.0, offset=0.0)})
        out = tr(msg)
        assert out.data.shape == (3000, 4)
        # Output differs from input (drift added) but stays near the constant baseline.
        assert not np.allclose(out.data, base)
        assert np.abs(out.data.mean() - 7.0) < 5.0

    def test_scale_zero_is_passthrough(self):
        tr = BaselineDriftTransformer(BaselineDriftSettings(scale=0.0, drift_fs=50.0, seed=0))
        base = np.random.default_rng(1).standard_normal((2000, 2))
        msg = AxisArray(base, dims=["time", "ch"], axes={"time": AxisArray.TimeAxis(fs=30000.0, offset=0.0)})
        out = tr(msg)
        np.testing.assert_allclose(out.data, base)

    def test_continuity_chunk_independent(self):
        """Chunked processing is bit-identical to a single-shot call."""
        fs = 30000.0
        single = BaselineDriftTransformer(BaselineDriftSettings(scale=1.0, drift_fs=50.0, seed=7))
        whole = single(_msg(5 * int(fs), fs, 0.0)).data

        chunked_tr = BaselineDriftTransformer(BaselineDriftSettings(scale=1.0, drift_fs=50.0, seed=7))
        parts, off = [], 0.0
        for _ in range(5):
            parts.append(chunked_tr(_msg(int(fs), fs, off)).data)
            off += 1.0
        chunked = np.concatenate(parts)
        np.testing.assert_allclose(whole, chunked, atol=1e-12)

    def test_reproducible_with_seed(self):
        a = BaselineDriftTransformer(BaselineDriftSettings(scale=1.0, seed=3))(_msg(3000, 30000.0, 0.0)).data
        b = BaselineDriftTransformer(BaselineDriftSettings(scale=1.0, seed=3))(_msg(3000, 30000.0, 0.0)).data
        np.testing.assert_allclose(a, b)

    def test_has_sub_hz_power(self):
        """The always-on drift produces a 1/f spectrum with real sub-Hz power."""
        fs = 30000.0
        tr = BaselineDriftTransformer(BaselineDriftSettings(scale=1.0, drift_fs=50.0, n_poles=24, seed=0))
        parts, off = [], 0.0
        for _ in range(30):  # 30 s so we resolve sub-Hz bins
            parts.append(tr(_msg(int(fs), fs, off)).data[:, 0])
            off += 1.0
        x = np.concatenate(parts)
        x = x - x.mean()
        f = rfftfreq(len(x), 1 / fs)
        p = np.abs(rfft(x)) ** 2
        # log-log slope over 0.1-10 Hz should be near -1 (pink).
        band = (f >= 0.1) & (f <= 10)
        slope = np.polyfit(np.log10(f[band]), np.log10(p[band]), 1)[0]
        assert -1.4 < slope < -0.7
        # Sub-Hz power must dominate mid-band power (the flat-baseline complaint).
        sub = p[(f > 0.1) & (f < 1.0)].mean()
        mid = p[(f >= 10) & (f < 100)].mean()
        assert sub > 20 * mid

    def test_finite_output(self):
        tr = BaselineDriftTransformer(BaselineDriftSettings(scale=10.0, drift_fs=50.0, n_poles=24, seed=0))
        off = 0.0
        for _ in range(50):
            out = tr(_msg(3000, 30000.0, off, n_ch=8))
            assert np.isfinite(out.data).all()
            off += 0.1
