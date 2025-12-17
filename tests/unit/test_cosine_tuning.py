"""Unit tests for ezmsg.simbiophys.cosine_tuning module."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys import (
    CosineTuningParams,
    CosineTuningSettings,
    CosineTuningTransformer,
)


class TestCosineTuningParams:
    """Tests for CosineTuningParams dataclass."""

    def test_from_random_basic(self):
        """Test random parameter generation."""
        params = CosineTuningParams.from_random(n_units=10, seed=42)

        assert params.n_units == 10
        assert params.b0.shape == (10,)
        assert params.m.shape == (10,)
        assert params.pd.shape == (10,)
        assert params.bs.shape == (10,)

        # Check default values
        assert np.allclose(params.b0, 10.0)  # baseline_hz default
        assert np.allclose(params.m, 20.0)  # modulation_hz default
        assert np.allclose(params.bs, 0.0)  # speed_modulation_hz default

        # Check preferred directions are in [0, 2*pi)
        assert np.all(params.pd >= 0)
        assert np.all(params.pd < 2 * np.pi)

    def test_from_random_custom_params(self):
        """Test random generation with custom parameters."""
        params = CosineTuningParams.from_random(
            n_units=5,
            baseline_hz=15.0,
            modulation_hz=30.0,
            speed_modulation_hz=5.0,
            seed=123,
        )

        assert params.n_units == 5
        assert np.allclose(params.b0, 15.0)
        assert np.allclose(params.m, 30.0)
        assert np.allclose(params.bs, 5.0)

    def test_from_random_reproducible(self):
        """Test that seed produces reproducible results."""
        params1 = CosineTuningParams.from_random(n_units=10, seed=42)
        params2 = CosineTuningParams.from_random(n_units=10, seed=42)

        assert np.array_equal(params1.pd, params2.pd)

    def test_validation_shape_mismatch(self):
        """Test that mismatched shapes raise error."""
        with pytest.raises(ValueError, match="same shape"):
            CosineTuningParams(
                b0=np.array([1.0, 2.0]),
                m=np.array([1.0, 2.0, 3.0]),
                pd=np.array([1.0, 2.0]),
                bs=np.array([1.0, 2.0]),
            )

    def test_validation_not_1d(self):
        """Test that non-1D arrays raise error."""
        with pytest.raises(ValueError, match="1D"):
            CosineTuningParams(
                b0=np.array([[1.0, 2.0]]),
                m=np.array([[1.0, 2.0]]),
                pd=np.array([[1.0, 2.0]]),
                bs=np.array([[1.0, 2.0]]),
            )

    def test_validation_empty(self):
        """Test that empty arrays raise error."""
        with pytest.raises(ValueError, match="length >= 1"):
            CosineTuningParams(
                b0=np.array([]),
                m=np.array([]),
                pd=np.array([]),
                bs=np.array([]),
            )


class TestCosineTuningTransformer:
    """Tests for CosineTuningTransformer."""

    def test_basic_transform(self):
        """Test basic velocity to firing rate transformation."""
        transformer = CosineTuningTransformer(
            CosineTuningSettings(
                n_units=4,
                baseline_hz=10.0,
                modulation_hz=20.0,
                speed_modulation_hz=0.0,
                seed=42,
            )
        )

        # Create velocity input: moving right at unit speed
        velocity = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(velocity, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        assert msg_out.data.shape == (3, 4)  # (n_samples, n_units)
        assert "ch" in msg_out.axes
        assert msg_out.axes["ch"].data.shape == (4,)

        # All rates should be positive (baseline + modulation term)
        assert np.all(msg_out.data >= 0)

    def test_stationary_baseline(self):
        """Test that stationary input produces baseline rates."""
        transformer = CosineTuningTransformer(
            CosineTuningSettings(
                n_units=3,
                baseline_hz=15.0,
                modulation_hz=25.0,
                speed_modulation_hz=5.0,
                seed=42,
            )
        )

        # Zero velocity
        velocity = np.array([[0.0, 0.0], [0.0, 0.0]])
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(velocity, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # When speed=0, rate = b0 + m*0*cos(...) + bs*0 = b0
        # But arctan2(0,0) can be arbitrary, so just check rates are ~baseline
        assert np.allclose(msg_out.data, 15.0)

    def test_directional_tuning(self):
        """Test that tuning varies with direction."""
        # Create transformer with known preferred direction
        params = CosineTuningParams(
            b0=np.array([10.0]),
            m=np.array([20.0]),
            pd=np.array([0.0]),  # Preferred direction = 0 (rightward)
            bs=np.array([0.0]),
        )

        transformer = CosineTuningTransformer(CosineTuningSettings(n_units=1, seed=42))
        # Manually set params
        transformer._state.params = params
        transformer._hash = 0

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # Moving right (direction = 0, aligned with pd)
        right = AxisArray(np.array([[1.0, 0.0]]), dims=["time", "ch"], axes={"time": time_axis})
        rate_right = transformer(right).data[0, 0]

        # Reset state for fresh processing
        transformer._hash = 0
        transformer._state.params = params

        # Moving left (direction = pi, opposite to pd)
        left = AxisArray(np.array([[-1.0, 0.0]]), dims=["time", "ch"], axes={"time": time_axis})
        rate_left = transformer(left).data[0, 0]

        # Right should have higher rate (cos(0 - 0) = 1 vs cos(pi - 0) = -1)
        # rate_right = 10 + 20*1*1 + 0 = 30
        # rate_left = 10 + 20*1*(-1) + 0 = -10, but clipped to min_rate=0
        assert rate_right > rate_left
        assert np.isclose(rate_right, 30.0)

    def test_speed_modulation(self):
        """Test speed modulation term (bs * |v|)."""
        params = CosineTuningParams(
            b0=np.array([10.0]),
            m=np.array([0.0]),  # No directional modulation
            pd=np.array([0.0]),
            bs=np.array([5.0]),  # Speed modulation
        )

        transformer = CosineTuningTransformer(CosineTuningSettings(n_units=1, seed=42))
        transformer._state.params = params
        transformer._hash = 0

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # Different speeds, same direction
        slow = AxisArray(np.array([[1.0, 0.0]]), dims=["time", "ch"], axes={"time": time_axis})
        rate_slow = transformer(slow).data[0, 0]

        transformer._hash = 0
        transformer._state.params = params

        fast = AxisArray(np.array([[2.0, 0.0]]), dims=["time", "ch"], axes={"time": time_axis})
        rate_fast = transformer(fast).data[0, 0]

        # rate = 10 + 0 + 5*|v|
        assert np.isclose(rate_slow, 10.0 + 5.0 * 1.0)
        assert np.isclose(rate_fast, 10.0 + 5.0 * 2.0)

    def test_min_rate_clipping(self):
        """Test that rates are clipped to min_rate."""
        params = CosineTuningParams(
            b0=np.array([5.0]),
            m=np.array([20.0]),
            pd=np.array([0.0]),
            bs=np.array([0.0]),
        )

        transformer = CosineTuningTransformer(CosineTuningSettings(n_units=1, min_rate=0.0, seed=42))
        transformer._state.params = params
        transformer._hash = 0

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # Moving opposite to preferred direction
        # rate = 5 + 20*1*cos(pi) = 5 - 20 = -15, should be clipped to 0
        msg_in = AxisArray(np.array([[-1.0, 0.0]]), dims=["time", "ch"], axes={"time": time_axis})
        msg_out = transformer(msg_in)

        assert msg_out.data[0, 0] >= 0.0

    def test_invalid_input_shape(self):
        """Test that invalid input shape raises error."""
        transformer = CosineTuningTransformer(CosineTuningSettings(n_units=3, seed=42))

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # Wrong number of columns (should be 2)
        bad_input = AxisArray(np.array([[1.0, 2.0, 3.0]]), dims=["time", "ch"], axes={"time": time_axis})

        with pytest.raises(ValueError, match="shape"):
            transformer(bad_input)

    def test_multiple_samples(self):
        """Test processing multiple samples at once."""
        transformer = CosineTuningTransformer(CosineTuningSettings(n_units=5, seed=42))

        # 100 velocity samples
        n_samples = 100
        np.random.seed(123)
        velocity = np.random.randn(n_samples, 2)

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(velocity, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        assert msg_out.data.shape == (n_samples, 5)
        assert np.all(msg_out.data >= 0)  # All rates non-negative
