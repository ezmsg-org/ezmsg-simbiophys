"""Unit tests for ezmsg.simbiophys.cosine_encoder module."""

import platform
import time

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys import (
    CosineEncoderSettings,
    CosineEncoderState,
    CosineEncoderTransformer,
)

requires_apple_silicon = pytest.mark.skipif(
    platform.machine() != "arm64" or platform.system() != "Darwin",
    reason="Requires Apple Silicon for MLX",
)


class TestCosineEncoderState:
    """Tests for CosineEncoderState."""

    def test_init_random_basic(self):
        """Test random parameter initialization."""
        state = CosineEncoderState()
        state.init_random(output_ch=10, seed=42)

        assert state.output_ch == 10
        assert state.baseline.shape == (1, 10)
        assert state.modulation.shape == (1, 10)
        assert state.pd.shape == (1, 10)
        assert state.speed_modulation.shape == (1, 10)

        # Check default values
        assert np.allclose(state.baseline, 0.0)
        assert np.allclose(state.modulation, 1.0)
        assert np.allclose(state.speed_modulation, 0.0)

        # Check preferred directions are in [0, 2*pi)
        assert np.all(state.pd >= 0)
        assert np.all(state.pd < 2 * np.pi)

    def test_init_random_custom_params(self):
        """Test random initialization with custom parameters."""
        state = CosineEncoderState()
        state.init_random(
            output_ch=5,
            baseline=15.0,
            modulation=30.0,
            speed_modulation=5.0,
            seed=123,
        )

        assert state.output_ch == 5
        assert np.allclose(state.baseline, 15.0)
        assert np.allclose(state.modulation, 30.0)
        assert np.allclose(state.speed_modulation, 5.0)

    def test_init_random_reproducible(self):
        """Test that seed produces reproducible results."""
        state1 = CosineEncoderState()
        state1.init_random(output_ch=10, seed=42)

        state2 = CosineEncoderState()
        state2.init_random(output_ch=10, seed=42)

        assert np.array_equal(state1.pd, state2.pd)

    def test_validation_shape_mismatch(self):
        """Test that mismatched shapes raise error."""
        state = CosineEncoderState()
        state.baseline = np.array([[1.0, 2.0]])
        state.modulation = np.array([[1.0, 2.0, 3.0]])
        state.pd = np.array([[1.0, 2.0]])
        state.speed_modulation = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="same shape"):
            state.validate()

    def test_validation_wrong_shape(self):
        """Test that wrong shape raises error."""
        state = CosineEncoderState()
        state.baseline = np.array([1.0, 2.0])  # 1D instead of 2D
        state.modulation = np.array([1.0, 2.0])
        state.pd = np.array([1.0, 2.0])
        state.speed_modulation = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="shape"):
            state.validate()

    def test_validation_empty(self):
        """Test that empty arrays raise error."""
        state = CosineEncoderState()
        state.baseline = np.array([[]]).reshape(1, 0)
        state.modulation = np.array([[]]).reshape(1, 0)
        state.pd = np.array([[]]).reshape(1, 0)
        state.speed_modulation = np.array([[]]).reshape(1, 0)

        with pytest.raises(ValueError, match="at least 1 channel"):
            state.validate()


class TestCosineEncoderTransformer:
    """Tests for CosineEncoderTransformer."""

    def test_basic_transform(self):
        """Test basic polar to encoded output transformation."""
        transformer = CosineEncoderTransformer(
            CosineEncoderSettings(
                output_ch=4,
                baseline=10.0,
                modulation=20.0,
                speed_modulation=0.0,
                seed=42,
            )
        )

        # Create polar input: magnitude=1, angle=0 (rightward direction)
        polar = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(polar, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        assert msg_out.data.shape == (3, 4)  # (n_samples, output_ch)
        assert "ch" in msg_out.axes
        assert msg_out.axes["ch"].data.shape == (4,)

    def test_stationary_baseline(self):
        """Test that zero magnitude produces baseline values."""
        transformer = CosineEncoderTransformer(
            CosineEncoderSettings(
                output_ch=3,
                baseline=15.0,
                modulation=25.0,
                speed_modulation=5.0,
                seed=42,
            )
        )

        # Zero magnitude (stationary)
        polar = np.array([[0.0, 0.0], [0.0, 0.0]])
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(polar, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # When magnitude=0, output = baseline + modulation*0*cos(...) + speed_mod*0 = baseline
        assert np.allclose(msg_out.data, 15.0)

    def test_directional_tuning(self):
        """Test that tuning varies with direction."""
        transformer = CosineEncoderTransformer(CosineEncoderSettings(output_ch=1, seed=42))

        # Manually set state with known preferred direction (shape: 1, output_ch)
        transformer._state.baseline = np.array([[10.0]])
        transformer._state.modulation = np.array([[20.0]])
        transformer._state.pd = np.array([[0.0]])  # Preferred direction = 0 (rightward)
        transformer._state.speed_modulation = np.array([[0.0]])
        transformer._state.ch_axis = AxisArray.CoordinateAxis(data=np.array(["ch0"]), dims=["ch"])
        transformer._hash = 0

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # Polar: magnitude=1, angle=0 (aligned with pd)
        aligned = AxisArray(np.array([[1.0, 0.0]]), dims=["time", "ch"], axes={"time": time_axis})
        output_aligned = transformer(aligned).data[0, 0]

        # Reset hash to reuse state
        transformer._hash = 0

        # Polar: magnitude=1, angle=pi (opposite to pd)
        opposite = AxisArray(np.array([[1.0, np.pi]]), dims=["time", "ch"], axes={"time": time_axis})
        output_opposite = transformer(opposite).data[0, 0]

        # Aligned should have higher output (cos(0 - 0) = 1 vs cos(pi - 0) = -1)
        # output_aligned = 10 + 20*1*1 = 30
        # output_opposite = 10 + 20*1*(-1) = -10
        assert output_aligned > output_opposite
        assert np.isclose(output_aligned, 30.0)
        assert np.isclose(output_opposite, -10.0)

    def test_speed_modulation(self):
        """Test speed modulation term (speed_modulation * magnitude)."""
        transformer = CosineEncoderTransformer(CosineEncoderSettings(output_ch=1, seed=42))

        # Manually set state (shape: 1, output_ch)
        transformer._state.baseline = np.array([[10.0]])
        transformer._state.modulation = np.array([[0.0]])  # No directional modulation
        transformer._state.pd = np.array([[0.0]])
        transformer._state.speed_modulation = np.array([[5.0]])
        transformer._state.ch_axis = AxisArray.CoordinateAxis(data=np.array(["ch0"]), dims=["ch"])
        transformer._hash = 0

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # Different magnitudes
        slow = AxisArray(np.array([[1.0, 0.0]]), dims=["time", "ch"], axes={"time": time_axis})
        output_slow = transformer(slow).data[0, 0]

        transformer._hash = 0

        fast = AxisArray(np.array([[2.0, 0.0]]), dims=["time", "ch"], axes={"time": time_axis})
        output_fast = transformer(fast).data[0, 0]

        # output = 10 + 0 + 5*magnitude
        assert np.isclose(output_slow, 10.0 + 5.0 * 1.0)
        assert np.isclose(output_fast, 10.0 + 5.0 * 2.0)

    def test_invalid_input_shape(self):
        """Test that invalid input shape raises error."""
        transformer = CosineEncoderTransformer(CosineEncoderSettings(output_ch=3, seed=42))

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # Wrong number of columns (should be 2)
        bad_input = AxisArray(np.array([[1.0, 2.0, 3.0]]), dims=["time", "ch"], axes={"time": time_axis})

        with pytest.raises(ValueError, match="shape"):
            transformer(bad_input)

    def test_multiple_samples(self):
        """Test processing multiple samples at once."""
        transformer = CosineEncoderTransformer(CosineEncoderSettings(output_ch=5, seed=42))

        # 100 polar coordinate samples
        n_samples = 100
        np.random.seed(123)
        magnitude = np.abs(np.random.randn(n_samples, 1))
        angle = np.random.uniform(-np.pi, np.pi, (n_samples, 1))
        polar = np.hstack([magnitude, angle])

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(polar, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        assert msg_out.data.shape == (n_samples, 5)


@requires_apple_silicon
def test_cosine_encoder_mlx_benchmark():
    """Benchmark CosineEncoderTransformer: numpy vs MLX input."""
    import mlx.core as mx

    n_samples = 500
    n_chunks = 200
    output_ch = 256
    fs = 100.0

    settings = CosineEncoderSettings(
        output_ch=output_ch,
        baseline=10.0,
        modulation=20.0,
        speed_modulation=5.0,
        seed=42,
    )

    # Pre-generate chunks as numpy
    rng = np.random.default_rng(42)
    np_chunks = []
    for i in range(n_chunks + 1):  # +1 for warmup
        magnitude = np.abs(rng.standard_normal((n_samples, 1))).astype(np.float32)
        angle = rng.uniform(-np.pi, np.pi, (n_samples, 1)).astype(np.float32)
        polar = np.hstack([magnitude, angle])
        np_chunks.append(
            AxisArray(
                polar,
                dims=["time", "ch"],
                axes={"time": AxisArray.LinearAxis(gain=1.0 / fs, offset=i * n_samples / fs)},
            )
        )

    # MLX versions
    mx_chunks = [AxisArray(data=mx.array(chunk.data), dims=chunk.dims, axes=chunk.axes) for chunk in np_chunks]

    # --- Numpy ---
    xformer_np = CosineEncoderTransformer(settings)
    xformer_np(np_chunks[0])  # Warmup

    t0 = time.perf_counter()
    np_outputs = [xformer_np(chunk) for chunk in np_chunks[1:]]
    t_numpy = time.perf_counter() - t0

    # --- MLX ---
    xformer_mx = CosineEncoderTransformer(settings)
    xformer_mx(mx_chunks[0])  # Warmup
    mx.eval(xformer_mx(mx_chunks[0]).data)

    t0 = time.perf_counter()
    mx_outputs = [xformer_mx(chunk) for chunk in mx_chunks[1:]]
    for out in mx_outputs:
        mx.eval(out.data)
    t_mlx = time.perf_counter() - t0

    # Verify output is MLX array
    last_mx = mx_outputs[-1]
    assert isinstance(last_mx.data, mx.array), f"Expected mx.array, got {type(last_mx.data)}"

    # Correctness: compare outputs
    for np_out, mx_out in zip(np_outputs, mx_outputs):
        np.testing.assert_allclose(np.asarray(mx_out.data), np_out.data, rtol=5e-3, atol=1e-4)

    print(
        f"\n  CosineEncoder benchmark ({n_chunks} chunks, {n_samples}×2 → {output_ch} ch):"
        f"\n    numpy: {t_numpy:.4f}s ({t_numpy / n_chunks * 1000:.2f} ms/chunk)"
        f"\n    mlx:   {t_mlx:.4f}s ({t_mlx / n_chunks * 1000:.2f} ms/chunk)"
        f"\n    ratio (mlx/numpy): {t_mlx / t_numpy:.2f}x"
    )
