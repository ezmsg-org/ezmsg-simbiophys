"""Unit tests for ezmsg.simbiophys.dynamic_colored_noise module."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys import (
    DynamicColoredNoiseSettings,
    DynamicColoredNoiseTransformer,
    compute_kasdin_coefficients,
)


class TestComputeKasdinCoefficients:
    """Tests for compute_kasdin_coefficients function."""

    def test_white_noise_coefficients(self):
        """Test that beta=0 (white noise) produces near-zero coefficients."""
        coeffs = compute_kasdin_coefficients(beta=0.0, n_poles=5)
        # For white noise, the filter should pass through unchanged
        # First coefficient: (0 - 0/2) * 1 / 1 = 0
        assert coeffs[0] == pytest.approx(0.0)

    def test_pink_noise_coefficients(self):
        """Test coefficients for pink noise (beta=1)."""
        coeffs = compute_kasdin_coefficients(beta=1.0, n_poles=5)
        # First coefficient: (0 - 1/2) * 1 / 1 = -0.5
        assert coeffs[0] == pytest.approx(-0.5)
        # Second coefficient: (1 - 1/2) * (-0.5) / 2 = -0.125
        assert coeffs[1] == pytest.approx(-0.125)

    def test_brown_noise_coefficients(self):
        """Test coefficients for brown noise (beta=2)."""
        coeffs = compute_kasdin_coefficients(beta=2.0, n_poles=5)
        # First coefficient: (0 - 2/2) * 1 / 1 = -1.0
        assert coeffs[0] == pytest.approx(-1.0)

    def test_coefficient_shape(self):
        """Test that output shape matches n_poles."""
        for n_poles in [1, 3, 5, 10, 20]:
            coeffs = compute_kasdin_coefficients(beta=1.0, n_poles=n_poles)
            assert coeffs.shape == (n_poles,)

    def test_coefficient_dtype(self):
        """Test that coefficients are float64."""
        coeffs = compute_kasdin_coefficients(beta=1.0, n_poles=5)
        assert coeffs.dtype == np.float64


class TestDynamicColoredNoiseTransformer:
    """Tests for DynamicColoredNoiseTransformer."""

    def test_basic_transform(self):
        """Test basic transformation produces output of correct shape."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, initial_beta=1.0, seed=42))

        # Input: constant beta values
        beta_input = np.full((100, 2), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        assert msg_out.data.shape == (100, 2)
        assert msg_out.data.dtype == np.float64

    def test_1d_input(self):
        """Test that 1D input produces 1D output."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, seed=42))

        beta_input = np.full(100, 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        assert msg_out.data.shape == (100,)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same output."""
        settings = DynamicColoredNoiseSettings(n_poles=5, seed=42)

        transformer1 = DynamicColoredNoiseTransformer(settings)
        transformer2 = DynamicColoredNoiseTransformer(settings)

        beta_input = np.full((50, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        out1 = transformer1(msg_in)
        out2 = transformer2(msg_in)

        assert np.array_equal(out1.data, out2.data)

    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different output."""
        transformer1 = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, seed=42))
        transformer2 = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, seed=123))

        beta_input = np.full((50, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        out1 = transformer1(msg_in)
        out2 = transformer2(msg_in)

        assert not np.array_equal(out1.data, out2.data)

    def test_scale_parameter(self):
        """Test that scale parameter affects output amplitude."""
        transformer_scale1 = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, scale=1.0, seed=42))
        transformer_scale2 = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, scale=2.0, seed=42))

        beta_input = np.full((100, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        out1 = transformer_scale1(msg_in)
        out2 = transformer_scale2(msg_in)

        # Scale 2 should have exactly 2x the values
        assert np.allclose(out2.data, 2.0 * out1.data)

    def test_continuity_across_chunks(self):
        """Test that output is continuous across chunk boundaries."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, seed=42))

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # Process in chunks
        outputs = []
        for i in range(5):
            beta_input = np.full((20, 1), 1.0)
            msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})
            outputs.append(transformer(msg_in).data)

        # Concatenate outputs
        concat_output = np.concatenate(outputs, axis=0)

        # Check no extreme discontinuities at chunk boundaries
        # The diff at boundaries should not be dramatically different from within chunks
        diffs = np.abs(np.diff(concat_output[:, 0]))
        boundary_indices = [19, 39, 59, 79]  # Indices just before chunk boundaries
        max_internal_diff = np.max(np.delete(diffs, boundary_indices))

        for idx in boundary_indices:
            # Boundary diffs should be within reasonable range of internal diffs
            assert diffs[idx] < 5 * max_internal_diff, f"Discontinuity at index {idx}"

    def test_varying_beta(self):
        """Test varying beta values across samples."""
        transformer = DynamicColoredNoiseTransformer(
            DynamicColoredNoiseSettings(n_poles=5, smoothing_tau=0.01, seed=42)
        )

        # Varying beta values
        beta_input = np.linspace(0.5, 1.5, 100)[:, np.newaxis]
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        assert msg_out.data.shape == (100, 1)
        # Output should still be valid noise
        assert np.all(np.isfinite(msg_out.data))

    def test_multi_channel(self):
        """Test multi-channel processing."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, seed=42))

        n_channels = 4
        beta_input = np.full((50, n_channels), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        assert msg_out.data.shape == (50, n_channels)

        # Each channel should be independent (different noise sequences)
        # Check that channels are not identical
        for i in range(n_channels - 1):
            assert not np.array_equal(msg_out.data[:, i], msg_out.data[:, i + 1])

    def test_beta_transition_smooth(self):
        """Test that beta transitions are smoothed."""
        transformer = DynamicColoredNoiseTransformer(
            DynamicColoredNoiseSettings(n_poles=5, smoothing_tau=0.05, seed=42)
        )

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # First chunk with beta=1
        beta1 = np.full((50, 1), 1.0)
        msg1 = AxisArray(beta1, dims=["time", "ch"], axes={"time": time_axis})
        out1 = transformer(msg1)

        # Second chunk with beta=2 (big change)
        beta2 = np.full((50, 1), 2.0)
        msg2 = AxisArray(beta2, dims=["time", "ch"], axes={"time": time_axis})
        out2 = transformer(msg2)

        # The transition should be smooth - check no extreme jumps at boundary
        boundary_diff = np.abs(out2.data[0, 0] - out1.data[-1, 0])
        max_internal_diff = max(np.max(np.abs(np.diff(out1.data[:, 0]))), np.max(np.abs(np.diff(out2.data[:, 0]))))

        # Boundary should not be dramatically worse than internal
        assert boundary_diff < 10 * max_internal_diff

    def test_spectral_properties_white_noise(self):
        """Test that beta=0 produces approximately white noise spectrum."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=10, initial_beta=0.0, seed=42))

        # Generate long sequence for spectral analysis
        n_samples = 10000
        beta_input = np.full((n_samples, 1), 0.0)
        time_axis = AxisArray.TimeAxis(fs=1000.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # Compute power spectrum
        fft = np.fft.rfft(msg_out.data[:, 0])
        psd = np.abs(fft) ** 2

        # For white noise, PSD should be relatively flat
        # Compare low and high frequency power (excluding DC)
        n_bins = len(psd)
        low_freq_power = np.mean(psd[1 : n_bins // 4])
        high_freq_power = np.mean(psd[3 * n_bins // 4 :])

        # For white noise, ratio should be close to 1
        ratio = low_freq_power / high_freq_power
        assert 0.3 < ratio < 3.0, f"White noise ratio: {ratio}"

    def test_spectral_properties_pink_noise(self):
        """Test that beta=1 produces approximately 1/f spectrum."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=10, initial_beta=1.0, seed=42))

        # Generate long sequence for spectral analysis
        n_samples = 10000
        beta_input = np.full((n_samples, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=1000.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # Compute power spectrum
        fft = np.fft.rfft(msg_out.data[:, 0])
        psd = np.abs(fft) ** 2

        # For pink noise, low frequencies should have more power than high
        n_bins = len(psd)
        low_freq_power = np.mean(psd[10 : n_bins // 4])  # Skip DC and very low
        high_freq_power = np.mean(psd[3 * n_bins // 4 :])

        # For pink noise, low should be significantly higher
        assert low_freq_power > high_freq_power, "Pink noise should have more low-freq power"

    def test_spectral_properties_brown_noise(self):
        """Test that beta=2 produces approximately 1/f^2 spectrum."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=10, initial_beta=2.0, seed=42))

        # Generate long sequence for spectral analysis
        n_samples = 10000
        beta_input = np.full((n_samples, 1), 2.0)
        time_axis = AxisArray.TimeAxis(fs=1000.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # Compute power spectrum
        fft = np.fft.rfft(msg_out.data[:, 0])
        psd = np.abs(fft) ** 2

        # For brown noise, ratio should be even larger than pink
        n_bins = len(psd)
        low_freq_power = np.mean(psd[10 : n_bins // 4])
        high_freq_power = np.mean(psd[3 * n_bins // 4 :])

        ratio = low_freq_power / high_freq_power
        assert ratio > 10, f"Brown noise should have much more low-freq power, ratio: {ratio}"

    def test_n_poles_effect(self):
        """Test that more poles extend low-frequency accuracy."""
        # This is a qualitative test - more poles should maintain
        # colored spectrum to lower frequencies
        settings_few = DynamicColoredNoiseSettings(n_poles=2, seed=42)
        settings_many = DynamicColoredNoiseSettings(n_poles=20, seed=42)

        transformer_few = DynamicColoredNoiseTransformer(settings_few)
        transformer_many = DynamicColoredNoiseTransformer(settings_many)

        n_samples = 5000
        beta_input = np.full((n_samples, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=1000.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        out_few = transformer_few(msg_in)
        out_many = transformer_many(msg_in)

        # Both should produce valid output
        assert np.all(np.isfinite(out_few.data))
        assert np.all(np.isfinite(out_many.data))

    def test_state_preservation(self):
        """Test that filter state is preserved across calls."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, seed=42))

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # First call
        beta1 = np.full((10, 1), 1.0)
        msg1 = AxisArray(beta1, dims=["time", "ch"], axes={"time": time_axis})
        transformer(msg1)

        # Check state exists
        assert len(transformer.state.filter_states) == 1
        assert transformer.state.filter_states[0].delay_line is not None

        # Delay line should not be all zeros after processing
        assert not np.allclose(transformer.state.filter_states[0].delay_line, 0.0)

    def test_smoothing_tau_zero_instant_change(self):
        """Test that smoothing_tau=0 gives instantaneous coefficient changes."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(n_poles=5, smoothing_tau=0.0, seed=42))

        # Start with beta=1
        time_axis = AxisArray.TimeAxis(fs=1000.0, offset=0.0)
        beta1 = np.full((100, 1), 1.0)
        msg1 = AxisArray(beta1, dims=["time", "ch"], axes={"time": time_axis})
        transformer(msg1)

        # Get coefficients for beta=1
        from ezmsg.simbiophys import compute_kasdin_coefficients

        expected_coeffs = compute_kasdin_coefficients(1.0, 5)

        # With tau=0, coefficients should exactly match target
        assert np.allclose(transformer.state.filter_states[0].coeffs, expected_coeffs)

        # Now change to beta=2
        beta2 = np.full((100, 1), 2.0)
        msg2 = AxisArray(beta2, dims=["time", "ch"], axes={"time": time_axis})
        transformer(msg2)

        expected_coeffs_2 = compute_kasdin_coefficients(2.0, 5)
        # Should immediately be at new target
        assert np.allclose(transformer.state.filter_states[0].coeffs, expected_coeffs_2)

    def test_smoothing_tau_time_constant_behavior(self):
        """Test that smoothing_tau controls the rate of coefficient change."""
        tau = 0.01  # 10ms time constant
        output_fs = 10000.0  # 10 kHz

        transformer = DynamicColoredNoiseTransformer(
            DynamicColoredNoiseSettings(output_fs=output_fs, n_poles=5, smoothing_tau=tau, initial_beta=0.0, seed=42)
        )

        # Process with beta=1 for exactly tau seconds (10ms = 100 samples at 10kHz)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)  # Input at 100Hz
        n_input = 1  # 1 input sample = 10ms at 100Hz
        beta_input = np.full((n_input, 1), 1.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        transformer(msg_in)

        # After tau seconds, should be ~63% of the way from initial to target
        from ezmsg.simbiophys import compute_kasdin_coefficients

        initial_coeffs = compute_kasdin_coefficients(0.0, 5)
        target_coeffs = compute_kasdin_coefficients(1.0, 5)

        # Expected: initial + (target - initial) * (1 - exp(-1)) ≈ initial + 0.632 * (target - initial)
        expected_progress = 1.0 - np.exp(-1.0)  # ≈ 0.632
        expected_coeffs = initial_coeffs + expected_progress * (target_coeffs - initial_coeffs)

        # Allow some tolerance due to discrete-time approximation
        actual_coeffs = transformer.state.filter_states[0].coeffs
        assert np.allclose(actual_coeffs, expected_coeffs, rtol=0.05)


class TestDynamicColoredNoiseResampling:
    """Tests for output_fs resampling functionality."""

    def test_upsampling_integer_ratio(self):
        """Test upsampling with integer ratio (e.g., 100 Hz -> 1000 Hz)."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(output_fs=1000.0, n_poles=5, seed=42))

        # Input at 100 Hz
        n_input = 10
        beta_input = np.full((n_input, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # Should produce 10x as many samples
        expected_samples = n_input * 10  # 1000/100 = 10
        assert msg_out.data.shape == (expected_samples, 1)

        # Output time axis should have correct gain
        assert msg_out.axes["time"].gain == pytest.approx(1.0 / 1000.0)

    def test_upsampling_non_integer_ratio(self):
        """Test upsampling with non-integer ratio."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(output_fs=300.0, n_poles=5, seed=42))

        # Input at 100 Hz, output at 300 Hz -> 3x ratio
        n_input = 10
        beta_input = np.full((n_input, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # 10 input samples * 3 = 30 output samples
        assert msg_out.data.shape == (30, 1)

    def test_downsampling_integer_ratio(self):
        """Test downsampling with integer ratio (e.g., 1000 Hz -> 100 Hz)."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(output_fs=100.0, n_poles=5, seed=42))

        # Input at 1000 Hz
        n_input = 100
        beta_input = np.full((n_input, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=1000.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # Should produce 1/10 as many samples
        expected_samples = n_input // 10  # 100/1000 = 0.1
        assert msg_out.data.shape == (expected_samples, 1)

    def test_fractional_samples_accumulation(self):
        """Test that fractional samples accumulate correctly across chunks."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(output_fs=30.0, n_poles=5, seed=42))

        # Input at 100 Hz, output at 30 Hz
        # Ratio is 0.3 samples per input sample
        # 10 input samples -> 3 output samples
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        total_output_samples = 0
        for _ in range(10):
            beta_input = np.full((10, 1), 1.0)
            msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})
            msg_out = transformer(msg_in)
            total_output_samples += msg_out.data.shape[0]

        # 100 input samples at 100 Hz = 1 second
        # At 30 Hz output, should be 30 samples total
        assert total_output_samples == 30

    def test_output_fs_none_matches_input(self):
        """Test that output_fs=None produces same rate as input."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(output_fs=None, n_poles=5, seed=42))

        n_input = 50
        beta_input = np.full((n_input, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # Same number of samples
        assert msg_out.data.shape == (n_input, 1)
        # Same time axis gain
        assert msg_out.axes["time"].gain == pytest.approx(time_axis.gain)

    def test_upsampling_continuity(self):
        """Test that upsampled output is continuous across chunks."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(output_fs=1000.0, n_poles=5, seed=42))

        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # Process multiple chunks
        outputs = []
        for _ in range(5):
            beta_input = np.full((10, 1), 1.0)
            msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})
            outputs.append(transformer(msg_in).data)

        # Concatenate and check continuity
        concat_output = np.concatenate(outputs, axis=0)
        diffs = np.abs(np.diff(concat_output[:, 0]))

        # Boundary indices (every 100 samples, since 10 input * 10x upsample = 100)
        boundary_indices = [99, 199, 299, 399]
        max_internal_diff = np.max(np.delete(diffs, boundary_indices))

        for idx in boundary_indices:
            assert diffs[idx] < 5 * max_internal_diff, f"Discontinuity at index {idx}"

    def test_upsampling_spectral_properties(self):
        """Test that upsampled pink noise maintains spectral properties."""
        transformer = DynamicColoredNoiseTransformer(
            DynamicColoredNoiseSettings(output_fs=10000.0, n_poles=10, initial_beta=1.0, seed=42)
        )

        # Input at 100 Hz, output at 10 kHz (100x upsampling)
        n_input = 100
        beta_input = np.full((n_input, 1), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # Compute power spectrum
        fft = np.fft.rfft(msg_out.data[:, 0])
        psd = np.abs(fft) ** 2

        # For pink noise, low frequencies should have more power
        n_bins = len(psd)
        low_freq_power = np.mean(psd[10 : n_bins // 4])
        high_freq_power = np.mean(psd[3 * n_bins // 4 :])

        assert low_freq_power > high_freq_power, "Upsampled pink noise should maintain 1/f spectrum"

    def test_multi_channel_upsampling(self):
        """Test upsampling with multiple channels."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(output_fs=500.0, n_poles=5, seed=42))

        n_input = 20
        n_channels = 3
        beta_input = np.full((n_input, n_channels), 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # 5x upsampling
        expected_samples = n_input * 5
        assert msg_out.data.shape == (expected_samples, n_channels)

        # Channels should still be independent
        for i in range(n_channels - 1):
            assert not np.array_equal(msg_out.data[:, i], msg_out.data[:, i + 1])

    def test_varying_beta_with_upsampling(self):
        """Test varying beta with upsampling."""
        transformer = DynamicColoredNoiseTransformer(
            DynamicColoredNoiseSettings(output_fs=500.0, n_poles=5, smoothing_tau=0.02, seed=42)
        )

        # Beta changes from 0.5 to 1.5 over 10 input samples
        beta_input = np.linspace(0.5, 1.5, 10)[:, np.newaxis]
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # Should have 5x upsampled output
        assert msg_out.data.shape == (50, 1)
        assert np.all(np.isfinite(msg_out.data))

    def test_1d_upsampling(self):
        """Test that 1D input with upsampling produces 1D output."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(output_fs=500.0, n_poles=5, seed=42))

        beta_input = np.full(20, 1.0)
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)
        msg_in = AxisArray(beta_input, dims=["time"], axes={"time": time_axis})

        msg_out = transformer(msg_in)

        # Should be 1D with 5x samples
        assert msg_out.data.shape == (100,)

    def test_empty_output_accumulation(self):
        """Test that when downsampling, small chunks accumulate properly."""
        transformer = DynamicColoredNoiseTransformer(DynamicColoredNoiseSettings(output_fs=10.0, n_poles=5, seed=42))

        # Input at 100 Hz, output at 10 Hz
        # 1 input sample = 0.1 output samples
        time_axis = AxisArray.TimeAxis(fs=100.0, offset=0.0)

        # First 5 samples: 0.5 output samples (rounds to 0)
        beta_input = np.full((5, 1), 1.0)
        msg_in = AxisArray(beta_input, dims=["time", "ch"], axes={"time": time_axis})
        msg_out = transformer(msg_in)

        # May produce 0 samples
        assert msg_out.data.shape[0] <= 1

        # But after 10 total input samples, should have 1 output sample
        beta_input2 = np.full((5, 1), 1.0)
        msg_in2 = AxisArray(beta_input2, dims=["time", "ch"], axes={"time": time_axis})
        msg_out2 = transformer(msg_in2)

        total_out = msg_out.data.shape[0] + msg_out2.data.shape[0]
        assert total_out == 1
