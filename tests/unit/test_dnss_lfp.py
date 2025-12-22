"""Unit tests for ezmsg.simbiophys.dnss.lfp module."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray
from scipy import signal

from ezmsg.simbiophys.dnss.lfp import (
    DEFAULT_FS,
    LFP_FREQS,
    LFP_PERIOD,
    OTHER_PERIOD,
    _generate_other_lfp_period,
    _generate_spike_lfp_period,
    lfp_generator,
)

# Compute sample counts from period in seconds
LFP_PERIOD_SAMPLES = int(LFP_PERIOD * DEFAULT_FS)
OTHER_PERIOD_SAMPLES = int(OTHER_PERIOD * DEFAULT_FS)


class TestGenerateSpikeLfpPeriod:
    """Tests for _generate_spike_lfp_period helper."""

    def test_output_shape(self):
        """Output has correct shape."""
        lfp = _generate_spike_lfp_period()
        assert lfp.shape == (LFP_PERIOD_SAMPLES,)

    def test_output_dtype(self):
        """Output is float64."""
        lfp = _generate_spike_lfp_period()
        assert lfp.dtype == np.float64

    def test_amplitude_range_hdmi(self):
        """HDMI mode has expected amplitude range."""
        lfp = _generate_spike_lfp_period(mode="hdmi")
        # Sum of 3 sinusoids with amplitude 894.4 each
        # Max when all aligned: ~2683, but phase shifts prevent perfect alignment
        assert np.abs(lfp).max() < 2700
        assert np.abs(lfp).max() > 1500  # At least partial constructive interference

    def test_different_modes(self):
        """Different modes produce different outputs."""
        hdmi = _generate_spike_lfp_period(mode="hdmi")
        pedestal = _generate_spike_lfp_period(mode="pedestal_norm")
        assert not np.allclose(hdmi, pedestal)


class TestGenerateOtherLfpPeriod:
    """Tests for _generate_other_lfp_period helper."""

    def test_output_shape(self):
        """Output has correct shape."""
        lfp = _generate_other_lfp_period()
        assert lfp.shape == (OTHER_PERIOD_SAMPLES,)

    def test_output_dtype(self):
        """Output is float64."""
        lfp = _generate_other_lfp_period()
        assert lfp.dtype == np.float64

    def test_amplitude_range_hdmi(self):
        """HDMI mode has amplitude of 6000."""
        lfp = _generate_other_lfp_period(mode="hdmi")
        assert np.abs(lfp).max() == pytest.approx(6000, rel=0.01)

    def test_amplitude_range_pedestal(self):
        """Pedestal mode has amplitude of 1000."""
        lfp = _generate_other_lfp_period(mode="pedestal")
        assert np.abs(lfp).max() == pytest.approx(1000, rel=0.01)

    def test_held_value_region(self):
        """Region from 29279 to 30000 holds constant value (at 30kHz)."""
        lfp = _generate_other_lfp_period(fs=DEFAULT_FS)
        # Convert time boundaries to samples
        first_stop = int(29_279 / 30_000 * DEFAULT_FS)
        second_start = int(1.0 * DEFAULT_FS)
        held_region = lfp[first_stop:second_start]
        assert np.all(held_region == held_region[0])


class TestLfpGeneratorBasics:
    """Basic functionality tests for lfp_generator."""

    def test_generator_priming(self):
        """Generator must be primed with next() before send()."""
        gen = lfp_generator()
        result = next(gen)
        assert result is None

    def test_empty_on_zero_samples(self):
        """Sending 0 or None returns empty array."""
        gen = lfp_generator()
        next(gen)

        result = gen.send(0)
        assert result.shape == (0,)

        result = gen.send(None)
        assert result.shape == (0,)

    def test_returns_1d_array(self):
        """Generator yields 1D arrays."""
        gen = lfp_generator()
        next(gen)
        result = gen.send(1000)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (1000,)

    def test_correct_length(self):
        """Output length matches requested samples."""
        gen = lfp_generator()
        next(gen)

        for n_samples in [1, 100, 1000, 10000, 30000, 45000]:
            result = gen.send(n_samples)
            assert len(result) == n_samples


class TestLfpGeneratorPeriodicity:
    """Tests for LFP pattern periodicity."""

    def test_spike_mode_period(self):
        """Spike mode repeats every LFP_PERIOD_SAMPLES samples."""
        gen = lfp_generator(pattern="spike", fs=DEFAULT_FS)
        next(gen)

        period1 = gen.send(LFP_PERIOD_SAMPLES)
        period2 = gen.send(LFP_PERIOD_SAMPLES)

        np.testing.assert_allclose(period1, period2)

    def test_other_mode_period(self):
        """Other mode repeats every OTHER_PERIOD_SAMPLES samples."""
        gen = lfp_generator(pattern="other", fs=DEFAULT_FS)
        next(gen)

        period1 = gen.send(OTHER_PERIOD_SAMPLES)
        period2 = gen.send(OTHER_PERIOD_SAMPLES)

        np.testing.assert_allclose(period1, period2)

    def test_chunked_matches_continuous(self):
        """Chunked access produces same result as continuous."""
        # Continuous
        gen1 = lfp_generator(fs=DEFAULT_FS)
        next(gen1)
        continuous = gen1.send(30000)

        # Chunked with various sizes
        gen2 = lfp_generator(fs=DEFAULT_FS)
        next(gen2)
        chunks = []
        chunk_sizes = [7500, 10000, 5000, 7500]
        for size in chunk_sizes:
            chunks.append(gen2.send(size))
        chunked = np.concatenate(chunks)

        np.testing.assert_allclose(continuous, chunked)

    def test_window_spanning_period_boundary(self):
        """Window spanning period boundary is handled correctly."""
        gen = lfp_generator(pattern="spike", fs=DEFAULT_FS)
        next(gen)

        # Skip to near end of period
        _ = gen.send(LFP_PERIOD_SAMPLES - 500)

        # Get window spanning boundary
        window = gen.send(1000)
        assert len(window) == 1000

        # Verify it wraps correctly
        gen2 = lfp_generator(pattern="spike", fs=DEFAULT_FS)
        next(gen2)
        full_period = gen2.send(LFP_PERIOD_SAMPLES)

        # First 500 samples should match end of period
        np.testing.assert_allclose(window[:500], full_period[-500:])
        # Last 500 samples should match start of next period
        np.testing.assert_allclose(window[500:], full_period[:500])


class TestLfpGeneratorSpectral:
    """Spectral analysis tests for LFP generator."""

    def test_spike_mode_frequencies(self):
        """Spike mode contains expected frequency components."""
        gen = lfp_generator(pattern="spike", mode="hdmi", fs=DEFAULT_FS)
        next(gen)
        lfp = gen.send(int(DEFAULT_FS * 10))  # 10 seconds for good frequency resolution

        # Compute PSD
        freqs, psd = signal.welch(lfp, fs=DEFAULT_FS, nperseg=int(DEFAULT_FS))

        # Find peaks
        peak_indices = signal.find_peaks(psd, height=np.max(psd) * 0.01)[0]
        peak_freqs = freqs[peak_indices]

        # Verify peaks at expected frequencies
        for expected_freq in LFP_FREQS:
            closest = peak_freqs[np.argmin(np.abs(peak_freqs - expected_freq))]
            assert abs(closest - expected_freq) < 0.5, f"Expected {expected_freq} Hz, found {closest} Hz"

    def test_other_mode_frequency_segments(self):
        """Other mode has different frequencies in different segments."""
        gen = lfp_generator(pattern="other", mode="hdmi", fs=DEFAULT_FS)
        next(gen)
        lfp = gen.send(OTHER_PERIOD_SAMPLES)

        # Analyze first segment (1 Hz) - ends at ~0.976 seconds
        first_stop = int(29_279 / 30_000 * DEFAULT_FS)
        segment1 = lfp[:first_stop]
        freqs, psd = signal.welch(segment1, fs=DEFAULT_FS, nperseg=min(len(segment1), int(DEFAULT_FS)))
        peak_idx = np.argmax(psd[1:]) + 1  # Skip DC
        assert freqs[peak_idx] == pytest.approx(1.0, abs=0.5)

        # Analyze 10 Hz segment (1.0 to 1.5 seconds)
        seg2_start = int(1.0 * DEFAULT_FS)
        seg2_stop = int(1.5 * DEFAULT_FS)
        segment2 = lfp[seg2_start:seg2_stop]
        freqs, psd = signal.welch(segment2, fs=DEFAULT_FS, nperseg=min(len(segment2), int(DEFAULT_FS)))
        peak_idx = np.argmax(psd[1:]) + 1
        assert freqs[peak_idx] == pytest.approx(10.0, abs=1.0)


class TestDNSSLFPTransformer:
    """Tests for DNSSLFPTransformer."""

    def _create_counter_input(self, n_time: int, offset: float = 0.0, counter_start: int = 0) -> "AxisArray":
        """Create a counter AxisArray input."""
        data = np.arange(counter_start, counter_start + n_time)
        return AxisArray(
            data=data,
            dims=["time"],
            axes={"time": AxisArray.TimeAxis(fs=DEFAULT_FS, offset=offset)},
        )

    def test_transformer_sync_call(self):
        """Test synchronous transformer via __call__."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPSettings, DNSSLFPTransformer

        transformer = DNSSLFPTransformer(DNSSLFPSettings(n_ch=4))
        input_msg = self._create_counter_input(n_time=600)
        result = transformer(input_msg)

        assert result is not None
        assert result.data.shape[1] == 4  # n_ch
        assert result.data.shape[0] == 600  # n_time
        assert "time" in result.dims
        assert "ch" in result.dims

    def test_transformer_output_shape(self):
        """Transformer output has correct shape (time, ch)."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPSettings, DNSSLFPTransformer

        n_ch = 16
        transformer = DNSSLFPTransformer(DNSSLFPSettings(n_ch=n_ch))
        input_msg = self._create_counter_input(n_time=100)

        result = transformer(input_msg)
        assert result.data.ndim == 2
        assert result.data.shape[1] == n_ch

    def test_transformer_channels_identical(self):
        """All channels have identical LFP values."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPSettings, DNSSLFPTransformer

        transformer = DNSSLFPTransformer(DNSSLFPSettings(n_ch=8))
        input_msg = self._create_counter_input(n_time=600)
        result = transformer(input_msg)

        # All columns should be identical
        for ch in range(1, result.data.shape[1]):
            np.testing.assert_allclose(result.data[:, 0], result.data[:, ch])

    def test_transformer_continuity(self):
        """Multiple calls produce continuous data."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPSettings, DNSSLFPTransformer

        transformer = DNSSLFPTransformer(DNSSLFPSettings(n_ch=4))

        # Get multiple chunks
        results = []
        counter = 0
        for i in range(5):
            input_msg = self._create_counter_input(n_time=600, offset=counter / DEFAULT_FS, counter_start=counter)
            results.append(transformer(input_msg))
            counter += 600

        # Concatenate first channel
        combined = np.concatenate([r.data[:, 0] for r in results])

        # Compare with generator output
        gen = lfp_generator(fs=DEFAULT_FS)
        next(gen)
        expected = gen.send(len(combined))

        np.testing.assert_allclose(combined, expected)

    def test_transformer_different_patterns(self):
        """Transformer works with different patterns."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPSettings, DNSSLFPTransformer

        spike_transformer = DNSSLFPTransformer(DNSSLFPSettings(n_ch=4, pattern="spike"))
        other_transformer = DNSSLFPTransformer(DNSSLFPSettings(n_ch=4, pattern="other"))

        input_msg = self._create_counter_input(n_time=600)

        spike_result = spike_transformer(input_msg)
        other_result = other_transformer(input_msg)

        # Both should produce valid output
        assert spike_result.data.shape[0] > 0
        assert other_result.data.shape[0] > 0

        # But with different values (different amplitude ranges)
        assert not np.allclose(spike_result.data, other_result.data)
