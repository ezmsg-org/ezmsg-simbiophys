"""Unit tests for ezmsg.simbiophys.dnss.lfp module."""

import numpy as np
import pytest
from scipy import signal

from ezmsg.simbiophys.dnss.lfp import (
    FS,
    LFP_FREQS,
    LFP_PERIOD,
    OTHER_PERIOD,
    _generate_other_lfp_period,
    _generate_spike_lfp_period,
    lfp_generator,
)


class TestGenerateSpikeLfpPeriod:
    """Tests for _generate_spike_lfp_period helper."""

    def test_output_shape(self):
        """Output has correct shape."""
        lfp = _generate_spike_lfp_period()
        assert lfp.shape == (LFP_PERIOD,)

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
        assert lfp.shape == (OTHER_PERIOD,)

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
        """Region from 29279 to 30000 holds constant value."""
        lfp = _generate_other_lfp_period()
        held_region = lfp[29279:30000]
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
        """Spike mode repeats every LFP_PERIOD samples."""
        gen = lfp_generator(pattern="spike")
        next(gen)

        period1 = gen.send(LFP_PERIOD)
        period2 = gen.send(LFP_PERIOD)

        np.testing.assert_allclose(period1, period2)

    def test_other_mode_period(self):
        """Other mode repeats every OTHER_PERIOD samples."""
        gen = lfp_generator(pattern="other")
        next(gen)

        period1 = gen.send(OTHER_PERIOD)
        period2 = gen.send(OTHER_PERIOD)

        np.testing.assert_allclose(period1, period2)

    def test_chunked_matches_continuous(self):
        """Chunked access produces same result as continuous."""
        # Continuous
        gen1 = lfp_generator()
        next(gen1)
        continuous = gen1.send(30000)

        # Chunked with various sizes
        gen2 = lfp_generator()
        next(gen2)
        chunks = []
        chunk_sizes = [7500, 10000, 5000, 7500]
        for size in chunk_sizes:
            chunks.append(gen2.send(size))
        chunked = np.concatenate(chunks)

        np.testing.assert_allclose(continuous, chunked)

    def test_window_spanning_period_boundary(self):
        """Window spanning period boundary is handled correctly."""
        gen = lfp_generator(pattern="spike")
        next(gen)

        # Skip to near end of period
        _ = gen.send(LFP_PERIOD - 500)

        # Get window spanning boundary
        window = gen.send(1000)
        assert len(window) == 1000

        # Verify it wraps correctly
        gen2 = lfp_generator(pattern="spike")
        next(gen2)
        full_period = gen2.send(LFP_PERIOD)

        # First 500 samples should match end of period
        np.testing.assert_allclose(window[:500], full_period[-500:])
        # Last 500 samples should match start of next period
        np.testing.assert_allclose(window[500:], full_period[:500])


class TestLfpGeneratorSpectral:
    """Spectral analysis tests for LFP generator."""

    def test_spike_mode_frequencies(self):
        """Spike mode contains expected frequency components."""
        gen = lfp_generator(pattern="spike", mode="hdmi")
        next(gen)
        lfp = gen.send(FS * 10)  # 10 seconds for good frequency resolution

        # Compute PSD
        freqs, psd = signal.welch(lfp, fs=FS, nperseg=FS)

        # Find peaks
        peak_indices = signal.find_peaks(psd, height=np.max(psd) * 0.01)[0]
        peak_freqs = freqs[peak_indices]

        # Verify peaks at expected frequencies
        for expected_freq in LFP_FREQS:
            closest = peak_freqs[np.argmin(np.abs(peak_freqs - expected_freq))]
            assert abs(closest - expected_freq) < 0.5, f"Expected {expected_freq} Hz, found {closest} Hz"

    def test_other_mode_frequency_segments(self):
        """Other mode has different frequencies in different segments."""
        gen = lfp_generator(pattern="other", mode="hdmi")
        next(gen)
        lfp = gen.send(OTHER_PERIOD)

        # Analyze first segment (1 Hz)
        segment1 = lfp[:29279]
        freqs, psd = signal.welch(segment1, fs=FS, nperseg=min(len(segment1), FS))
        peak_idx = np.argmax(psd[1:]) + 1  # Skip DC
        assert freqs[peak_idx] == pytest.approx(1.0, abs=0.5)

        # Analyze 10 Hz segment
        segment2 = lfp[30000:45000]
        freqs, psd = signal.welch(segment2, fs=FS, nperseg=min(len(segment2), FS))
        peak_idx = np.argmax(psd[1:]) + 1
        assert freqs[peak_idx] == pytest.approx(10.0, abs=1.0)


class TestDNSSLFPProducer:
    """Tests for DNSSLFPProducer (CompositeProducer using Counter + Transformer)."""

    def test_producer_sync_call(self):
        """Test synchronous producer via __call__."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPProducer, DNSSLFPSettings

        producer = DNSSLFPProducer(DNSSLFPSettings(dispatch_rate=None, n_ch=4))
        result = producer()

        assert result is not None
        assert result.data.shape[1] == 4  # n_ch
        assert result.data.shape[0] == 600  # default n_time
        assert "time" in result.dims
        assert "ch" in result.dims

    @pytest.mark.asyncio
    async def test_producer_async_call(self):
        """Test asynchronous producer via __acall__."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPProducer, DNSSLFPSettings

        producer = DNSSLFPProducer(DNSSLFPSettings(dispatch_rate=None, n_ch=8))
        result = await producer.__acall__()

        assert result is not None
        assert result.data.shape[1] == 8
        assert result.data.dtype == np.float64

    def test_producer_output_shape(self):
        """Producer output has correct shape (time, ch)."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPProducer, DNSSLFPSettings

        n_ch = 16
        producer = DNSSLFPProducer(DNSSLFPSettings(dispatch_rate=None, n_ch=n_ch))

        result = producer()
        assert result.data.ndim == 2
        assert result.data.shape[1] == n_ch

    def test_producer_channels_identical(self):
        """All channels have identical LFP values."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPProducer, DNSSLFPSettings

        producer = DNSSLFPProducer(DNSSLFPSettings(dispatch_rate=None, n_ch=8))
        result = producer()

        # All columns should be identical
        for ch in range(1, result.data.shape[1]):
            np.testing.assert_allclose(result.data[:, 0], result.data[:, ch])

    def test_producer_continuity(self):
        """Multiple calls produce continuous data."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPProducer, DNSSLFPSettings

        producer = DNSSLFPProducer(DNSSLFPSettings(dispatch_rate=None, n_ch=4))

        # Get multiple chunks
        results = [producer() for _ in range(5)]

        # Concatenate first channel
        combined = np.concatenate([r.data[:, 0] for r in results])

        # Compare with generator output
        gen = lfp_generator()
        next(gen)
        expected = gen.send(len(combined))

        np.testing.assert_allclose(combined, expected)

    def test_producer_time_axis(self):
        """Time axis offset updates correctly."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPProducer, DNSSLFPSettings

        producer = DNSSLFPProducer(DNSSLFPSettings(dispatch_rate=None, n_ch=4))

        result1 = producer()
        t0_1 = result1.axes["time"].offset
        n_samples_1 = result1.data.shape[0]

        result2 = producer()
        t0_2 = result2.axes["time"].offset

        # Second chunk should start where first ended
        expected_t0_2 = t0_1 + n_samples_1 / FS
        assert t0_2 == pytest.approx(expected_t0_2, rel=1e-6)

    def test_producer_different_patterns(self):
        """Producer works with different patterns."""
        from ezmsg.simbiophys.dnss.lfp import DNSSLFPProducer, DNSSLFPSettings

        spike_producer = DNSSLFPProducer(DNSSLFPSettings(dispatch_rate=None, n_ch=4, pattern="spike"))
        other_producer = DNSSLFPProducer(DNSSLFPSettings(dispatch_rate=None, n_ch=4, pattern="other"))

        spike_result = spike_producer()
        other_result = other_producer()

        # Both should produce valid output
        assert spike_result.data.shape[0] > 0
        assert other_result.data.shape[0] > 0

        # But with different values (different amplitude ranges)
        assert not np.allclose(spike_result.data, other_result.data)
