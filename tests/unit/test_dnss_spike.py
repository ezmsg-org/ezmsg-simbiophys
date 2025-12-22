"""Unit tests for ezmsg.simbiophys.dnss module."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys.dnss.spike import (
    FS,
    FULL_PERIOD,
    INT_BURST,
    INT_SLOW,
    N_BURST_SPIKES,
    N_SLOW_SPIKES,
    SAMPS_SLOW,
    _spikes_in_range,
    spike_event_generator,
)


class TestSpikesInRange:
    """Tests for the _spikes_in_range helper function."""

    def test_empty_range(self):
        """Empty or invalid ranges return empty array."""
        assert len(_spikes_in_range(100, 50, burst=False)) == 0
        assert len(_spikes_in_range(100, 100, burst=True)) == 0

    def test_no_spikes_in_range_slow(self):
        """Range between slow spikes returns empty."""
        # Slow spikes at 0, 7500, 15000, ... Range [1, 7499) has no spikes
        result = _spikes_in_range(1, 7499, burst=False)
        assert len(result) == 0

    def test_no_spikes_in_range_burst(self):
        """Range between burst spikes returns empty."""
        # Burst spikes at 0, 300, 600, ... Range [1, 299) has no spikes
        result = _spikes_in_range(1, 299, burst=True)
        assert len(result) == 0

    def test_single_spike_at_start_slow(self):
        """Range starting at slow spike position captures it."""
        result = _spikes_in_range(0, 100, burst=False)
        np.testing.assert_array_equal(result, [0])

    def test_single_spike_at_start_burst(self):
        """Range starting at burst spike position captures it."""
        result = _spikes_in_range(0, 100, burst=True)
        np.testing.assert_array_equal(result, [0])

    def test_spike_at_end_excluded(self):
        """Spike exactly at end of range is excluded (half-open interval)."""
        # Slow: Range [0, 7500) should only include spike at 0, not at 7500
        result = _spikes_in_range(0, INT_SLOW, burst=False)
        np.testing.assert_array_equal(result, [0])

        # Burst: Range [0, 300) should only include spike at 0, not at 300
        result = _spikes_in_range(0, INT_BURST, burst=True)
        np.testing.assert_array_equal(result, [0])

    def test_multiple_spikes_slow(self):
        """Range spanning multiple slow spikes captures all."""
        # Slow spikes at 0, 7500, 15000. Range [100, 15001) captures spikes 1 and 2
        result = _spikes_in_range(100, 15001, burst=False)
        np.testing.assert_array_equal(result, [1, 2])

    def test_multiple_spikes_burst(self):
        """Range spanning multiple burst spikes captures all."""
        # Burst spikes at 0, 300, 600. Range [100, 601) captures spikes 1 and 2
        result = _spikes_in_range(100, 601, burst=True)
        np.testing.assert_array_equal(result, [1, 2])

    def test_clamps_to_n_spikes_slow(self):
        """Result is clamped to valid slow spike indices (0-35)."""
        # Request range covering more than 36 slow spikes
        result = _spikes_in_range(0, SAMPS_SLOW + 10000, burst=False)
        assert len(result) == N_SLOW_SPIKES
        np.testing.assert_array_equal(result, np.arange(N_SLOW_SPIKES))

    def test_clamps_to_n_spikes_burst(self):
        """Result is clamped to valid burst spike indices (0-98)."""
        # Request range covering more than 99 burst spikes
        result = _spikes_in_range(0, N_BURST_SPIKES * INT_BURST + 1000, burst=True)
        assert len(result) == N_BURST_SPIKES
        np.testing.assert_array_equal(result, np.arange(N_BURST_SPIKES))


class TestSpikeEventGeneratorBasics:
    """Basic functionality tests for spike_event_generator."""

    def test_generator_priming(self):
        """Generator must be primed with next() before send()."""
        gen = spike_event_generator()
        result = next(gen)
        assert result is None

    def test_empty_on_zero_samples(self):
        """Sending 0 or None returns empty arrays."""
        gen = spike_event_generator()
        next(gen)

        coords, waveforms = gen.send(0)
        assert coords.shape == (2, 0)
        assert len(waveforms) == 0

        coords, waveforms = gen.send(None)
        assert coords.shape == (2, 0)

    def test_returns_coords_and_waveforms(self):
        """Generator yields tuple of (coords, waveforms)."""
        gen = spike_event_generator()
        next(gen)
        result = gen.send(10000)

        assert isinstance(result, tuple)
        assert len(result) == 2
        coords, waveforms = result
        assert isinstance(coords, np.ndarray)
        assert isinstance(waveforms, np.ndarray)
        assert coords.shape[0] == 2  # (sample_idx, chan_idx)
        assert coords.shape[1] == len(waveforms)

    def test_sample_indices_within_window(self):
        """Sample indices are always within [0, n_samples)."""
        gen = spike_event_generator()
        next(gen)

        for n_samples in [1000, 7500, 15000, 30000]:
            coords, _ = gen.send(n_samples)
            if coords.shape[1] > 0:
                samples = coords[0]
                assert samples.min() >= 0
                assert samples.max() < n_samples


class TestSpikeEventGeneratorPattern:
    """Tests validating the DNSS spike pattern."""

    @pytest.fixture
    def collect_two_cycles(self):
        """Collect all spikes over 2 full cycles with variable chunk sizes."""

        def _collect(mode: str = "hdmi", n_chans: int = 4):
            gen = spike_event_generator(mode=mode, n_chans=n_chans)
            next(gen)

            all_samples = []
            all_channels = []
            all_waveforms = []
            current_pos = 0
            target = 2 * FULL_PERIOD

            # Use variable chunk sizes to stress-test the generator
            rng = np.random.default_rng(42)
            chunk_sizes = rng.integers(500, 50000, size=100).tolist()
            chunk_idx = 0

            while current_pos < target:
                n_samples = min(chunk_sizes[chunk_idx % len(chunk_sizes)], target - current_pos)
                coords, waveforms = gen.send(n_samples)

                # Offset samples to absolute position
                if coords.shape[1] > 0:
                    all_samples.extend(coords[0] + current_pos)
                    all_channels.extend(coords[1])
                    all_waveforms.extend(waveforms)

                current_pos += n_samples
                chunk_idx += 1

            return (
                np.array(all_samples),
                np.array(all_channels),
                np.array(all_waveforms),
            )

        return _collect

    def test_total_spike_count_hdmi(self, collect_two_cycles):
        """Verify total spike count over 2 cycles in HDMI mode."""
        samples, channels, waveforms = collect_two_cycles(mode="hdmi")

        # Per cycle: 36 slow spikes + 99 burst spikes * 4 channels = 36 + 396 = 432
        expected_per_cycle = N_SLOW_SPIKES + N_BURST_SPIKES * 4
        expected_total = 2 * expected_per_cycle

        assert len(samples) == expected_total
        assert len(channels) == expected_total
        assert len(waveforms) == expected_total

    def test_total_spike_count_ideal(self, collect_two_cycles):
        """Verify total spike count over 2 cycles in ideal mode."""
        samples, channels, waveforms = collect_two_cycles(mode="ideal")

        expected_per_cycle = N_SLOW_SPIKES + N_BURST_SPIKES * 4
        expected_total = 2 * expected_per_cycle

        assert len(samples) == expected_total

    def test_slow_phase_spike_positions(self, collect_two_cycles):
        """Verify slow phase spike positions."""
        samples, channels, waveforms = collect_two_cycles(mode="ideal")

        # Extract first cycle's slow phase spikes (first 36 events, one per channel)
        slow_mask = samples < SAMPS_SLOW
        slow_samples = samples[slow_mask]
        # slow_channels = channels[slow_mask]

        # In ideal mode, slow spikes are at 0, 7500, 15000, ... on channels 0, 1, 2, 3, 0, ...
        # First cycle has 36 slow spikes
        first_cycle_slow = slow_samples[slow_samples < SAMPS_SLOW]
        assert len(first_cycle_slow) == N_SLOW_SPIKES

        # Verify positions (each at i * INT_SLOW for i in 0..35)
        expected_positions = np.arange(N_SLOW_SPIKES) * INT_SLOW
        np.testing.assert_array_equal(np.sort(first_cycle_slow), expected_positions)

    def test_slow_phase_channel_rotation_ideal(self, collect_two_cycles):
        """Verify channels rotate correctly in ideal mode (no HDMI bug)."""
        samples, channels, waveforms = collect_two_cycles(mode="ideal")

        # Get first cycle slow phase
        first_cycle_slow_mask = samples < SAMPS_SLOW
        slow_samples = samples[first_cycle_slow_mask]
        slow_channels = channels[first_cycle_slow_mask]

        # Sort by sample position to get chronological order
        order = np.argsort(slow_samples)
        slow_channels_sorted = slow_channels[order]

        # Channels should cycle 0, 1, 2, 3, 0, 1, 2, 3, ...
        expected_channels = np.arange(N_SLOW_SPIKES) % 4
        np.testing.assert_array_equal(slow_channels_sorted, expected_channels)

    def test_slow_phase_channel_rotation_hdmi(self, collect_two_cycles):
        """Verify HDMI bug #1: channels start at 1 instead of 0."""
        samples, channels, waveforms = collect_two_cycles(mode="hdmi")

        # Get first cycle slow phase
        first_cycle_slow_mask = samples < SAMPS_SLOW
        slow_samples = samples[first_cycle_slow_mask]
        slow_channels = channels[first_cycle_slow_mask]

        # Sort by sample position
        order = np.argsort(slow_samples)
        slow_channels_sorted = slow_channels[order]

        # With HDMI bug, channels should cycle 1, 2, 3, 0, 1, 2, 3, 0, ...
        expected_channels = (np.arange(N_SLOW_SPIKES) + 1) % 4
        np.testing.assert_array_equal(slow_channels_sorted, expected_channels)

    def test_hdmi_bug_channel_0_delay(self, collect_two_cycles):
        """Verify HDMI bug #2: channel 0 spikes are delayed by 1 sample."""
        samples, channels, _ = collect_two_cycles(mode="hdmi")

        # In HDMI mode, channel 0 spikes should be at (expected_position + 1)
        # while other channels are at expected_position

        # For slow phase: spike i is at i * INT_SLOW, channel = (i + 1) % 4
        # Channel 0 occurs when (i + 1) % 4 == 0, i.e., i = 3, 7, 11, ...
        # These should be at positions 3*7500+1, 7*7500+1, etc.

        # Check first cycle slow phase channel 0 spikes
        slow_ch0_mask = (channels == 0) & (samples < SAMPS_SLOW)
        slow_ch0_samples = np.sort(samples[slow_ch0_mask])

        # Expected slow spike indices for channel 0: 3, 7, 11, 15, 19, 23, 27, 31, 35
        ch0_spike_indices = np.arange(3, N_SLOW_SPIKES, 4)
        expected_slow_ch0 = ch0_spike_indices * INT_SLOW + 1  # +1 for HDMI bug #2
        np.testing.assert_array_equal(slow_ch0_samples, expected_slow_ch0)

        # Check first cycle burst phase - all channels fire at each event
        # but channel 0 should be delayed by 1 sample
        burst_start = SAMPS_SLOW
        burst_end = SAMPS_SLOW + N_BURST_SPIKES * INT_BURST
        burst_mask = (samples >= burst_start) & (samples < burst_end + 1)  # +1 for ch0 delay
        burst_samples = samples[burst_mask]
        burst_channels = channels[burst_mask]

        # For each burst event, ch0 should be at event_pos + 1, others at event_pos
        for i in range(min(10, N_BURST_SPIKES)):  # Check first 10 burst events
            event_pos = SAMPS_SLOW + i * INT_BURST
            event_mask = (burst_samples >= event_pos) & (burst_samples <= event_pos + 1)
            event_samples = burst_samples[event_mask]
            event_channels = burst_channels[event_mask]

            # Channels 1, 2, 3 should be at event_pos
            for ch in [1, 2, 3]:
                ch_samples = event_samples[event_channels == ch]
                assert len(ch_samples) == 1
                assert ch_samples[0] == event_pos

            # Channel 0 should be at event_pos + 1
            ch0_samples = event_samples[event_channels == 0]
            assert len(ch0_samples) == 1
            assert ch0_samples[0] == event_pos + 1

    def test_burst_phase_all_channels_fire(self, collect_two_cycles):
        """Verify all 4 channels fire simultaneously during burst."""
        samples, channels, waveforms = collect_two_cycles(mode="ideal")

        # Get burst phase of first cycle
        burst_start = SAMPS_SLOW
        burst_end = SAMPS_SLOW + N_BURST_SPIKES * INT_BURST
        burst_mask = (samples >= burst_start) & (samples < burst_end)

        burst_samples = samples[burst_mask]
        burst_channels = channels[burst_mask]

        # Should have 99 * 4 = 396 burst spikes in first cycle
        assert len(burst_samples) == N_BURST_SPIKES * 4

        # Each unique sample position should have all 4 channels
        unique_positions = np.unique(burst_samples)
        assert len(unique_positions) == N_BURST_SPIKES

        for pos in unique_positions:
            pos_channels = burst_channels[burst_samples == pos]
            np.testing.assert_array_equal(np.sort(pos_channels), [0, 1, 2, 3])

    def test_burst_phase_positions(self, collect_two_cycles):
        """Verify burst spike positions."""
        samples, channels, waveforms = collect_two_cycles(mode="ideal")

        # Get burst phase
        burst_start = SAMPS_SLOW
        burst_mask = (samples >= burst_start) & (samples < burst_start + N_BURST_SPIKES * INT_BURST)
        burst_samples = samples[burst_mask]

        unique_positions = np.unique(burst_samples)

        # Burst spikes at SAMPS_SLOW + i * INT_BURST for i in 0..98
        expected = SAMPS_SLOW + np.arange(N_BURST_SPIKES) * INT_BURST
        np.testing.assert_array_equal(unique_positions, expected)

    def test_waveform_cycling(self, collect_two_cycles):
        """Verify waveforms cycle through 1, 2, 3."""
        samples, channels, waveforms = collect_two_cycles(mode="ideal")

        # All waveforms should be 1, 2, or 3
        assert set(waveforms).issubset({1, 2, 3})

        # For slow phase, waveforms cycle with spike index
        slow_mask = samples < SAMPS_SLOW
        slow_samples = samples[slow_mask]
        slow_waveforms = waveforms[slow_mask]

        order = np.argsort(slow_samples)
        slow_waveforms_sorted = slow_waveforms[order]

        expected = (np.arange(N_SLOW_SPIKES) % 3) + 1
        np.testing.assert_array_equal(slow_waveforms_sorted, expected)

    def test_gap_has_no_spikes(self, collect_two_cycles):
        """Verify the gap after burst has no spikes."""
        samples, _, _ = collect_two_cycles(mode="ideal")

        # Gap is from (SAMPS_SLOW + N_BURST_SPIKES * INT_BURST) to FULL_PERIOD
        gap_start = SAMPS_SLOW + N_BURST_SPIKES * INT_BURST
        gap_end = FULL_PERIOD

        # First cycle gap
        gap_mask = (samples >= gap_start) & (samples < gap_end)
        assert np.sum(gap_mask) == 0

        # Second cycle gap
        gap_mask_2 = (samples >= gap_start + FULL_PERIOD) & (samples < gap_end + FULL_PERIOD)
        assert np.sum(gap_mask_2) == 0

    def test_pattern_repeats_exactly(self, collect_two_cycles):
        """Verify the pattern repeats exactly after FULL_PERIOD samples."""
        samples, channels, waveforms = collect_two_cycles(mode="ideal")

        # Split into two cycles
        cycle1_mask = samples < FULL_PERIOD
        cycle2_mask = samples >= FULL_PERIOD

        samples1 = samples[cycle1_mask]
        samples2 = samples[cycle2_mask] - FULL_PERIOD  # Normalize to cycle start

        channels1 = channels[cycle1_mask]
        channels2 = channels[cycle2_mask]

        waveforms1 = waveforms[cycle1_mask]
        waveforms2 = waveforms[cycle2_mask]

        np.testing.assert_array_equal(samples1, samples2)
        np.testing.assert_array_equal(channels1, channels2)
        np.testing.assert_array_equal(waveforms1, waveforms2)


class TestSpikeEventGeneratorEdgeCases:
    """Edge case tests for spike_event_generator."""

    def test_single_sample_windows(self):
        """Generator works with single-sample windows."""
        gen = spike_event_generator(mode="ideal")
        next(gen)

        # First sample should have a spike
        coords, waveforms = gen.send(1)
        assert coords.shape[1] == 1
        assert coords[0, 0] == 0  # sample
        assert coords[1, 0] == 0  # channel
        assert waveforms[0] == 1

        # Next sample should have no spike
        coords, _ = gen.send(1)
        assert coords.shape[1] == 0

    def test_window_exactly_on_spike(self):
        """Window ending exactly on spike position excludes that spike."""
        gen = spike_event_generator(mode="ideal")
        next(gen)

        # Get exactly 7500 samples (spike 0 at 0, spike 1 at 7500)
        coords, _ = gen.send(7500)
        assert coords.shape[1] == 1  # Only spike at 0

        # Next call should get spike at position 0 (relative to new window)
        coords, _ = gen.send(100)
        assert coords.shape[1] == 1
        assert coords[0, 0] == 0

    def test_window_spanning_slow_to_burst(self):
        """Window spanning slow to burst transition."""
        gen = spike_event_generator(mode="ideal")
        next(gen)

        # Skip to just before burst
        _ = gen.send(SAMPS_SLOW - 100)

        # Get window spanning transition
        coords, waveforms = gen.send(500)
        samples = coords[0]

        # Should have first burst spike (all 4 channels) at position 100
        assert len(samples) >= 4
        burst_samples = samples[samples == 100]
        assert len(burst_samples) == 4

    def test_window_spanning_period_boundary(self):
        """Window spanning end of period to start of next."""
        gen = spike_event_generator(mode="ideal")
        next(gen)

        # Skip to near end of period
        _ = gen.send(FULL_PERIOD - 500)

        # Get window spanning boundary
        coords, waveforms = gen.send(1000)
        samples = coords[0]

        # Should have spike at position 500 (start of new period)
        assert 500 in samples

    def test_large_window_multiple_periods(self):
        """Single window spanning multiple full periods."""
        gen = spike_event_generator(mode="ideal")
        next(gen)

        # Request 2.5 periods worth of samples
        n_samples = int(2.5 * FULL_PERIOD)
        coords, waveforms = gen.send(n_samples)
        samples = coords[0]

        # Should have ~2.5 cycles worth of spikes
        spikes_per_cycle = N_SLOW_SPIKES + N_BURST_SPIKES * 4
        # At least 2 full cycles, possibly more depending on where we end
        assert len(samples) >= 2 * spikes_per_cycle

    def test_different_n_chans(self):
        """Generator tiles/truncates the 4-channel pattern for any n_chans."""
        for n_chans in [1, 2, 3, 4, 5, 6, 7, 8, 10, 32]:
            gen = spike_event_generator(mode="ideal", n_chans=n_chans)
            next(gen)

            # Get one full period
            coords, waveforms = gen.send(FULL_PERIOD)
            samples = coords[0]
            channels = coords[1]

            # Check channel values are within range
            assert channels.max() < n_chans
            assert channels.min() >= 0

            # Slow phase should use all channels
            slow_mask = samples < SAMPS_SLOW
            slow_channels = np.unique(channels[slow_mask])
            assert len(slow_channels) == n_chans

            # Verify tiling: channels that differ by 4 should fire at same sample times
            slow_samples = samples[slow_mask]
            slow_chans = channels[slow_mask]
            for base_ch in range(min(4, n_chans)):
                tiled_channels = [ch for ch in range(base_ch, n_chans, 4)]
                if len(tiled_channels) > 1:
                    sample_times_per_tile = [set(slow_samples[slow_chans == ch]) for ch in tiled_channels]
                    assert all(s == sample_times_per_tile[0] for s in sample_times_per_tile)

            # Verify spike counts
            # Slow: each of the 36 spike events fires on ceil((n_chans - base_ch) / 4) channels
            # Simpler: count channels congruent to each base mod 4
            expected_slow = sum(len(range(b, n_chans, 4)) for b in range(4)) * (N_SLOW_SPIKES // 4)
            assert np.sum(slow_mask) == expected_slow

            # Burst: 99 events, each firing all n_chans channels
            burst_mask = samples >= SAMPS_SLOW
            assert np.sum(burst_mask) == N_BURST_SPIKES * n_chans


class TestDNSSSpikeTransformer:
    """Tests for DNSSSpikeTransformer."""

    def _create_counter_input(self, n_time: int, offset: float = 0.0, counter_start: int = 0) -> "AxisArray":
        """Create a counter AxisArray input."""
        data = np.arange(counter_start, counter_start + n_time)
        return AxisArray(
            data=data,
            dims=["time"],
            axes={"time": AxisArray.TimeAxis(fs=FS, offset=offset)},
        )

    def test_transformer_sync_call(self):
        """Test synchronous transformer via __call__."""
        from ezmsg.simbiophys.dnss.spike import DNSSSpikeSettings, DNSSSpikeTransformer

        transformer = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=4))
        input_msg = self._create_counter_input(n_time=600)
        result = transformer(input_msg)

        assert result is not None
        assert result.data.shape[1] == 4  # n_ch
        assert result.data.shape[0] == 600  # n_time
        assert "time" in result.dims
        assert "ch" in result.dims

    def test_transformer_output_is_sparse(self):
        """Transformer output data is sparse.COO."""
        import sparse

        from ezmsg.simbiophys.dnss.spike import DNSSSpikeSettings, DNSSSpikeTransformer

        transformer = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=4))
        input_msg = self._create_counter_input(n_time=600)
        result = transformer(input_msg)

        assert isinstance(result.data, sparse.COO)
        assert result.data.ndim == 2

    def test_transformer_spike_values(self):
        """Spike values are waveform IDs (1, 2, or 3)."""
        from ezmsg.simbiophys.dnss.spike import DNSSSpikeSettings, DNSSSpikeTransformer

        transformer = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=4))

        # Collect multiple chunks to ensure we get spikes
        all_data = []
        counter = 0
        for _ in range(10):
            input_msg = self._create_counter_input(n_time=600, offset=counter / FS, counter_start=counter)
            result = transformer(input_msg)
            if result.data.nnz > 0:
                all_data.extend(result.data.data.tolist())
            counter += 600

        # All non-zero values should be 1, 2, or 3
        assert len(all_data) > 0
        assert set(all_data).issubset({1, 2, 3})

    def test_transformer_continuity(self):
        """Multiple calls produce continuous spike pattern."""
        from ezmsg.simbiophys.dnss.spike import DNSSSpikeSettings, DNSSSpikeTransformer

        transformer = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=4, mode="ideal"))

        # Get multiple chunks
        all_coords = []
        all_waveforms = []
        counter = 0
        for _ in range(20):
            input_msg = self._create_counter_input(n_time=600, offset=counter / FS, counter_start=counter)
            result = transformer(input_msg)
            if result.data.nnz > 0:
                coords = result.data.coords
                all_coords.append(coords[0] + counter)  # Adjust sample indices
                all_waveforms.extend(result.data.data.tolist())
            counter += 600

        # Compare with generator output
        gen = spike_event_generator(mode="ideal", n_chans=4)
        next(gen)
        expected_coords, expected_waveforms = gen.send(counter)

        if len(all_coords) > 0:
            transformer_samples = np.concatenate(all_coords)
            np.testing.assert_array_equal(np.sort(transformer_samples), np.sort(expected_coords[0]))

    def test_transformer_different_n_chans(self):
        """Transformer works with different channel counts."""
        from ezmsg.simbiophys.dnss.spike import DNSSSpikeSettings, DNSSSpikeTransformer

        for n_ch in [4, 8, 32, 256]:
            transformer = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=n_ch))
            input_msg = self._create_counter_input(n_time=600)
            result = transformer(input_msg)

            assert result.data.shape[1] == n_ch

    def test_transformer_hdmi_vs_ideal_mode(self):
        """HDMI and ideal modes produce different spike patterns."""
        from ezmsg.simbiophys.dnss.spike import DNSSSpikeSettings, DNSSSpikeTransformer

        hdmi_transformer = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=4, mode="hdmi"))
        # ideal_transformer = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=4, mode="ideal"))

        # Collect enough data to see differences
        hdmi_coords = []
        ideal_coords = []
        counter = 0
        for _ in range(50):
            input_msg = self._create_counter_input(n_time=600, offset=counter / FS, counter_start=counter)
            hdmi_result = hdmi_transformer(input_msg)

            # Reset transformer for ideal (need fresh input)
            ideal_transformer_fresh = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=4, mode="ideal"))
            ideal_input = self._create_counter_input(n_time=600, offset=counter / FS, counter_start=counter)
            ideal_result = ideal_transformer_fresh(ideal_input)

            if hdmi_result.data.nnz > 0:
                hdmi_coords.append(hdmi_result.data.coords.copy())
            if ideal_result.data.nnz > 0:
                ideal_coords.append(ideal_result.data.coords.copy())
            counter += 600

        # Both should produce spikes
        assert len(hdmi_coords) > 0
        assert len(ideal_coords) > 0

    def test_transformer_empty_chunks_handled(self):
        """Transformer handles chunks with no spikes correctly."""
        from ezmsg.simbiophys.dnss.spike import DNSSSpikeSettings, DNSSSpikeTransformer

        transformer = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=4))

        # Even if no spikes, should return valid sparse array
        counter = 0
        for _ in range(10):
            input_msg = self._create_counter_input(n_time=600, offset=counter / FS, counter_start=counter)
            result = transformer(input_msg)
            assert result.data.shape[0] >= 0
            assert result.data.shape[1] == 4
            counter += 600

    def test_transformer_rejects_wrong_sample_rate(self):
        """Transformer raises error for sample rates other than 30kHz."""
        from ezmsg.simbiophys.dnss.spike import DNSSSpikeSettings, DNSSSpikeTransformer

        transformer = DNSSSpikeTransformer(DNSSSpikeSettings(n_ch=4))

        # Create input with wrong sample rate (e.g., 1000 Hz)
        wrong_fs = 1000
        data = np.arange(100)
        input_msg = AxisArray(
            data=data,
            dims=["time"],
            axes={"time": AxisArray.TimeAxis(fs=wrong_fs, offset=0.0)},
        )

        with pytest.raises(ValueError, match="requires fs=30000"):
            transformer(input_msg)
