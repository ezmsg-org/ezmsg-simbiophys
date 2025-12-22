"""Produce data mimicking Blackrock Neurotech's Digital Neural Signal Simulator"""

from typing import Generator

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import sparse
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, replace

"""
## Spike Pattern

We know the overall spiking pattern and the individual waveforms from the DNSS source code.

The pattern during single spikes:

```
 ** Ch 1 1-----------2-----------3-----------1-----------2- ... and
 ** Ch 2 ---2-----------3-----------1-----------2---------- ... on
 ** Ch 3 ------3-----------1-----------2-----------3------- ... and
 ** Ch 4 ---------1-----------2-----------3-----------1---- ... on
 ** Ch 5 1-----------2-----------3-----------1-----------2- ... and
```

The first waveform is inserted immediately into channels 1::4 as soon as the pattern starts.
Waveforms are inserted 7500 samples (0.25 seconds). At each iteration, the next waveform (% 3 waveforms)
is inserted into the next channel (% 4 channels).
This repeats until there are 36 spike + gap periods (36 * 7500), for 9 seconds total.


During bursting:

```
 ** Ch 1 1---2---3---1---2---3---1---2---3---1---2---3---1- ... and
 ** Ch 2 1---2---3---1---2---3---1---2---3---1---2---3---1- ... on
 ** Ch 3 1---2---3---1---2---3---1---2---3---1---2---3---1- ... and
 ** Ch 4 1---2---3---1---2---3---1---2---3---1---2---3---1- ... on
 ** Ch 5 1---2---3---1---2---3---1---2---3---1---2---3---1- ... and
```

Bursts begin immediately after the last gap following the 36th spike during the slow period.
During bursting, a spike occurs in all channels simultaneously with the same waveform.
Each spike in a burst comprises a spike waveform + gap for 300 total samples (0.01 s).
Waveforms cycle in order, 1, 2, 3, 1, 2, 3, and so on.
There are 99 total spikes in a burst for 0.99 second total.

A burst ends with a 300 sample gap (so 600 from the onset of the last spike in the burst)
before the slow period begins again.

** WARNING **

The above pattern describes what's supposed to happen, but there are 2 bugs when using the DNSS' HDMI output.

* First bug: The pattern actually starts on channel 2, not channel 1.
* Second bug: The first channel in a bank has its spikes (but not LFPs) delayed by 1 sample.
"""


# Pattern constants
INT_SLOW = 7500  # Samples between slow spikes
N_SLOW_SPIKES = 36
SAMPS_SLOW = INT_SLOW * N_SLOW_SPIKES  # 270,000 samples (9 seconds)

INT_BURST = 300  # Samples between burst spikes
N_BURST_SPIKES = 99
GAP_BURST = 300  # Gap after last burst spike
SAMPS_BURST = INT_BURST * N_BURST_SPIKES + GAP_BURST  # 30,000 samples (1 second)

FULL_PERIOD = SAMPS_SLOW + SAMPS_BURST  # 300,000 samples (10 seconds)

# Sample rate (default for DNSS)
FS = 30_000


def _spikes_in_range(start: int, end: int, burst: bool = False) -> npt.NDArray[np.int_]:
    """
    Return indices of spikes occurring in [start, end) for slow or burst phase.

    Spike i occurs at position i * interval. Returns array of i values
    where start <= i * interval < end.

    Args:
        start: Start of range (inclusive).
        end: End of range (exclusive).
        burst: If True, use burst phase parameters; otherwise use slow phase.
    """
    interval = INT_BURST if burst else INT_SLOW
    n_spikes = N_BURST_SPIKES if burst else N_SLOW_SPIKES

    if end <= start:
        return np.array([], dtype=np.int_)

    # i * interval >= start => i >= ceil(start / interval)
    i_min = int(np.ceil(start / interval))
    # i * interval < end => i < end / interval => i <= floor((end - 1) / interval)
    i_max = (end - 1) // interval + 1  # exclusive upper bound

    i_min = max(0, i_min)
    i_max = min(n_spikes, i_max)

    if i_min >= i_max:
        return np.array([], dtype=np.int_)

    return np.arange(i_min, i_max, dtype=np.int_)


def spike_event_generator(
    mode: str = "hdmi",
    n_chans: int = 4,
) -> Generator[
    tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
    int,
    None,
]:
    """
    Generator yielding spike event indices for the DNSS pattern.

    This is a send-able generator. After priming with next(), use send(n_samples)
    to get spikes for the next n_samples window. The generator maintains internal
    state tracking the current sample position.

    Args:
        mode: "hdmi" to reproduce HDMI bugs, "ideal" for ideal pattern.
        n_chans: Number of channels in the slow-phase rotation (default 4).

    Yields:
        Tuple of (coords, waveform_ids):
        - coords: Shape (2, n_spikes) array of [sample_indices, channel_indices]
        - waveform_ids: Waveform shape identifiers (1, 2, or 3)

    Example:
        gen = spike_event_generator()
        next(gen)  # Prime the generator
        coords, waveforms = gen.send(30000)  # Get spikes in first 30000 samples
        coords, waveforms = gen.send(15000)  # Get spikes in next 15000 samples
    """
    hdmi_mode = mode.lower() == "hdmi"
    ch_offset = 1 if hdmi_mode else 0  # HDMI bug #1: pattern starts on channel 1

    current_sample = 0
    empty_coords = np.array([[], []], dtype=np.int_)
    empty_waveforms = np.array([], dtype=np.int_)

    n_samples = yield None  # Prime - caller does next(gen)

    while True:
        if n_samples is None or n_samples <= 0:
            n_samples = yield (empty_coords, empty_waveforms)
            continue

        window_start = current_sample
        window_end = current_sample + n_samples

        result_arrays: list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]] = []

        # Process the window in chunks that stay within a single period
        pos = window_start
        while pos < window_end:
            pos_in_period = pos % FULL_PERIOD
            remaining = window_end - pos
            chunk_end_in_period = min(pos_in_period + remaining, FULL_PERIOD)
            window_offset = pos - window_start  # Offset from window_start for this chunk

            # === Slow phase: [0, SAMPS_SLOW) ===
            if pos_in_period < SAMPS_SLOW:
                slow_start = pos_in_period
                slow_end = min(chunk_end_in_period, SAMPS_SLOW)

                spike_indices = _spikes_in_range(slow_start, slow_end, burst=False)
                if len(spike_indices) > 0:
                    n_spikes = len(spike_indices)

                    # Compute base values for each spike
                    spike_pos_in_period = spike_indices * INT_SLOW
                    base_sample_idxs = spike_pos_in_period - pos_in_period + window_offset
                    base_channels = (spike_indices + ch_offset) % 4  # Always 4-channel base pattern
                    base_waveforms = (spike_indices % 3) + 1

                    # The 4-channel pattern tiles across all channels
                    # e.g., with n_chans=5, base ch 0 fires on ch 0 and 4
                    # For n_chans < 4, only spikes on channels 0..n_chans-1 are present
                    max_tiles = (n_chans + 3) // 4  # Ceiling division
                    sample_idxs = np.repeat(base_sample_idxs, max_tiles)
                    waveforms = np.repeat(base_waveforms, max_tiles)
                    base_channels_repeated = np.repeat(base_channels, max_tiles)
                    tile_indices = np.tile(np.arange(max_tiles, dtype=np.int_), n_spikes)
                    channels = base_channels_repeated + tile_indices * 4

                    # Filter to valid channels (handles both partial tiles and n_chans < 4)
                    valid_mask = channels < n_chans
                    sample_idxs = sample_idxs[valid_mask]
                    waveforms = waveforms[valid_mask]
                    channels = channels[valid_mask]

                    # HDMI bug #2: channel 0 spikes delayed by 1 sample
                    if hdmi_mode:
                        sample_idxs = sample_idxs + (channels == 0).astype(np.int_)

                    if len(channels) > 0:
                        result_arrays.append((sample_idxs, channels, waveforms))

            # === Burst phase: [SAMPS_SLOW, SAMPS_SLOW + N_BURST_SPIKES * INT_BURST) ===
            # Spikes occur at SAMPS_SLOW + i * INT_BURST for i in 0..98
            # The final GAP_BURST samples have no spikes
            burst_spike_end = SAMPS_SLOW + N_BURST_SPIKES * INT_BURST

            if chunk_end_in_period > SAMPS_SLOW and pos_in_period < burst_spike_end:
                burst_start = max(pos_in_period, SAMPS_SLOW)
                burst_end = min(chunk_end_in_period, burst_spike_end)

                # Convert to relative positions within burst phase
                rel_start = burst_start - SAMPS_SLOW
                rel_end = burst_end - SAMPS_SLOW

                spike_indices = _spikes_in_range(rel_start, rel_end, burst=True)
                if len(spike_indices) > 0:
                    n_spikes = len(spike_indices)

                    # Compute base positions and waveforms for each spike
                    spike_pos_in_period = SAMPS_SLOW + spike_indices * INT_BURST
                    base_sample_idxs = spike_pos_in_period - pos_in_period + window_offset
                    base_waveforms = (spike_indices % 3) + 1

                    # Expand: each spike fires on all n_chans channels
                    sample_idxs = np.repeat(base_sample_idxs, n_chans)
                    waveforms = np.repeat(base_waveforms, n_chans)
                    channels = np.tile(np.arange(n_chans, dtype=np.int_), n_spikes)

                    # HDMI bug #2: channel 0 spikes delayed by 1 sample
                    if hdmi_mode:
                        sample_idxs = sample_idxs + (channels == 0).astype(np.int_)

                    result_arrays.append((sample_idxs, channels, waveforms))

            pos += chunk_end_in_period - pos_in_period

        current_sample = window_end

        # Concatenate all result arrays and build coords
        if result_arrays:
            sample_idxs = np.concatenate([r[0] for r in result_arrays])
            chan_idxs = np.concatenate([r[1] for r in result_arrays])
            waveform_ids = np.concatenate([r[2] for r in result_arrays])
            coords = np.array([sample_idxs, chan_idxs], dtype=np.int_)
            n_samples = yield (coords, waveform_ids)
        else:
            n_samples = yield (empty_coords, empty_waveforms)


# =============================================================================
# Transformer-based implementation
# =============================================================================


class DNSSSpikeSettings(ez.Settings):
    """Settings for DNSS spike transformer."""

    n_ch: int = 256
    """Number of channels."""

    mode: str = "hdmi"
    """Mode: "hdmi" reproduces HDMI bugs, "ideal" for ideal pattern."""


@processor_state
class DNSSSpikeTransformerState:
    """State for DNSS spike transformer."""

    spike_gen: Generator | None = None
    template: AxisArray | None = None


class DNSSSpikeTransformer(BaseStatefulTransformer[DNSSSpikeSettings, AxisArray, AxisArray, DNSSSpikeTransformerState]):
    """
    Transforms input AxisArray into DNSS spike signal.

    Takes timing information from input message and generates spike data as sparse COO arrays.
    """

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize the spike generator."""
        # Verify sample rate is 30kHz - spike patterns are tied to this rate
        time_axis = message.axes["time"]
        fs = getattr(time_axis, "fs", 1.0 / time_axis.gain)
        expected_gain = 1.0 / FS
        if not np.isclose(time_axis.gain, expected_gain, rtol=1e-6):
            raise ValueError(
                f"DNSSSpikeTransformer requires fs={FS} Hz (gain={expected_gain:.6e}), "
                f"but received fs={fs:.1f} Hz (gain={time_axis.gain:.6e}). "
                f"Spike patterns cannot be resampled to other rates."
            )

        self._state.spike_gen = spike_event_generator(
            mode=self.settings.mode,
            n_chans=self.settings.n_ch,
        )
        next(self._state.spike_gen)

        # Pre-construct template AxisArray with channel axis
        self._state.template = AxisArray(
            data=sparse.COO(
                coords=np.array([[], []], dtype=np.int_),
                data=np.array([], dtype=np.int_),
                shape=(0, self.settings.n_ch),
            ),
            dims=["time", "ch"],
            axes={
                "time": message.axes["time"],
                "ch": AxisArray.CoordinateAxis(
                    data=np.arange(self.settings.n_ch),
                    dims=["ch"],
                ),
            },
        )

    def _process(self, message: AxisArray) -> AxisArray:
        """Transform input into spike signal."""
        n_samples = message.data.shape[0]

        # Generate spike events
        coords, waveform_ids = self._state.spike_gen.send(n_samples)

        # Create sparse COO array
        spike_data = sparse.COO(
            coords=coords,
            data=waveform_ids,
            shape=(n_samples, self.settings.n_ch),
        )

        return replace(
            self._state.template,
            data=spike_data,
            axes={
                "time": message.axes["time"],
                "ch": self._state.template.axes["ch"],
            },
        )


class DNSSSpikeUnit(BaseTransformerUnit[DNSSSpikeSettings, AxisArray, AxisArray, DNSSSpikeTransformer]):
    """Unit for generating DNSS spikes from counter input."""

    SETTINGS = DNSSSpikeSettings
