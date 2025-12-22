"""
# LFP

## Spike Mode

The LFP is a sum of 3 sinusoids at frequencies of 1.0, 3.0, and 9.0 Hz.

The amplitudes and phase shifts are calculated empirically from analyzing some data.
Digitally, and transmitted via the HDMI, there is almost no phase delay in these sinusoids;
they are shifted 2, 1, and 2 samples after the pattern start, respectively. Their amplitudes
are all 894.4.

When using analog components such as the pedestal and headstage, the intrinsic filtering
characteristics cause non-linear phase delay, shifting the sinusoids an uneven amount.

## "Other" Mode

The "Other" pattern comprises sequential sine waves of increasing frequency.
All of amplitude 6_000 (HDMI) or 1_000 (pedestal).
29_279 samples of a 1 Hz sine wave, phase shift 0.
721 samples held at last value preceding an incomplete wave.
15_000 samples (0.5 seconds) of 10 Hz sine, phase shift 720.
285 samples of 80 Hz sine wave, phase shift 90 samples.
7_500 samples of 100 Hz sine wave, no phase shift.
7_215 samples (0.24 seconds) of 1000 Hz sine, phase shift 0.
"""

from typing import Generator

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, replace

# Default sample rate for DNSS
DEFAULT_FS = 30_000

# Spike mode: 3 sinusoids summed, repeats every 1 second
LFP_FREQS = [1.0, 3.0, 9.0]
LFP_PERIOD = 1.0  # seconds

# Other mode: sequential sine waves, repeats every 2 seconds
OTHER_PERIOD = 2.0  # seconds

# Gain/shift coefficients by mode
LFP_GAINS: dict[str, list[float]] = {
    "hdmi": [894.4, 894.4, 894.4],
    "pedestal_norm": [604.7679245, 727.13212154, 702.44150471],
    "pedestal_wide": [675.2529287, 679.43195229, 677.98366296],
}

# Time shifts in seconds (originally measured at 30kHz)
LFP_TIME_SHIFTS: dict[str, list[float]] = {
    "hdmi": [2 / 30_000, 1 / 30_000, 2 / 30_000],
    "pedestal_norm": [3_625 / 30_000, 396 / 30_000, 23 / 30_000],
    "pedestal_wide": [525 / 30_000, 63 / 30_000, 7 / 30_000],
}


def _generate_spike_lfp_period(mode: str = "hdmi", fs: float = DEFAULT_FS) -> npt.NDArray[np.float64]:
    """
    Generate one period (1 second) of LFP for spike mode.

    Args:
        mode: "hdmi" for digital output, "pedestal_norm" or "pedestal_wide" for analog.
        fs: Sample rate in Hz.

    Returns:
        Array of shape (n_samples,) containing LFP values, where n_samples = int(LFP_PERIOD * fs).
    """
    n_samples = int(LFP_PERIOD * fs)
    gains = LFP_GAINS[mode]
    t_shifts = LFP_TIME_SHIFTS[mode]

    t_vec = np.arange(n_samples) / fs
    lfp = np.zeros(n_samples, dtype=np.float64)

    for freq, gain, phi in zip(LFP_FREQS, gains, t_shifts):
        lfp += gain * np.sin(2 * np.pi * freq * (t_vec + phi))

    return lfp


def _generate_other_lfp_period(mode: str = "hdmi", fs: float = DEFAULT_FS) -> npt.NDArray[np.float64]:
    """
    Generate one period (2 seconds) of LFP for "other" mode.

    Args:
        mode: "hdmi" for digital output, "pedestal_norm" or "pedestal_wide" for analog.
        fs: Sample rate in Hz.

    Returns:
        Array of shape (n_samples,) containing LFP values, where n_samples = int(OTHER_PERIOD * fs).
    """
    n_samples = int(OTHER_PERIOD * fs)

    # Other mode pattern parameters (times in seconds, originally defined at 30kHz)
    freqs = [1, 10, 80, 100, 1000]
    time_shifts = [0.0, 720 / 30_000, 90 / 30_000, 0.0, 0.0]  # seconds
    time_starts = [0.0, 1.0, 1.5, 45_285 / 30_000, 52_785 / 30_000]  # seconds
    time_stops = [29_279 / 30_000, 1.5, 45_285 / 30_000, 52_785 / 30_000, 2.0]  # seconds

    lfp = np.zeros(n_samples, dtype=np.float64)

    for freq, phi, t_start, t_stop in zip(freqs, time_shifts, time_starts, time_stops):
        # Convert time boundaries to sample indices
        start = int(t_start * fs)
        stop = int(t_stop * fs)
        n_seg = stop - start
        if n_seg <= 0:
            continue
        t_vec = np.arange(n_seg) / fs
        lfp[start:stop] = np.sin(2 * np.pi * freq * (t_vec + phi))

    # Hold value at end of first incomplete wave
    first_stop = int(time_stops[0] * fs)
    second_start = int(time_starts[1] * fs)
    if first_stop > 0 and second_start > first_stop:
        lfp[first_stop:second_start] = lfp[first_stop - 1]

    # Apply gain
    gain = 6_000 if mode.startswith("hdmi") else 1_000
    lfp *= gain

    return lfp


def lfp_generator(
    pattern: str = "spike",
    mode: str = "hdmi",
    fs: float = DEFAULT_FS,
) -> Generator[npt.NDArray[np.float64], int, None]:
    """
    Generator yielding LFP samples for the DNSS pattern.

    This is a send-able generator. After priming with next(), use send(n_samples)
    to get LFP values for the next n_samples window. The generator maintains internal
    state tracking the current sample position.

    Args:
        pattern: "spike" for normal neural signal mode, "other" for other mode.
        mode: "hdmi" for digital output, "pedestal_norm" or "pedestal_wide" for analog.
        fs: Sample rate in Hz.

    Yields:
        1D array of LFP values (same for all channels).

    Example:
        gen = lfp_generator(fs=30000)
        next(gen)  # Prime the generator
        lfp = gen.send(30000)  # Get 1 second of LFP
        lfp = gen.send(15000)  # Get next 0.5 seconds
    """
    # Pre-generate one full period at the specified sample rate
    if pattern.lower() == "other":
        period = _generate_other_lfp_period(mode=mode, fs=fs)
        period_len = int(OTHER_PERIOD * fs)
    else:
        period = _generate_spike_lfp_period(mode=mode, fs=fs)
        period_len = int(LFP_PERIOD * fs)

    current_sample = 0
    empty = np.array([], dtype=np.float64)

    n_samples = yield None  # Prime - caller does next(gen)

    while True:
        if n_samples is None or n_samples <= 0:
            n_samples = yield empty
            continue

        # Build output by extracting from the repeating period
        result = np.empty(n_samples, dtype=np.float64)
        result_pos = 0

        while result_pos < n_samples:
            pos_in_period = current_sample % period_len
            remaining = n_samples - result_pos
            chunk_size = min(remaining, period_len - pos_in_period)

            result[result_pos : result_pos + chunk_size] = period[pos_in_period : pos_in_period + chunk_size]

            result_pos += chunk_size
            current_sample += chunk_size

        n_samples = yield result


# =============================================================================
# Transformer-based implementation
# =============================================================================


class DNSSLFPSettings(ez.Settings):
    """Settings for DNSS LFP transformer."""

    n_ch: int = 256
    """Number of channels."""

    pattern: str = "spike"
    """LFP pattern: "spike" for normal neural signal mode, "other" for other mode."""

    mode: str = "hdmi"
    """Mode: "hdmi" for digital output, "pedestal_norm" or "pedestal_wide" for analog."""


@processor_state
class DNSSLFPTransformerState:
    """State for DNSS LFP transformer."""

    lfp_gen: Generator | None = None
    template: AxisArray | None = None


class DNSSLFPTransformer(BaseStatefulTransformer[DNSSLFPSettings, AxisArray, AxisArray, DNSSLFPTransformerState]):
    """
    Transforms input AxisArray into DNSS LFP signal.

    Takes timing information from input message and generates LFP data.
    All channels receive identical LFP values.
    """

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize the LFP generator."""
        # Get sample rate from input message (fs = 1/gain for LinearAxis)
        time_axis = message.axes["time"]
        fs = getattr(time_axis, "fs", 1.0 / time_axis.gain)
        self._state.lfp_gen = lfp_generator(
            pattern=self.settings.pattern,
            mode=self.settings.mode,
            fs=fs,
        )
        next(self._state.lfp_gen)

        # Pre-construct template AxisArray with channel axis
        self._state.template = AxisArray(
            data=np.zeros((0, self.settings.n_ch), dtype=np.float64),
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
        """Transform input into LFP signal."""
        n_samples = message.data.shape[0]

        # Generate LFP samples
        lfp_1d = self._state.lfp_gen.send(n_samples)

        # Tile across channels
        if self.settings.n_ch > 1:
            lfp_data = np.tile(lfp_1d[:, np.newaxis], (1, self.settings.n_ch))
        else:
            lfp_data = lfp_1d[:, np.newaxis]

        return replace(
            self._state.template,
            data=lfp_data,
            axes={
                "time": message.axes["time"],
                "ch": self._state.template.axes["ch"],
            },
        )


class DNSSLFPUnit(BaseTransformerUnit[DNSSLFPSettings, AxisArray, AxisArray, DNSSLFPTransformer]):
    """Unit for generating DNSS LFP from counter input."""

    SETTINGS = DNSSLFPSettings
