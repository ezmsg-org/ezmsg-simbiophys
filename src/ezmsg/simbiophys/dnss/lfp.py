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
    CompositeProducer,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, replace

from .._base import BaseCounterFirstProducerUnit
from ..counter import CounterProducer, CounterSettings

# Sample rate (default for DNSS)
FS = 30_000

# Spike mode: 3 sinusoids summed, repeats every 1 second
LFP_FREQS = [1.0, 3.0, 9.0]
LFP_PERIOD = FS  # 30,000 samples (1 second)

# Other mode: sequential sine waves, repeats every 2 seconds
OTHER_PERIOD = FS * 2  # 60,000 samples (2 seconds)

# Gain/shift coefficients by mode
LFP_GAINS: dict[str, list[float]] = {
    "hdmi": [894.4, 894.4, 894.4],
    "pedestal_norm": [604.7679245, 727.13212154, 702.44150471],
    "pedestal_wide": [675.2529287, 679.43195229, 677.98366296],
}

LFP_SAMPLE_SHIFTS: dict[str, list[int]] = {
    "hdmi": [2, 1, 2],
    "pedestal_norm": [3_625, 396, 23],
    "pedestal_wide": [525, 63, 7],
}


def _generate_spike_lfp_period(mode: str = "hdmi") -> npt.NDArray[np.float64]:
    """
    Generate one period (1 second) of LFP for spike mode.

    Returns:
        Array of shape (LFP_PERIOD,) containing LFP values.
    """
    gains = LFP_GAINS[mode]
    sample_shifts = LFP_SAMPLE_SHIFTS[mode]
    t_shifts = [s / FS for s in sample_shifts]

    t_vec = np.arange(LFP_PERIOD) / FS
    lfp = np.zeros(LFP_PERIOD, dtype=np.float64)

    for freq, gain, phi in zip(LFP_FREQS, gains, t_shifts):
        lfp += gain * np.sin(2 * np.pi * freq * (t_vec + phi))

    return lfp


def _generate_other_lfp_period(mode: str = "hdmi") -> npt.NDArray[np.float64]:
    """
    Generate one period (2 seconds) of LFP for "other" mode.

    Returns:
        Array of shape (OTHER_PERIOD,) containing LFP values.
    """
    # Other mode pattern parameters
    freqs = [1, 10, 80, 100, 1000]
    shifts = [0, 720, 90, 0, 0]  # Sample shifts
    starts = [0, 30_000, 45_000, 45_285, 52_785]
    stops = [29_279, 45_000, 45_285, 52_785, 60_000]

    lfp = np.zeros(OTHER_PERIOD, dtype=np.float64)

    for freq, shift, start, stop in zip(freqs, shifts, starts, stops):
        phi = shift / FS
        t_vec = np.arange(stop - start) / FS
        lfp[start:stop] = np.sin(2 * np.pi * freq * (t_vec + phi))

    # Hold value at end of first incomplete wave
    lfp[stops[0] : starts[1]] = lfp[stops[0] - 1]

    # Apply gain
    gain = 6_000 if mode.startswith("hdmi") else 1_000
    lfp *= gain

    return lfp


def lfp_generator(
    pattern: str = "spike",
    mode: str = "hdmi",
) -> Generator[npt.NDArray[np.float64], int, None]:
    """
    Generator yielding LFP samples for the DNSS pattern.

    This is a send-able generator. After priming with next(), use send(n_samples)
    to get LFP values for the next n_samples window. The generator maintains internal
    state tracking the current sample position.

    Args:
        pattern: "spike" for normal neural signal mode, "other" for other mode.
        mode: "hdmi" for digital output, "pedestal_norm" or "pedestal_wide" for analog.

    Yields:
        1D array of LFP values (same for all channels).

    Example:
        gen = lfp_generator()
        next(gen)  # Prime the generator
        lfp = gen.send(30000)  # Get 1 second of LFP
        lfp = gen.send(15000)  # Get next 0.5 seconds
    """
    # Pre-generate one full period
    if pattern.lower() == "other":
        period = _generate_other_lfp_period(mode=mode)
        period_len = OTHER_PERIOD
    else:
        period = _generate_spike_lfp_period(mode=mode)
        period_len = LFP_PERIOD

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
# Transformer-based implementation (preferred)
# =============================================================================


class DNSSLFPTransformerSettings(ez.Settings):
    """Settings for DNSS LFP transformer."""

    pattern: str = "spike"
    """LFP pattern: "spike" for normal neural signal mode, "other" for other mode."""

    mode: str = "hdmi"
    """Mode: "hdmi" for digital output, "pedestal_norm" or "pedestal_wide" for analog."""


@processor_state
class DNSSLFPTransformerState:
    """State for DNSS LFP transformer."""

    lfp_gen: Generator | None = None


class DNSSLFPTransformer(
    BaseStatefulTransformer[DNSSLFPTransformerSettings, AxisArray, AxisArray, DNSSLFPTransformerState]
):
    """
    Transforms input AxisArray into DNSS LFP signal.

    Takes timing information from input message and generates LFP data.
    All channels receive identical LFP values.
    """

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize the LFP generator."""
        self._state.lfp_gen = lfp_generator(
            pattern=self.settings.pattern,
            mode=self.settings.mode,
        )
        next(self._state.lfp_gen)

    def _process(self, message: AxisArray) -> AxisArray:
        """Transform input into LFP signal."""
        n_samples = message.data.shape[0]
        n_chans = message.data.shape[1] if message.data.ndim > 1 else 1

        # Generate LFP samples
        lfp_1d = self._state.lfp_gen.send(n_samples)

        # Tile across channels
        if n_chans > 1:
            lfp_data = np.tile(lfp_1d[:, np.newaxis], (1, n_chans))
        else:
            lfp_data = lfp_1d[:, np.newaxis]

        return replace(message, data=lfp_data)


class DNSSLFPGenerator(BaseTransformerUnit[DNSSLFPTransformerSettings, AxisArray, AxisArray, DNSSLFPTransformer]):
    """Unit for generating DNSS LFP from counter input."""

    SETTINGS = DNSSLFPTransformerSettings


# =============================================================================
# Composite producer (standalone, like OscillatorProducer)
# =============================================================================


class DNSSLFPSettings(ez.Settings):
    """Settings for standalone DNSS LFP producer."""

    n_time: int = 600
    """Number of samples per block (default: 600 = 20ms at 30kHz)."""

    fs: float = 30_000.0
    """Sample rate in Hz."""

    n_ch: int = 256
    """Number of channels."""

    dispatch_rate: float | str | None = None
    """Dispatch rate: Hz, 'realtime', 'ext_clock', or None (fast as possible)."""

    pattern: str = "spike"
    """LFP pattern: "spike" or "other"."""

    mode: str = "hdmi"
    """Mode: "hdmi", "pedestal_norm", or "pedestal_wide"."""


class DNSSLFPProducer(CompositeProducer[DNSSLFPSettings, AxisArray]):
    """
    Produces DNSS LFP signal as a standalone producer.

    Internally uses Counter for timing and DNSSLFPTransformer for LFP generation.
    """

    @staticmethod
    def _initialize_processors(
        settings: DNSSLFPSettings,
    ) -> dict[str, CounterProducer | DNSSLFPTransformer]:
        return {
            "counter": CounterProducer(
                CounterSettings(
                    n_time=settings.n_time,
                    fs=settings.fs,
                    n_ch=settings.n_ch,
                    dispatch_rate=settings.dispatch_rate,
                )
            ),
            "lfp": DNSSLFPTransformer(
                DNSSLFPTransformerSettings(
                    pattern=settings.pattern,
                    mode=settings.mode,
                )
            ),
        }


class DNSSLFPUnit(BaseCounterFirstProducerUnit[DNSSLFPSettings, AxisArray, AxisArray, DNSSLFPProducer]):
    """Unit for standalone DNSS LFP production."""

    SETTINGS = DNSSLFPSettings
