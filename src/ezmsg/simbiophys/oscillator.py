"""Oscillator/sinusoidal signal generators."""

import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseClockDrivenProducer,
    BaseClockDrivenUnit,
    ClockDrivenSettings,
    ClockDrivenState,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis, replace


class SinGeneratorSettings(ClockDrivenSettings):
    """Settings for :obj:`SinGenerator`."""

    n_ch: int = 1
    """Number of channels to output."""

    freq: float | npt.ArrayLike = 1.0
    """The frequency of the sinusoid, in Hz. Scalar or per-channel array."""

    amp: float | npt.ArrayLike = 1.0
    """The amplitude of the sinusoid. Scalar or per-channel array."""

    phase: float | npt.ArrayLike = 0.0
    """The initial phase of the sinusoid, in radians. Scalar or per-channel array."""


@processor_state
class SinGeneratorState(ClockDrivenState):
    """State for SinGenerator."""

    template: AxisArray | None = None
    # Pre-computed arrays for efficient processing, shape (1, 1) or (1, n_ch)
    ang_freq: np.ndarray | None = None  # 2*pi*freq
    amp: np.ndarray | None = None
    phase: np.ndarray | None = None


class SinProducer(BaseClockDrivenProducer[SinGeneratorSettings, SinGeneratorState]):
    """
    Generates sinusoidal waveforms synchronized to clock ticks.

    Each clock tick produces a block of sinusoidal data based on the
    sample rate (fs) and chunk size (n_time) settings.
    """

    def _reset_state(self, time_axis: LinearAxis) -> None:
        """Initialize template and pre-compute parameter arrays."""
        n_ch = self.settings.n_ch

        # Create template
        self._state.template = AxisArray(
            data=np.zeros((0, n_ch)),
            dims=["time", "ch"],
            axes={
                "time": time_axis,
                "ch": AxisArray.CoordinateAxis(
                    data=np.arange(n_ch),
                    dims=["ch"],
                ),
            },
        )

        # Convert settings to arrays and validate
        freq = np.atleast_1d(self.settings.freq)
        amp = np.atleast_1d(self.settings.amp)
        phase = np.atleast_1d(self.settings.phase)

        for name, arr in [("freq", freq), ("amp", amp), ("phase", phase)]:
            if arr.size > 1 and arr.size != n_ch:
                raise ValueError(
                    f"{name} has length {arr.size} but n_ch is {n_ch}. "
                    f"Per-channel arrays must have length equal to n_ch."
                )

        # Reshape for broadcasting: (1, n_ch) or (1, 1)
        freq = freq.reshape(1, -1) if freq.size > 1 else freq.reshape(1, 1)
        amp = amp.reshape(1, -1) if amp.size > 1 else amp.reshape(1, 1)
        phase = phase.reshape(1, -1) if phase.size > 1 else phase.reshape(1, 1)

        # Store pre-computed values
        self._state.ang_freq = 2.0 * np.pi * freq
        self._state.amp = amp
        self._state.phase = phase

    def _produce(self, n_samples: int, time_axis: LinearAxis) -> AxisArray:
        """Generate sinusoidal waveform for this chunk."""
        # Calculate sinusoid: amp * sin(ang_freq*t + phase)
        # t shape: (n_time,) -> (n_time, 1) for broadcasting with (1, n_ch)
        t = (np.arange(n_samples) + self._state.counter)[:, np.newaxis] * time_axis.gain
        sin_data = self._state.amp * np.sin(self._state.ang_freq * t + self._state.phase)

        # Tile if all params were scalar but n_ch > 1
        if sin_data.shape[1] < self.settings.n_ch:
            sin_data = np.tile(sin_data, (1, self.settings.n_ch))

        return replace(
            self._state.template,
            data=sin_data,
            axes={
                **self._state.template.axes,
                "time": time_axis,
            },
        )


class SinGenerator(BaseClockDrivenUnit[SinGeneratorSettings, SinProducer]):
    """
    Generates sinusoidal waveforms synchronized to clock ticks.

    Receives timing from INPUT_CLOCK (LinearAxis from Clock) and outputs
    sinusoidal AxisArray on OUTPUT_SIGNAL.
    """

    SETTINGS = SinGeneratorSettings
