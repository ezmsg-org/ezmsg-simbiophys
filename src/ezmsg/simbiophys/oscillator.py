"""Oscillator/sinusoidal signal generators."""

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, replace


class SinGeneratorSettings(ez.Settings):
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
class SinTransformerState:
    """State for SinTransformer."""

    template: AxisArray | None = None
    # Pre-computed arrays for efficient processing, shape (1, 1) or (1, n_ch)
    ang_freq: np.ndarray | None = None  # 2*pi*freq
    amp: np.ndarray | None = None
    phase: np.ndarray | None = None


class SinTransformer(BaseStatefulTransformer[SinGeneratorSettings, AxisArray, AxisArray, SinTransformerState]):
    """
    Transforms counter values into sinusoidal waveforms.

    Takes AxisArray with integer counter values and generates sinusoidal
    output based on the time axis sample rate.
    """

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize template and pre-compute parameter arrays."""
        n_ch = self.settings.n_ch

        # Create template
        self._state.template = AxisArray(
            data=np.zeros((0, n_ch)),
            dims=["time", "ch"],
            axes={
                "time": message.axes["time"],
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

    def _process(self, message: AxisArray) -> AxisArray:
        """Transform input counter values into sinusoidal waveform."""
        # Calculate sinusoid: amp * sin(ang_freq*t + phase)
        # t shape: (n_time,) -> (n_time, 1) for broadcasting with (1, n_ch)
        t = message.data[:, np.newaxis] * message.axes["time"].gain
        sin_data = self._state.amp * np.sin(self._state.ang_freq * t + self._state.phase)

        # Tile if all params were scalar but n_ch > 1
        if sin_data.shape[1] < self.settings.n_ch:
            sin_data = np.tile(sin_data, (1, self.settings.n_ch))

        return replace(
            self._state.template,
            data=sin_data,
            axes={
                "time": message.axes["time"],
                "ch": self._state.template.axes["ch"],
            },
        )


class SinGenerator(BaseTransformerUnit[SinGeneratorSettings, AxisArray, AxisArray, SinTransformer]):
    """
    Transforms counter input into sinusoidal waveform.

    Receives timing from INPUT_SIGNAL (AxisArray from Counter) and outputs
    sinusoidal AxisArray.
    """

    SETTINGS = SinGeneratorSettings
