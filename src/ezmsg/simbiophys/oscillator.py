"""Oscillator/sinusoidal signal generators."""

import ezmsg.core as ez
import numpy as np
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

    freq: float = 1.0
    """The frequency of the sinusoid, in Hz."""

    amp: float = 1.0
    """The amplitude of the sinusoid."""

    phase: float = 0.0
    """The initial phase of the sinusoid, in radians."""


@processor_state
class SinTransformerState:
    """State for SinTransformer."""

    template: AxisArray | None = None


class SinTransformer(BaseStatefulTransformer[SinGeneratorSettings, AxisArray, AxisArray, SinTransformerState]):
    """
    Transforms counter values into sinusoidal waveforms.

    Takes AxisArray with integer counter values and generates sinusoidal
    output based on the time axis sample rate.
    """

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize template with channel axis."""
        n_ch = self.settings.n_ch
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

    def _process(self, message: AxisArray) -> AxisArray:
        """Transform input counter values into sinusoidal waveform."""
        n_ch = self.settings.n_ch

        # Get sample rate from time axis
        time_axis = message.axes["time"]
        dt = time_axis.gain  # 1/fs

        # Calculate sinusoid: amp * sin(2*pi*freq*t + phase)
        # t = counter * dt
        ang_freq = 2.0 * np.pi * self.settings.freq
        t = message.data * dt
        sin_data = self.settings.amp * np.sin(ang_freq * t + self.settings.phase)

        # Tile across channels if needed
        if n_ch > 1:
            sin_data = np.tile(sin_data[:, np.newaxis], (1, n_ch))
        else:
            sin_data = sin_data[:, np.newaxis]

        # Create output using template
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
