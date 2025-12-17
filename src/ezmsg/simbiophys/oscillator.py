"""Oscillator/sinusoidal signal generators."""

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import (
    BaseTransformer,
    BaseTransformerUnit,
    CompositeProducer,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ._base import BaseCounterFirstProducerUnit
from .counter import CounterProducer, CounterSettings


class SinGeneratorSettings(ez.Settings):
    """
    Settings for :obj:`SinGenerator`.
    See :obj:`sin` for parameter descriptions.
    """

    axis: str | None = "time"
    """
    The name of the axis over which the sinusoid passes.
    Note: The axis must exist in the msg.axes and be of type AxisArray.LinearAxis.
    """

    freq: float = 1.0
    """The frequency of the sinusoid, in Hz."""

    amp: float = 1.0  # Amplitude
    """The amplitude of the sinusoid."""

    phase: float = 0.0  # Phase offset (in radians)
    """The initial phase of the sinusoid, in radians."""


class SinTransformer(BaseTransformer[SinGeneratorSettings, AxisArray, AxisArray]):
    """Transforms counter values into sinusoidal waveforms."""

    def _process(self, message: AxisArray) -> AxisArray:
        """Transform input counter values into sinusoidal waveform."""
        axis = self.settings.axis or message.dims[0]

        ang_freq = 2.0 * np.pi * self.settings.freq
        w = (ang_freq * message.get_axis(axis).gain) * message.data
        out_data = self.settings.amp * np.sin(w + self.settings.phase)

        return replace(message, data=out_data)


class SinGenerator(BaseTransformerUnit[SinGeneratorSettings, AxisArray, AxisArray, SinTransformer]):
    """Unit for generating sinusoidal waveforms."""

    SETTINGS = SinGeneratorSettings


def sin(
    axis: str | None = "time",
    freq: float = 1.0,
    amp: float = 1.0,
    phase: float = 0.0,
) -> SinTransformer:
    """
    Construct a generator of sinusoidal waveforms in AxisArray objects.

    Returns:
        A primed generator that expects .send(axis_array) of sample counts
        and yields an AxisArray of sinusoids.
    """
    return SinTransformer(SinGeneratorSettings(axis=axis, freq=freq, amp=amp, phase=phase))


class OscillatorSettings(ez.Settings):
    """Settings for :obj:`Oscillator`"""

    n_time: int
    """Number of samples to output per block."""

    fs: float
    """Sampling rate of signal output in Hz"""

    n_ch: int = 1
    """Number of channels to output per block"""

    dispatch_rate: float | str | None = None
    """(Hz) | 'realtime' | 'ext_clock'"""

    freq: float = 1.0
    """Oscillation frequency in Hz"""

    amp: float = 1.0
    """Amplitude"""

    phase: float = 0.0
    """Phase offset (in radians)"""

    sync: bool = False
    """Adjust `freq` to sync with sampling rate"""


class OscillatorProducer(CompositeProducer[OscillatorSettings, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: OscillatorSettings,
    ) -> dict[str, CounterProducer | SinTransformer]:
        # Calculate synchronous settings if necessary
        freq = settings.freq
        mod = None
        if settings.sync:
            period = 1.0 / settings.freq
            mod = round(period * settings.fs)
            freq = 1.0 / (mod / settings.fs)

        return {
            "counter": CounterProducer(
                CounterSettings(
                    n_time=settings.n_time,
                    fs=settings.fs,
                    n_ch=settings.n_ch,
                    dispatch_rate=settings.dispatch_rate,
                    mod=mod,
                )
            ),
            "sin": SinTransformer(SinGeneratorSettings(freq=freq, amp=settings.amp, phase=settings.phase)),
        }


class Oscillator(BaseCounterFirstProducerUnit[OscillatorSettings, AxisArray, AxisArray, OscillatorProducer]):
    """Generates sinusoidal waveforms using a counter and sine transformer."""

    SETTINGS = OscillatorSettings
