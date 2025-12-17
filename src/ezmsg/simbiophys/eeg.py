"""EEG signal synthesis."""

import ezmsg.core as ez
from ezmsg.sigproc.math.add import Add
from ezmsg.util.messages.axisarray import AxisArray

from .clock import Clock, ClockSettings
from .noise import PinkNoise, PinkNoiseSettings
from .oscillator import Oscillator, OscillatorSettings


class EEGSynthSettings(ez.Settings):
    """See :obj:`OscillatorSettings`."""

    fs: float = 500.0  # Hz
    n_time: int = 100
    alpha_freq: float = 10.5  # Hz
    n_ch: int = 8


class EEGSynth(ez.Collection):
    """
    A :obj:`Collection` that chains a :obj:`Clock` to both :obj:`PinkNoise`
    and :obj:`Oscillator`, then :obj:`Add` s the result.

    Unlike the Oscillator, WhiteNoise, and PinkNoise composite processors which have linear
    flows, this class has a diamond flow, with clock branching to both PinkNoise and Oscillator,
    which then are combined in Add.

    Optional: Refactor as a ProducerUnit, similar to Clock, but we manually add all the other
     transformers.
    """

    SETTINGS = EEGSynthSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    CLOCK = Clock()
    NOISE = PinkNoise()
    OSC = Oscillator()
    ADD = Add()

    def configure(self) -> None:
        self.CLOCK.apply_settings(ClockSettings(dispatch_rate=self.SETTINGS.fs / self.SETTINGS.n_time))

        self.OSC.apply_settings(
            OscillatorSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate="ext_clock",
                freq=self.SETTINGS.alpha_freq,
            )
        )

        self.NOISE.apply_settings(
            PinkNoiseSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate="ext_clock",
                scale=5.0,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_SIGNAL, self.OSC.INPUT_SIGNAL),
            (self.CLOCK.OUTPUT_SIGNAL, self.NOISE.INPUT_SIGNAL),
            (self.OSC.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_A),
            (self.NOISE.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_B),
            (self.ADD.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
