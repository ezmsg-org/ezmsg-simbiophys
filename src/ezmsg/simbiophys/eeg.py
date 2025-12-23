"""EEG signal synthesis."""

import ezmsg.core as ez
from ezmsg.baseproc import Clock, ClockSettings, Counter, CounterSettings
from ezmsg.sigproc.math.add import Add
from ezmsg.util.messages.axisarray import AxisArray

from .noise import WhiteNoise, WhiteNoiseSettings
from .oscillator import SinGenerator, SinGeneratorSettings


class EEGSynthSettings(ez.Settings):
    """Settings for EEG synthesizer."""

    fs: float = 500.0
    """Sample rate in Hz."""

    n_time: int = 100
    """Number of samples per block."""

    alpha_freq: float = 10.5
    """Alpha frequency in Hz."""

    n_ch: int = 8
    """Number of channels."""


class EEGSynth(ez.Collection):
    """
    A Collection that generates synthetic EEG signals.

    Combines pink noise with alpha oscillations using a diamond flow:
    Clock -> Counter -> {Noise, Oscillator} -> Add -> Output

    Network flow:
        Clock -> Counter -> {Noise, Oscillator}
        Noise -> Add.A
        Oscillator -> Add.B
        Add -> OUTPUT
    """

    SETTINGS = EEGSynthSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    CLOCK = Clock()
    COUNTER = Counter()
    NOISE = WhiteNoise()
    OSC = SinGenerator()
    ADD = Add()

    def configure(self) -> None:
        dispatch_rate = self.SETTINGS.fs / self.SETTINGS.n_time
        self.CLOCK.apply_settings(ClockSettings(dispatch_rate=dispatch_rate))

        self.COUNTER.apply_settings(
            CounterSettings(
                fs=self.SETTINGS.fs,
                n_time=self.SETTINGS.n_time,
            )
        )

        self.NOISE.apply_settings(
            WhiteNoiseSettings(
                n_ch=self.SETTINGS.n_ch,
                scale=5.0,
            )
        )

        self.OSC.apply_settings(
            SinGeneratorSettings(
                n_ch=self.SETTINGS.n_ch,
                freq=self.SETTINGS.alpha_freq,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            # Clock drives Counter
            (self.CLOCK.OUTPUT_SIGNAL, self.COUNTER.INPUT_SIGNAL),
            # Counter fans out to both Noise and Oscillator
            (self.COUNTER.OUTPUT_SIGNAL, self.OSC.INPUT_SIGNAL),
            (self.COUNTER.OUTPUT_SIGNAL, self.NOISE.INPUT_SIGNAL),
            # Combine outputs
            (self.OSC.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_A),
            (self.NOISE.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_B),
            (self.ADD.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
