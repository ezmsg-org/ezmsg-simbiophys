"""Convert 2D cursor velocity to simulated extracellular electrophysiology.

This module provides a complete system that encodes cursor velocity into
realistic ecephys signals containing both spike waveforms and LFP-like
background activity.

Pipeline:
    velocity (x,y) --+--> Velocity2Spike --> spikes --|
                     |                                +--> Add --> ecephys
                     +--> Velocity2LFP ----> lfp -----|

This is the top-level system for velocity-encoded neural simulation. Use this
when you need full ecephys-like output suitable for testing BCI decoders.

See Also:
    :mod:`ezmsg.simbiophys.system.velocity2spike`: Spike-only encoding.
    :mod:`ezmsg.simbiophys.system.velocity2lfp`: LFP-only encoding.
"""

import ezmsg.core as ez
from ezmsg.sigproc.math.add import Add
from ezmsg.util.messages.axisarray import AxisArray

from .velocity2lfp import Velocity2LFP, Velocity2LFPSettings
from .velocity2spike import Velocity2Spike, Velocity2SpikeSettings


class VelocityEncoderSettings(ez.Settings):
    """Settings for :obj:`VelocityEncoder`."""

    output_fs: float = 30_000.0
    """Output sampling rate in Hz."""

    output_ch: int = 256
    """Number of output channels (simulated electrodes)."""

    seed: int = 6767
    """Random seed for reproducible spike and LFP generation."""


class VelocityEncoder(ez.Collection):
    """Encode cursor velocity into simulated extracellular electrophysiology.

    This system combines spike and LFP encoding to produce realistic ecephys
    signals. It runs two parallel pipelines:

    1. **Spike branch** (:obj:`Velocity2Spike`): Generates cosine-tuned spike
       waveforms based on velocity direction and magnitude.
    2. **LFP branch** (:obj:`Velocity2LFP`): Generates velocity-modulated
       colored noise representing local field potentials.

    The outputs are summed to produce the final ecephys signal.

    Input:
        AxisArray with shape (N, 2) containing cursor velocity in pixels/second.
        Dimension 0 is time, dimension 1 is [vx, vy].

    Output:
        AxisArray with shape (M, output_ch) containing combined spike and LFP
        signals at output_fs sampling rate.

    Example:
        >>> encoder = VelocityEncoder(VelocityEncoderSettings(
        ...     output_fs=30_000.0,
        ...     output_ch=256,
        ...     seed=42,
        ... ))
    """

    SETTINGS = VelocityEncoderSettings

    # Velocity inputs (via mouse / gamepad system, or via task parsing system)
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    SPIKES = Velocity2Spike()
    LFP = Velocity2LFP()
    ADD = Add()  # Add colored noise and waveforms
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def configure(self) -> None:
        self.SPIKES.apply_settings(
            Velocity2SpikeSettings(
                output_fs=self.SETTINGS.output_fs, output_ch=self.SETTINGS.output_ch, seed=self.SETTINGS.seed
            )
        )
        self.LFP.apply_settings(
            Velocity2LFPSettings(
                output_fs=self.SETTINGS.output_fs, output_ch=self.SETTINGS.output_ch, seed=self.SETTINGS.seed
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.SPIKES.INPUT_SIGNAL),
            (self.SPIKES.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_A),
            (self.INPUT_SIGNAL, self.LFP.INPUT_SIGNAL),
            (self.LFP.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_B),
            (self.ADD.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
