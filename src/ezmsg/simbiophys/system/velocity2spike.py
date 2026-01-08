"""Convert polar velocity coordinates to simulated spike waveforms.

This module provides a system that encodes velocity (in polar coordinates) into
spike activity using a cosine tuning model, then generates spike events and
inserts realistic waveforms.

Pipeline:
    polar coords (magnitude, angle) -> cosine encoder -> clip -> Poisson events -> waveforms

Note:
    This system expects polar coordinates as input. Use CoordinateSpaces with
    mode=CART2POL upstream to convert Cartesian velocity (vx, vy) to polar
    coordinates (magnitude, angle).

See Also:
    :mod:`ezmsg.simbiophys.system.velocity2lfp`: Velocity to LFP encoding.
    :mod:`ezmsg.simbiophys.system.velocity2ecephys`: Combined spike + LFP encoding.
"""

import ezmsg.core as ez
import numpy as np
from ezmsg.event.kernel import ArrayKernel, MultiKernel
from ezmsg.event.kernel_insert import SparseKernelInserterSettings, SparseKernelInserterUnit
from ezmsg.event.poissonevents import PoissonEventSettings, PoissonEventUnit
from ezmsg.sigproc.math.clip import Clip, ClipSettings
from ezmsg.util.messages.axisarray import AxisArray

from ..cosine_encoder import CosineEncoderSettings, CosineEncoderUnit
from ..dnss.wfs import wf_orig


class Velocity2SpikeSettings(ez.Settings):
    """Settings for :obj:`Velocity2Spike`."""

    output_fs: float = 30_000.0
    """Output sampling rate in Hz."""

    output_ch: int = 256
    """Number of output channels (simulated electrodes)."""

    baseline_rate: float = 10.0
    """Baseline firing rate in Hz."""

    modulation_depth: float = 20.0 / 314.0
    """Directional modulation depth in Hz per (pixel/second).
    At max velocity (~314 px/s), this gives ~20 Hz modulation."""

    min_rate: float = 0.0
    """Minimum firing rate (Hz). Rates are clipped to this value."""

    seed: int = 6767
    """Random seed for reproducible preferred directions and waveform selection."""


class Velocity2Spike(ez.Collection):
    """Encode velocity (polar coordinates) into simulated spike waveforms.

    This system converts polar velocity coordinates into multi-channel spike activity:

    1. **Cosine tuning**: Each channel has a preferred direction; firing rate
       is modulated by the cosine of the angle between velocity and preferred
       direction, scaled by velocity magnitude.
    2. **Poisson spike generation**: Converts firing rates to discrete spike
       events using an inhomogeneous Poisson process.
    3. **Waveform insertion**: Inserts realistic spike waveforms at event times.

    Input:
        AxisArray with shape (N, 2) containing polar velocity coordinates.
        Dimension 0 is time, dimension 1 is [magnitude, angle].
        Use CoordinateSpaces(mode=CART2POL) upstream if starting from (vx, vy).

    Output:
        AxisArray with shape (M, output_ch) containing spike waveforms at
        output_fs sampling rate.
    """

    SETTINGS = Velocity2SpikeSettings

    # Polar velocity inputs (magnitude, angle)
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    RATE_ENCODER = CosineEncoderUnit()
    CLIP_RATE = Clip()
    SPIKE_EVENT = PoissonEventUnit()
    WAVEFORMS = SparseKernelInserterUnit()
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def configure(self) -> None:
        self.RATE_ENCODER.apply_settings(
            CosineEncoderSettings(
                output_ch=self.SETTINGS.output_ch,
                baseline=self.SETTINGS.baseline_rate,
                modulation=self.SETTINGS.modulation_depth,
                seed=self.SETTINGS.seed,
            )
        )
        self.CLIP_RATE.apply_settings(ClipSettings(min=self.SETTINGS.min_rate))
        self.SPIKE_EVENT.apply_settings(
            PoissonEventSettings(
                output_fs=self.SETTINGS.output_fs,
                assume_counts=False,
            )
        )
        self.WAVEFORMS.apply_settings(
            SparseKernelInserterSettings(
                kernel=MultiKernel({i + 1: ArrayKernel(wf.astype(np.float32)) for i, wf in enumerate(wf_orig)}),
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.RATE_ENCODER.INPUT_SIGNAL),
            (self.RATE_ENCODER.OUTPUT_SIGNAL, self.CLIP_RATE.INPUT_SIGNAL),
            (self.CLIP_RATE.OUTPUT_SIGNAL, self.SPIKE_EVENT.INPUT_SIGNAL),
            (self.SPIKE_EVENT.OUTPUT_SIGNAL, self.WAVEFORMS.INPUT_SIGNAL),
            (self.WAVEFORMS.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
