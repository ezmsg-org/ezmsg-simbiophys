"""Convert 2D cursor velocity to simulated LFP-like colored noise.

This module provides a system that encodes cursor velocity into the spectral
properties of colored (1/f^beta) noise, producing LFP-like signals.

Pipeline:
    velocity (x,y) -> polar coords -> scale to beta range -> colored noise -> mix to channels

The velocity magnitude modulates one noise source's spectral exponent, and the
velocity angle modulates another. These two sources are then mixed across
output channels using a spatial mixing matrix.

See Also:
    :mod:`ezmsg.simbiophys.system.velocity2spike`: Velocity to spike encoding.
    :mod:`ezmsg.simbiophys.system.velocity2ecephys`: Combined spike + LFP encoding.
"""

import ezmsg.core as ez
import numpy as np
from ezmsg.sigproc.affinetransform import AffineTransform, AffineTransformSettings
from ezmsg.sigproc.coordinatespaces import CoordinateMode, CoordinateSpaces, CoordinateSpacesSettings
from ezmsg.sigproc.linear import LinearTransform, LinearTransformSettings
from ezmsg.sigproc.math.clip import Clip, ClipSettings
from ezmsg.util.messages.axisarray import AxisArray

from ..dynamic_colored_noise import DynamicColoredNoiseSettings, DynamicColoredNoiseUnit


class Velocity2LFPSettings(ez.Settings):
    """Settings for :obj:`Velocity2LFP`."""

    output_fs: float = 30_000.0
    """Output sampling rate in Hz."""

    output_ch: int = 256
    """Number of output channels (simulated electrodes)."""

    seed: int = 6767
    """Random seed for reproducible mixing matrix."""


class Velocity2LFP(ez.Collection):
    """Encode cursor velocity into LFP-like colored noise.

    This system converts 2D cursor velocity into multi-channel LFP-like signals:

    1. **Coordinate transform**: Converts Cartesian velocity (x, y) to polar
       coordinates (magnitude, angle).
    2. **Scale to beta**: Maps velocity magnitude (0-314 px/s) and angle (0-2pi)
       to spectral exponent beta (0.5-2.0) for colored noise generation.
    3. **Clip**: Ensures beta values stay within valid range [0, 2].
    4. **Colored noise**: Generates 1/f^beta noise where beta is dynamically
       modulated by the scaled velocity. Two noise sources are generated:
       one modulated by magnitude, one by angle.
    5. **Spatial mixing**: Projects the 2 noise sources onto output_ch channels
       using a sinusoidal mixing matrix with random perturbations.

    Input:
        AxisArray with shape (N, 2) containing cursor velocity in pixels/second.
        Dimension 0 is time, dimension 1 is [vx, vy].

    Output:
        AxisArray with shape (M, output_ch) containing LFP-like colored noise
        at output_fs sampling rate.
    """

    SETTINGS = Velocity2LFPSettings

    # Velocity inputs (via mouse / gamepad, or via task parsing system)
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    COORDS = CoordinateSpaces()
    ALPHA_EXP = LinearTransform()
    CLIP_ALPHA = Clip()
    PINK_NOISE = DynamicColoredNoiseUnit()
    MIX_NOISE = AffineTransform()  # Project 2 colored noise sources to n_chans sensors
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def configure(self) -> None:
        self.COORDS.apply_settings(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch"))
        self.ALPHA_EXP.apply_settings(
            LinearTransformSettings(
                scale=[1.5 / 314, 1.5 / (2 * np.pi)],
                offset=[0.5, 0.5],
                axis="ch",
            )
        )
        self.CLIP_ALPHA.apply_settings(ClipSettings(min=0.0, max=2.0))
        self.PINK_NOISE.apply_settings(
            DynamicColoredNoiseSettings(
                output_fs=self.SETTINGS.output_fs,
                n_poles=5,
                smoothing_tau=0.01,
                initial_beta=1.0,
                scale=1.0,
                seed=self.SETTINGS.seed,
            )
        )
        rng = np.random.default_rng(self.SETTINGS.seed)
        ch_idx = np.arange(self.SETTINGS.output_ch)
        weights = np.array(
            [
                np.sin(2 * np.pi * ch_idx / self.SETTINGS.output_ch),  # radius source
                np.cos(2 * np.pi * ch_idx / self.SETTINGS.output_ch),  # angle (phi) source
            ]
        ) + 0.3 * rng.standard_normal((2, self.SETTINGS.output_ch))
        self.MIX_NOISE.apply_settings(AffineTransformSettings(weights=weights, axis="ch"))

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.COORDS.INPUT_SIGNAL),
            (self.COORDS.OUTPUT_SIGNAL, self.ALPHA_EXP.INPUT_SIGNAL),
            (self.ALPHA_EXP.OUTPUT_SIGNAL, self.CLIP_ALPHA.INPUT_SIGNAL),
            (self.CLIP_ALPHA.OUTPUT_SIGNAL, self.PINK_NOISE.INPUT_SIGNAL),
            (self.PINK_NOISE.OUTPUT_SIGNAL, self.MIX_NOISE.INPUT_SIGNAL),
            (self.MIX_NOISE.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
