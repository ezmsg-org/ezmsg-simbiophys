"""Convert polar velocity coordinates to simulated LFP-like colored noise.

This module provides a system that encodes velocity (in polar coordinates) into
the spectral properties of colored (1/f^beta) noise, producing LFP-like signals.

Pipeline:
    polar coords (magnitude, angle) -> cosine encoder (beta values) -> clip
                                    -> colored noise -> mix to channels

The velocity is encoded using a cosine tuning model where multiple noise
sources have different preferred directions. Each source's spectral exponent
(beta) is modulated by the velocity direction and magnitude. These sources
are then mixed across output channels using a spatial mixing matrix.

Note:
    This system expects polar coordinates as input. Use CoordinateSpaces with
    mode=CART2POL upstream to convert Cartesian velocity (vx, vy) to polar
    coordinates (magnitude, angle).

See Also:
    :mod:`ezmsg.simbiophys.system.velocity2spike`: Velocity to spike encoding.
    :mod:`ezmsg.simbiophys.system.velocity2ecephys`: Combined spike + LFP encoding.
"""

import ezmsg.core as ez
import numpy as np
from ezmsg.sigproc.affinetransform import AffineTransform, AffineTransformSettings
from ezmsg.sigproc.math.clip import Clip, ClipSettings
from ezmsg.util.messages.axisarray import AxisArray

from ..cosine_encoder import CosineEncoderSettings, CosineEncoderUnit
from ..dynamic_colored_noise import DynamicColoredNoiseSettings, DynamicColoredNoiseUnit


class Velocity2LFPSettings(ez.Settings):
    """Settings for :obj:`Velocity2LFP`."""

    output_fs: float = 30_000.0
    """Output sampling rate in Hz."""

    output_ch: int = 256
    """Number of output channels (simulated electrodes)."""

    n_lfp_sources: int = 8
    """Number of cosine-encoded LFP sources. Each source has a different
    preferred direction and generates colored noise with velocity-modulated
    spectral exponent."""

    max_velocity: float = 315.0

    seed: int = 6767
    """Random seed for reproducible preferred directions and mixing matrix."""


class Velocity2LFP(ez.Collection):
    """Encode velocity (polar coordinates) into LFP-like colored noise.

    This system converts polar velocity coordinates into multi-channel LFP-like signals:

    1. **Cosine encoder**: Each of n_lfp_sources has a different preferred
       direction. The spectral exponent beta (0-2) is modulated by the cosine
       of the angle between velocity and preferred direction, scaled by speed.
    2. **Clip**: Ensures beta values stay within valid range [0, 2].
    3. **Colored noise**: Generates 1/f^beta noise where beta is dynamically
       modulated per source.
    4. **Spatial mixing**: Projects the n_lfp_sources onto output_ch channels
       using a sinusoidal mixing matrix with random perturbations.

    Input:
        AxisArray with shape (N, 2) containing polar velocity coordinates.
        Dimension 0 is time, dimension 1 is [magnitude, angle].
        Use CoordinateSpaces(mode=CART2POL) upstream if starting from (vx, vy).

    Output:
        AxisArray with shape (M, output_ch) containing LFP-like colored noise
        at output_fs sampling rate.
    """

    SETTINGS = Velocity2LFPSettings

    # Polar velocity inputs (magnitude, angle)
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    BETA_ENCODER = CosineEncoderUnit()
    CLIP_BETA = Clip()
    PINK_NOISE = DynamicColoredNoiseUnit()
    MIX_NOISE = AffineTransform()  # Project n_lfp_sources to output_ch sensors
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def configure(self) -> None:
        # Input is polar coords: [magnitude, angle]
        # magnitude ranges from 0 to ~max_velocity px/s, angle from -pi to +pi

        # Configure cosine encoder to output beta values in range [0, 2]
        # baseline=1.0 (middle of range), modulation=1/315 so at max velocity we get full range
        self.BETA_ENCODER.apply_settings(
            CosineEncoderSettings(
                output_ch=self.SETTINGS.n_lfp_sources,
                baseline=1.0,
                modulation=1.0 / self.SETTINGS.max_velocity,
                seed=self.SETTINGS.seed,
            )
        )

        self.CLIP_BETA.apply_settings(ClipSettings(min=0.0, max=2.0))

        self.PINK_NOISE.apply_settings(
            DynamicColoredNoiseSettings(
                output_fs=self.SETTINGS.output_fs,
                n_poles=5,
                smoothing_tau=0.01,
                initial_beta=1.0,
                scale=20.0,
                seed=self.SETTINGS.seed,
            )
        )

        # Create mixing matrix: n_lfp_sources -> output_ch
        # Use sinusoids at different frequencies for spatial patterns
        rng = np.random.default_rng(self.SETTINGS.seed)
        ch_idx = np.arange(self.SETTINGS.output_ch)
        n_sources = self.SETTINGS.n_lfp_sources

        # Each source gets a sinusoidal spatial pattern with different frequency
        # Plus random perturbations for more realistic mixing
        weights = np.zeros((n_sources, self.SETTINGS.output_ch))
        for i in range(n_sources):
            # Different spatial frequency for each source
            freq = (i + 1) / n_sources
            phase = 2 * np.pi * i / n_sources
            weights[i, :] = np.sin(2 * np.pi * freq * ch_idx / self.SETTINGS.output_ch + phase)

        # Add random perturbations
        weights += 0.3 * rng.standard_normal((n_sources, self.SETTINGS.output_ch))

        self.MIX_NOISE.apply_settings(AffineTransformSettings(weights=weights, axis="ch"))

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.BETA_ENCODER.INPUT_SIGNAL),
            (self.BETA_ENCODER.OUTPUT_SIGNAL, self.CLIP_BETA.INPUT_SIGNAL),
            (self.CLIP_BETA.OUTPUT_SIGNAL, self.PINK_NOISE.INPUT_SIGNAL),
            (self.PINK_NOISE.OUTPUT_SIGNAL, self.MIX_NOISE.INPUT_SIGNAL),
            (self.MIX_NOISE.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
