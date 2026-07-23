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

from ..baseline_drift import BaselineDriftSettings, BaselineDriftUnit
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

    drift_scale: float = 8.0
    """Amplitude of always-on slow 1/f baseline drift added per output channel.
    This wander is velocity-independent, so low-frequency power (baseline drift)
    is present even at rest. Set to 0 to disable. Roughly ~15% of the per-channel
    LFP std at the default settings."""

    drift_fs: float = 50.0
    """Internal generation rate (Hz) for the baseline drift. Lower rates push the
    1/f corner to lower frequencies (slower wander) for a given pole count."""

    seed: int = 6767
    """Random seed for reproducible preferred directions and mixing matrix."""


class Velocity2LFP(ez.Collection):
    """Encode velocity (polar coordinates) into LFP-like colored noise.

    This system converts polar velocity coordinates into multi-channel LFP-like signals:

    1. **Cosine encoder**: Each of n_lfp_sources has a different preferred
       direction. The spectral exponent beta (0-1.95) is modulated by the cosine
       of the angle between velocity and preferred direction, scaled by speed.
    2. **Clip**: Ensures beta values stay within valid range [0, 1.95]. The
       ceiling is kept just below 2.0 to keep the noise filter stable (see
       ``configure``).
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
    INPUT_SIGNAL = ez.InputTopic(AxisArray)
    BETA_ENCODER = CosineEncoderUnit()
    CLIP_BETA = Clip()
    PINK_NOISE = DynamicColoredNoiseUnit()
    MIX_NOISE = AffineTransform()  # Project n_lfp_sources to output_ch sensors
    BASELINE_DRIFT = BaselineDriftUnit()  # Always-on slow 1/f wander per channel
    OUTPUT_SIGNAL = ez.OutputTopic(AxisArray)

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

        # Cap strictly below 2.0. At beta == 2 the Kasdin all-pole filter has a
        # pole exactly on the unit circle (y[n] = x[n] + y[n-1], a pure
        # integrator), so its delay-line state does an unbounded random walk.
        # When sustained high velocity pins beta at the ceiling the state grows
        # large, and a subsequent beta change unleashes it as a big transient
        # followed by ringing. 1.95 keeps the pole inside the unit circle.
        self.CLIP_BETA.apply_settings(ClipSettings(min=0.0, max=1.95))

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

        # Create mixing matrix factory: n_lfp_sources -> output_ch
        # Using a callable so the weights are rebuilt automatically if the
        # number of input sources changes at runtime (e.g. via live settings).
        output_ch = self.SETTINGS.output_ch
        seed = self.SETTINGS.seed

        def make_mixing_weights(n_in: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            ch_idx = np.arange(output_ch)
            weights = np.zeros((n_in, output_ch))
            for i in range(n_in):
                freq = (i + 1) / n_in
                phase = 2 * np.pi * i / n_in
                weights[i, :] = np.sin(2 * np.pi * freq * ch_idx / output_ch + phase)
            weights += 0.3 * rng.standard_normal((n_in, output_ch))
            return weights

        self.MIX_NOISE.apply_settings(AffineTransformSettings(weights=make_mixing_weights, axis="ch"))

        # Add always-on slow 1/f drift per output channel. The main filter runs
        # at output_fs and so cannot produce sub-Hz power (its 1/f corner is a few
        # hundred Hz); this generates the low-frequency wander separately.
        self.BASELINE_DRIFT.apply_settings(
            BaselineDriftSettings(
                scale=self.SETTINGS.drift_scale,
                drift_fs=self.SETTINGS.drift_fs,
                beta=1.0,
                seed=self.SETTINGS.seed,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.BETA_ENCODER.INPUT_SIGNAL),
            (self.BETA_ENCODER.OUTPUT_SIGNAL, self.CLIP_BETA.INPUT_SIGNAL),
            (self.CLIP_BETA.OUTPUT_SIGNAL, self.PINK_NOISE.INPUT_SIGNAL),
            (self.PINK_NOISE.OUTPUT_SIGNAL, self.MIX_NOISE.INPUT_SIGNAL),
            (self.MIX_NOISE.OUTPUT_SIGNAL, self.BASELINE_DRIFT.INPUT_SIGNAL),
            (self.BASELINE_DRIFT.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
