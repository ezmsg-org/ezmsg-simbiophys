"""Generic cosine-tuning encoder for polar coordinates.

This module provides a generalized cosine-tuning encoder that maps polar
coordinates (magnitude, angle) to multiple output channels with configurable
preferred directions, baseline, and modulation parameters.

The encoding formula is:
    output = baseline + modulation * magnitude * cos(angle - preferred_direction)
             + speed_modulation * magnitude

This implements the offset model from "Decoding arm speed during reaching"
(https://ncbi.nlm.nih.gov/pmc/articles/PMC6286377/) with generic terminology
suitable for various applications:
    - Neural firing rate encoding (baseline=10Hz, modulation=20Hz)
    - LFP spectral parameter modulation (baseline=1.0, modulation=0.5)
    - Any other cosine-tuning based encoding

Input:
    Polar coordinates (magnitude, angle) as AxisArray with shape (n_samples, 2).
    Use CoordinateSpaces(mode=CART2POL) upstream to convert from Cartesian.

Output:
    AxisArray with shape (n_samples, output_ch) containing encoded values.
"""

from pathlib import Path

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class CosineEncoderSettings(ez.Settings):
    """Settings for CosineEncoder.

    Either `model_file` OR the random generation parameters should be specified.
    If `model_file` is provided, parameters are loaded from file.
    Otherwise, parameters are randomly generated.
    """

    # File-based parameters
    model_file: str | None = None
    """Path to .npz file with encoder parameters (baseline, modulation, pd, speed_modulation).
    Also supports legacy neural tuning files with keys (b0, m, pd, bs)."""

    # Random generation parameters
    output_ch: int = 10
    """Number of output channels (used if model_file is None)."""

    baseline: float = 0.0
    """Baseline output value for all channels (used if model_file is None)."""

    modulation: float = 1.0
    """Directional modulation depth for all channels (used if model_file is None)."""

    speed_modulation: float = 0.0
    """Speed modulation (non-directional) for all channels (used if model_file is None)."""

    seed: int | None = None
    """Random seed for reproducibility of preferred directions (used if model_file is None)."""


@processor_state
class CosineEncoderState:
    """State for cosine encoder transformer.

    Holds the per-channel encoding parameters. All arrays have shape (1, output_ch)
    for efficient broadcasting during processing.

    Attributes:
        baseline: Baseline output value for each channel.
        modulation: Directional modulation depth for each channel.
        pd: Preferred direction (radians) for each channel.
        speed_modulation: Speed modulation (non-directional) for each channel.
        ch_axis: Pre-built channel axis for output messages.
    """

    baseline: npt.NDArray[np.floating] | None = None
    modulation: npt.NDArray[np.floating] | None = None
    pd: npt.NDArray[np.floating] | None = None
    speed_modulation: npt.NDArray[np.floating] | None = None
    ch_axis: AxisArray.CoordinateAxis | None = None

    @property
    def output_ch(self) -> int:
        """Number of output channels."""
        return self.baseline.shape[1] if self.baseline is not None else 0

    def validate(self) -> None:
        """Validate that all parameters have consistent shapes."""
        if any(x is None for x in [self.baseline, self.modulation, self.pd, self.speed_modulation]):
            raise ValueError("All parameters must be set")
        if not (self.baseline.shape == self.modulation.shape == self.pd.shape == self.speed_modulation.shape):
            raise ValueError("All parameters must have the same shape")
        if self.baseline.ndim != 2 or self.baseline.shape[0] != 1:
            raise ValueError("Parameters must have shape (1, output_ch)")
        if self.baseline.shape[1] < 1:
            raise ValueError("Parameters must have at least 1 channel")

    def load_from_file(
        self,
        filepath: str | Path,
        output_ch: int | None = None,
    ) -> None:
        """Load parameters from a .npz file.

        The file should contain arrays with keys matching the parameter names.
        For backwards compatibility with neural tuning files, the following
        key mappings are supported:
            - 'b0' -> baseline
            - 'm' -> modulation
            - 'pd' -> pd (preferred direction)
            - 'bs' -> speed_modulation

        Args:
            filepath: Path to .npz file containing parameter arrays.
            output_ch: Number of channels to use. If None, uses all in file.
        """
        params = np.load(filepath)

        # Support both new names and legacy neural tuning names
        baseline = np.asarray(params.get("baseline", params.get("b0")), dtype=np.float64).ravel()
        modulation = np.asarray(params.get("modulation", params.get("m")), dtype=np.float64).ravel()
        pd = np.asarray(params["pd"], dtype=np.float64).ravel()
        speed_modulation = np.asarray(params.get("speed_modulation", params.get("bs")), dtype=np.float64).ravel()

        if output_ch is not None:
            baseline = baseline[:output_ch]
            modulation = modulation[:output_ch]
            pd = pd[:output_ch]
            speed_modulation = speed_modulation[:output_ch]

        # Reshape to (1, output_ch) for broadcasting
        self.baseline = baseline[np.newaxis, :]
        self.modulation = modulation[np.newaxis, :]
        self.pd = pd[np.newaxis, :]
        self.speed_modulation = speed_modulation[np.newaxis, :]

        # Create channel axis for output messages
        ch_labels = np.array([f"ch{i}" for i in range(len(baseline))])
        self.ch_axis = AxisArray.CoordinateAxis(data=ch_labels, dims=["ch"])

        self.validate()

    def init_random(
        self,
        output_ch: int,
        baseline: float = 0.0,
        modulation: float = 1.0,
        speed_modulation: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """Initialize encoder parameters with random preferred directions.

        Args:
            output_ch: Number of output channels.
            baseline: Baseline value for all channels.
            modulation: Directional modulation depth for all channels.
            speed_modulation: Speed modulation (non-directional) for all channels.
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)

        # Shape (1, output_ch) for efficient broadcasting
        self.baseline = np.full((1, output_ch), baseline, dtype=np.float64)
        self.modulation = np.full((1, output_ch), modulation, dtype=np.float64)
        self.pd = rng.uniform(0.0, 2.0 * np.pi, size=(1, output_ch)).astype(np.float64)
        self.speed_modulation = np.full((1, output_ch), speed_modulation, dtype=np.float64)

        # Create channel axis for output messages
        ch_labels = np.array([f"ch{i}" for i in range(output_ch)])
        self.ch_axis = AxisArray.CoordinateAxis(data=ch_labels, dims=["ch"])

        self.validate()


class CosineEncoderTransformer(
    BaseStatefulTransformer[CosineEncoderSettings, AxisArray, AxisArray, CosineEncoderState]
):
    """Transform polar coordinates to multi-channel encoded output.

    Input: AxisArray with shape (n_samples, 2) containing polar coordinates
           (magnitude, angle) where magnitude is speed and angle is direction.
    Output: AxisArray with shape (n_samples, output_ch) containing encoded values.

    The encoding formula is:
        output = baseline + modulation * magnitude * cos(angle - pd)
                 + speed_modulation * magnitude

    This is a generic encoder suitable for various applications including:
    - Neural firing rate encoding (baseline=10Hz, modulation=20Hz)
    - LFP spectral parameter modulation (baseline=1.0, modulation=0.5)
    - Any other cosine-tuning based encoding
    """

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize encoder parameters."""
        if self.settings.model_file is not None:
            self.state.load_from_file(
                self.settings.model_file,
                output_ch=None,  # Use all channels from file
            )
        else:
            self.state.init_random(
                output_ch=self.settings.output_ch,
                baseline=self.settings.baseline,
                modulation=self.settings.modulation,
                speed_modulation=self.settings.speed_modulation,
                seed=self.settings.seed,
            )

    def _process(self, message: AxisArray) -> AxisArray:
        """Transform polar coordinates to encoded output."""
        polar = np.asarray(message.data, dtype=np.float64)

        if polar.ndim != 2 or polar.shape[1] != 2:
            raise ValueError(f"Expected polar coords with shape (n_samples, 2), got {polar.shape}")

        # Extract polar components (from CART2POL: magnitude, angle)
        magnitude = polar[:, 0:1]  # (n_samples, 1)
        angle = polar[:, 1:2]  # (n_samples, 1)

        # Compute output: baseline + modulation * magnitude * cos(angle - pd) + speed_mod * magnitude
        # State arrays are pre-shaped to (1, output_ch) for broadcasting
        output = (
            self.state.baseline
            + self.state.modulation * magnitude * np.cos(angle - self.state.pd)
            + self.state.speed_modulation * magnitude
        )

        return replace(
            message,
            data=output,
            dims=["time", "ch"],
            axes={**message.axes, "ch": self.state.ch_axis},
        )


class CosineEncoderUnit(BaseTransformerUnit[CosineEncoderSettings, AxisArray, AxisArray, CosineEncoderTransformer]):
    """Unit wrapper for CosineEncoderTransformer."""

    SETTINGS = CosineEncoderSettings
