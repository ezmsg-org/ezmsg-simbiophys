"""Cosine tuning model for neural encoding of velocity/movement.

Implements the offset model from "Decoding arm speed during reaching"
(https://ncbi.nlm.nih.gov/pmc/articles/PMC6286377/):

    firing_rate = b0 + m * |v| * cos(θ - θ_pd) + bs * |v|

Where:
    - b0: baseline firing rate
    - m: directional modulation depth
    - θ: velocity direction (angle)
    - θ_pd: preferred direction
    - bs: speed modulation (non-directional)
    - |v|: velocity magnitude (speed)

For spike generation from firing rates, use EventsFromRatesTransformer
from ezmsg-event.
"""

from dataclasses import dataclass
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


@dataclass
class CosineTuningParams:
    """Parameters for cosine tuning model.

    All arrays must have the same shape (n_units,).

    Attributes:
        b0: Baseline firing rate (Hz) for each unit.
        m: Directional modulation depth for each unit.
        pd: Preferred direction (radians) for each unit.
        bs: Speed modulation (non-directional) for each unit.
    """

    b0: npt.NDArray[np.floating]
    m: npt.NDArray[np.floating]
    pd: npt.NDArray[np.floating]
    bs: npt.NDArray[np.floating]

    def __post_init__(self):
        """Validate that all parameters have consistent shapes."""
        if not (self.b0.shape == self.m.shape == self.pd.shape == self.bs.shape):
            raise ValueError("All parameters must have the same shape")
        if self.b0.ndim != 1:
            raise ValueError("Parameters must be 1D arrays")
        if len(self.b0) < 1:
            raise ValueError("Parameters must have length >= 1")

    @property
    def n_units(self) -> int:
        """Number of neural units."""
        return len(self.b0)

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        n_units: int | None = None,
        weight_gain: float = 1.0,
    ) -> "CosineTuningParams":
        """Load parameters from a .npz file.

        Args:
            filepath: Path to .npz file containing b0, m, pd, bs arrays.
            n_units: Number of units to use. If None, uses all units in file.
            weight_gain: Scaling factor applied to m and bs parameters.

        Returns:
            CosineTuningParams instance.
        """
        params = np.load(filepath)

        b0 = np.asarray(params["b0"], dtype=np.float64)
        m = np.asarray(params["m"], dtype=np.float64)
        pd = np.asarray(params["pd"], dtype=np.float64)
        bs = np.asarray(params["bs"], dtype=np.float64)

        if n_units is not None:
            b0 = b0[:n_units]
            m = m[:n_units]
            pd = pd[:n_units]
            bs = bs[:n_units]

        m = m * weight_gain
        bs = bs * weight_gain

        return cls(b0=b0, m=m, pd=pd, bs=bs)

    @classmethod
    def from_random(
        cls,
        n_units: int,
        baseline_hz: float = 10.0,
        modulation_hz: float = 20.0,
        speed_modulation_hz: float = 0.0,
        seed: int | None = None,
    ) -> "CosineTuningParams":
        """Generate random tuning parameters.

        Args:
            n_units: Number of neural units.
            baseline_hz: Baseline firing rate (Hz) for all units.
            modulation_hz: Directional modulation depth for all units.
            speed_modulation_hz: Speed modulation (non-directional) for all units.
            seed: Random seed for reproducibility.

        Returns:
            CosineTuningParams instance with random preferred directions.
        """
        rng = np.random.default_rng(seed)

        return cls(
            b0=np.full(n_units, baseline_hz, dtype=np.float64),
            m=np.full(n_units, modulation_hz, dtype=np.float64),
            pd=rng.uniform(0.0, 2.0 * np.pi, size=n_units).astype(np.float64),
            bs=np.full(n_units, speed_modulation_hz, dtype=np.float64),
        )


class CosineTuningSettings(ez.Settings):
    """Settings for CosineTuningTransformer.

    Either `model_file` OR the random generation parameters should be specified.
    If `model_file` is provided, parameters are loaded from file.
    Otherwise, parameters are randomly generated.
    """

    # File-based parameters
    model_file: str | None = None
    """Path to .npz file with tuning parameters (b0, m, pd, bs)."""

    weight_gain: float = 1.0
    """Scaling factor for m and bs when loading from file."""

    # Random generation parameters
    n_units: int = 50
    """Number of neural units (used if model_file is None)."""

    baseline_hz: float = 10.0
    """Baseline firing rate in Hz (used if model_file is None)."""

    modulation_hz: float = 20.0
    """Directional modulation depth in Hz (used if model_file is None)."""

    speed_modulation_hz: float = 0.0
    """Speed modulation (non-directional) in Hz (used if model_file is None)."""

    seed: int | None = None
    """Random seed for reproducibility (used if model_file is None)."""

    # Output settings
    min_rate: float = 0.0
    """Minimum firing rate (Hz). Rates are clipped to this value."""


@processor_state
class CosineTuningState:
    """State for cosine tuning transformer."""

    params: CosineTuningParams | None = None
    """Tuning curve parameters."""


class CosineTuningTransformer(BaseStatefulTransformer[CosineTuningSettings, AxisArray, AxisArray, CosineTuningState]):
    """Transform 2D velocity into firing rates using cosine tuning model.

    Input: AxisArray with shape (n_samples, 2) containing velocity (vx, vy).
    Output: AxisArray with shape (n_samples, n_units) containing firing rates (Hz).

    The model implements:
        rate = b0 + m * |v| * cos(θ - θ_pd) + bs * |v|

    For spike generation, chain with EventsFromRatesTransformer from ezmsg-event.
    """

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize tuning parameters."""
        if self.settings.model_file is not None:
            self.state.params = CosineTuningParams.from_file(
                self.settings.model_file,
                n_units=None,  # Use all units from file
                weight_gain=self.settings.weight_gain,
            )
        else:
            self.state.params = CosineTuningParams.from_random(
                n_units=self.settings.n_units,
                baseline_hz=self.settings.baseline_hz,
                modulation_hz=self.settings.modulation_hz,
                speed_modulation_hz=self.settings.speed_modulation_hz,
                seed=self.settings.seed,
            )

    def _process(self, message: AxisArray) -> AxisArray:
        """Transform velocity to firing rates."""
        v = np.asarray(message.data, dtype=np.float64)

        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(f"Expected velocity with shape (n_samples, 2), got {v.shape}")

        # Extract velocity components
        vx = v[:, 0]
        vy = v[:, 1]

        # Calculate speed (magnitude) and direction (angle)
        speed = np.hypot(vx, vy)[:, np.newaxis]  # (n_samples, 1)
        theta = np.arctan2(vy, vx)[:, np.newaxis]  # (n_samples, 1)

        # Get parameters as row vectors for broadcasting
        params = self.state.params
        b0 = params.b0[np.newaxis, :]  # (1, n_units)
        m = params.m[np.newaxis, :]  # (1, n_units)
        pd = params.pd[np.newaxis, :]  # (1, n_units)
        bs = params.bs[np.newaxis, :]  # (1, n_units)

        # Compute firing rates: b0 + m * |v| * cos(θ - θ_pd) + bs * |v|
        rates = b0 + m * speed * np.cos(theta - pd) + bs * speed

        # Clip to minimum rate
        rates = np.maximum(rates, self.settings.min_rate)

        # Create channel axis
        ch_labels = np.array([f"unit{i}" for i in range(params.n_units)])
        ch_axis = AxisArray.CoordinateAxis(data=ch_labels, dims=["ch"])

        return replace(
            message,
            data=rates,
            dims=["time", "ch"],
            axes={**message.axes, "ch": ch_axis},
        )


class CosineTuningUnit(BaseTransformerUnit[CosineTuningSettings, AxisArray, AxisArray, CosineTuningTransformer]):
    """Unit wrapper for CosineTuningTransformer."""

    SETTINGS = CosineTuningSettings
