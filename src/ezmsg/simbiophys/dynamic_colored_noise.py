"""Dynamic colored noise generation with time-varying spectral exponent.

Generates 1/f^β noise where β can change dynamically based on input.
Uses the Kasdin IIR filter method with stateful processing for continuity
across chunks.

Reference:
    N. Jeremy Kasdin, "Discrete Simulation of Colored Noise and Stochastic
    Processes and 1/f^α Power Law Noise Generation," Proceedings of the IEEE,
    Vol. 83, No. 5, May 1995, pages 802-827.
"""

from dataclasses import dataclass, field

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


def compute_kasdin_coefficients(beta: float, n_poles: int) -> npt.NDArray[np.float64]:
    """Compute IIR filter coefficients for 1/f^β noise using Kasdin's method.

    The coefficients are computed using the recurrence relation:
        a₀ = 1
        aₖ = (k - 1 - β/2) · aₖ₋₁ / k    for k = 1, 2, ..., n_poles

    Args:
        beta: Spectral exponent. Common values:
            - 0: white noise
            - 1: pink noise (1/f)
            - 2: brown/red noise (1/f²)
            TODO: jit / vectorize over multiple values of beta
        n_poles: Number of IIR filter poles. More poles extend accuracy to
            lower frequencies but increase computation. 5 is a reasonable default.

    Returns:
        Array of shape (n_poles,) containing filter coefficients a₁ through aₙ.
    """
    coeffs = np.zeros(n_poles, dtype=np.float64)
    a = 1.0
    for k in range(1, n_poles + 1):
        a = (k - 1 - beta / 2) * a / k
        coeffs[k - 1] = a
    return coeffs


@dataclass
class ColoredNoiseFilterState:
    """State for a single channel's colored noise filter."""

    delay_line: npt.NDArray[np.float64]
    """Previous output samples (filter memory)."""

    coeffs: npt.NDArray[np.float64]
    """Current filter coefficients (exponentially smoothed)."""


class DynamicColoredNoiseSettings(ez.Settings):
    output_fs: float | None = None
    """Output sampling rate in Hz. If None, output rate matches input rate."""

    n_poles: int = 5
    """Number of IIR filter poles. More poles extend accuracy to
    lower frequencies. Default 5 provides good balance."""

    smoothing_tau: float = 0.01
    """Time constant (in seconds) for exponential smoothing of
    coefficient changes. Coefficients reach ~63% of target in τ seconds,
    ~95% in 3τ seconds. Set to 0 for instantaneous changes (no smoothing).
    Default 0.01 (10ms) provides smooth transitions without sluggishness."""

    initial_beta: float = 1.0
    """Initial spectral exponent before any input is received."""

    scale: float = 1.0
    """Output amplitude scaling factor."""

    seed: int | None = None
    """Random seed for reproducibility. If None, uses system entropy."""


@processor_state
class DynamicColoredNoiseState:
    filter_states: list[ColoredNoiseFilterState] = field(default_factory=list)
    """Per-channel filter states."""

    rng: np.random.Generator | None = None
    """Random number generator."""

    sample_remainder: float = 0.0
    """Fractional sample accumulator for resampling."""

    output_gain: float = 0.0
    """Time step between output samples (1/output_fs)."""

    samples_per_bin: float = 1.0
    """Number of output samples per input sample."""

    alpha: float = 1.0
    """Exponential smoothing factor for coefficient updates."""


class DynamicColoredNoiseTransformer(
    BaseStatefulTransformer[DynamicColoredNoiseSettings, AxisArray, AxisArray, DynamicColoredNoiseState]
):
    """Transform spectral exponent (β) input into colored noise output.

    Input: AxisArray with β values. Shape can be:
        - (n_samples,) for single-channel output
        - (n_samples, n_channels) for multi-channel output

    Output: AxisArray with colored noise having spectral density ~ 1/f^β.
        If output_fs is set, output will be resampled to that rate.

    The transformer maintains filter state across chunks to ensure continuity
    (no discontinuities at chunk boundaries). When β changes, coefficients
    are exponentially smoothed with time constant `smoothing_tau` to avoid
    transients.

    Each input β sample is treated as a "bin" - all output samples generated
    within that bin use that β value as the target. Coefficient smoothing
    handles transitions between bins.

    Example:
        >>> settings = DynamicColoredNoiseSettings(
        ...     output_fs=30000.0,  # 30 kHz output
        ...     n_poles=5,
        ...     smoothing_tau=0.01,  # 10ms time constant
        ...     initial_beta=1.0
        ... )
        >>> transformer = DynamicColoredNoiseTransformer(settings)
        >>> # Input: β values at 100 Hz
        >>> beta_input = AxisArray(np.full((100, 2), 1.5), dims=["time", "ch"], ...)
        >>> noise_output = transformer(beta_input)  # Output at 30 kHz
    """

    def _hash_message(self, message: AxisArray) -> int:
        """Hash based on number of channels and sample rate to detect stream changes."""
        time_axis = message.axes.get("time")
        # LinearAxis has gain (1/fs) rather than fs directly
        gain = time_axis.gain if time_axis is not None else 0.0
        # Number of channels is dim 1 for 2D data, or 1 for 1D data
        n_channels = message.data.shape[1] if message.data.ndim > 1 else 1
        return hash((n_channels, gain))

    def _reset_state(self, message: AxisArray) -> None:
        """Initialize filter states and compute timing parameters."""
        # Determine number of channels
        if message.data.ndim == 1:
            n_channels = 1
        else:
            n_channels = message.data.shape[1] if message.data.ndim > 1 else 1

        # Initialize RNG
        self._state.rng = np.random.default_rng(self.settings.seed)

        # Initialize filter states for each channel
        initial_coeffs = compute_kasdin_coefficients(self.settings.initial_beta, self.settings.n_poles)

        self._state.filter_states = []
        for _ in range(n_channels):
            self._state.filter_states.append(
                ColoredNoiseFilterState(
                    delay_line=np.zeros(self.settings.n_poles, dtype=np.float64),
                    coeffs=initial_coeffs.copy(),
                )
            )

        # Compute timing parameters (only depends on sample rate, not data)
        time_axis = message.axes.get("time")
        input_gain = time_axis.gain if time_axis is not None else 1.0
        input_fs = 1.0 / input_gain

        output_fs = self.settings.output_fs if self.settings.output_fs else input_fs
        self._state.output_gain = 1.0 / output_fs
        self._state.samples_per_bin = input_gain * output_fs

        # Compute exponential smoothing factor
        tau = self.settings.smoothing_tau
        if tau > 0:
            self._state.alpha = 1.0 - np.exp(-self._state.output_gain / tau)
        else:
            self._state.alpha = 1.0  # Instantaneous (no smoothing)

        # Initialize resampling state
        self._state.sample_remainder = 0.0

    def _process(self, message: AxisArray) -> AxisArray:
        """Generate colored noise based on input β values."""
        beta_data = np.asarray(message.data, dtype=np.float64)

        # Handle 1D input
        was_1d = beta_data.ndim == 1
        if was_1d:
            beta_data = beta_data[:, np.newaxis]

        n_input_samples, n_channels = beta_data.shape

        # Get precomputed timing parameters from state
        output_gain = self._state.output_gain
        samples_per_bin = self._state.samples_per_bin
        alpha = self._state.alpha

        # Get offset from message for output
        time_axis = message.axes.get("time")
        input_offset = time_axis.offset if time_axis is not None else 0.0

        # Calculate total output samples for this chunk
        # Use remainder from previous chunk for continuity
        total_fractional = n_input_samples * samples_per_bin + self._state.sample_remainder
        n_output_samples = int(total_fractional)
        new_remainder = total_fractional - n_output_samples

        if n_output_samples == 0:
            # Not enough input to produce output yet
            self._state.sample_remainder = total_fractional
            # Return empty output
            empty_data = np.zeros((0, n_channels), dtype=np.float64)
            if was_1d:
                empty_data = empty_data[:, 0]
            return replace(
                message,
                data=empty_data,
                dims=message.dims,
                axes={
                    **message.axes,
                    "time": replace(time_axis, gain=output_gain, offset=input_offset),
                },
            )

        # Generate white noise for all output samples
        white_noise = self._state.rng.standard_normal((n_output_samples, n_channels))
        output = np.zeros_like(white_noise)

        # Process each channel
        for ch in range(n_channels):
            output[:, ch] = self._process_resampled(
                white_noise[:, ch],
                beta_data[:, ch],
                self._state.filter_states[ch],
                samples_per_bin,
                self._state.sample_remainder,
                alpha,
            )

        # Update remainder for next chunk
        self._state.sample_remainder = new_remainder

        # Apply scaling
        output *= self.settings.scale

        # Handle 1D output if input was 1D
        if was_1d:
            output = output[:, 0]

        return replace(
            message,
            data=output,
            dims=["time"] if was_1d else ["time", "ch"],
            axes={
                **message.axes,
                "time": replace(time_axis, gain=output_gain, offset=input_offset),
            },
        )

    def _process_resampled(
        self,
        white_noise: npt.NDArray[np.float64],
        beta_values: npt.NDArray[np.float64],
        fs: ColoredNoiseFilterState,
        samples_per_bin: float,
        initial_remainder: float,
        alpha: float,
    ) -> npt.NDArray[np.float64]:
        """Process with resampling - each input β defines a bin of output samples.

        Args:
            white_noise: Pre-generated white noise for output samples.
            beta_values: Input β values (one per input bin).
            fs: Filter state for this channel.
            samples_per_bin: Number of output samples per input bin (may be fractional).
            initial_remainder: Fractional sample offset from previous chunk.
            alpha: Exponential smoothing factor (1 - exp(-dt/tau)).

        Returns:
            Colored noise output samples.
        """
        n_output = len(white_noise)
        n_input = len(beta_values)
        output = np.zeros(n_output, dtype=np.float64)
        n_poles = self.settings.n_poles

        # Track fractional position for bin boundaries
        cumulative_samples = initial_remainder

        for bin_idx in range(n_input):
            # Calculate how many output samples this bin contributes
            next_cumulative = cumulative_samples + samples_per_bin
            bin_start_out = int(cumulative_samples)
            bin_end_out = min(int(next_cumulative), n_output)

            if bin_end_out <= bin_start_out:
                # This bin doesn't contribute any complete samples yet
                cumulative_samples = next_cumulative
                continue

            # Get target coefficients for this bin's β
            beta = beta_values[bin_idx]
            # Handle nan/inf beta values by using current coefficients (no update)
            if np.isfinite(beta):
                target_coeffs = compute_kasdin_coefficients(beta, n_poles)
            else:
                target_coeffs = fs.coeffs

            # Generate output samples for this bin
            for i in range(bin_start_out, bin_end_out):
                if i >= n_output:
                    break

                # Exponentially smooth coefficients toward target
                fs.coeffs += alpha * (target_coeffs - fs.coeffs)

                # IIR filter: x[n] = w[n] - sum(a_k * x[n-k])
                output[i] = white_noise[i] - np.dot(fs.coeffs, fs.delay_line)

                # Update delay line
                fs.delay_line = np.roll(fs.delay_line, 1)
                fs.delay_line[0] = output[i]

            cumulative_samples = next_cumulative

        return output


class DynamicColoredNoiseUnit(
    BaseTransformerUnit[
        DynamicColoredNoiseSettings,
        AxisArray,
        AxisArray,
        DynamicColoredNoiseTransformer,
    ]
):
    """Unit wrapper for DynamicColoredNoiseTransformer."""

    SETTINGS = DynamicColoredNoiseSettings
