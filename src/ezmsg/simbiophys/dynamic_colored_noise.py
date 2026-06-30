"""Dynamic colored noise generation with time-varying spectral exponent.

Generates 1/f^β noise where β can change dynamically based on input.
Uses the Kasdin IIR filter method with stateful processing for continuity
across chunks.

Reference:
    N. Jeremy Kasdin, "Discrete Simulation of Colored Noise and Stochastic
    Processes and 1/f^α Power Law Noise Generation," Proceedings of the IEEE,
    Vol. 83, No. 5, May 1995, pages 802-827.
"""

import ezmsg.core as ez
import numba
import numpy as np
import numpy.typing as npt
from array_api_compat import get_namespace
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


def compute_kasdin_coefficients_batch(betas: npt.NDArray, n_poles: int) -> npt.NDArray[np.float64]:
    """Compute IIR filter coefficients for multiple β values simultaneously.

    Vectorized version of :func:`compute_kasdin_coefficients` that processes
    an array of beta values in one call.

    Args:
        betas: Array of spectral exponents, shape ``(n_betas,)``.
        n_poles: Number of IIR filter poles.

    Returns:
        Array of shape ``(n_betas, n_poles)`` containing filter coefficients.
    """
    betas = np.asarray(betas, dtype=np.float64)
    a = np.ones(len(betas), dtype=np.float64)
    half_betas = betas / 2.0
    coeffs = np.zeros((len(betas), n_poles), dtype=np.float64)
    for k in range(1, n_poles + 1):
        a *= (k - 1 - half_betas) / k
        coeffs[:, k - 1] = a
    return coeffs


@numba.jit(nopython=True, cache=True)
def _kasdin_filter_loop(
    output: np.ndarray,  # (n_output, n_channels) pre-zeroed, filled in place
    white_noise: np.ndarray,  # (n_output, n_channels)
    target_coeffs_all: np.ndarray,  # (n_input, n_channels, n_poles)
    finite_mask: np.ndarray,  # (n_input, n_channels) bool
    delay_lines: np.ndarray,  # (n_channels, n_poles) state, mutated in place
    coeffs: np.ndarray,  # (n_channels, n_poles) state, mutated in place
    alpha: float,
    samples_per_bin: float,
    sample_remainder: float,
    n_output_samples: int,
) -> None:
    """Run the time-varying Kasdin IIR recursion (compiled hot loop).

    This is a verbatim, channel-decomposed port of the per-sample loop that
    used to live in :meth:`DynamicColoredNoiseTransformer._process`. The
    recursion is inherently sequential (``output[i]`` feeds the delay line that
    produces ``output[i+1]``), so it cannot be vectorized over time; numba
    compiles the scalar loop to native code instead. Channels are independent,
    so iterating over them inside the sample loop is numerically identical to
    the original vectorized-over-channels form.

    For non-finite β channels the original smoothed the coefficients toward
    themselves (an EMA no-op); here we simply skip the update, which is
    bit-for-bit equivalent.
    """
    n_input_samples = target_coeffs_all.shape[0]
    n_channels = coeffs.shape[0]
    n_poles = coeffs.shape[1]

    cumulative_samples = sample_remainder
    for bin_idx in range(n_input_samples):
        next_cumulative = cumulative_samples + samples_per_bin
        bin_start_out = int(cumulative_samples)
        bin_end_out = min(int(next_cumulative), n_output_samples)

        if bin_end_out <= bin_start_out:
            cumulative_samples = next_cumulative
            continue

        for i in range(bin_start_out, bin_end_out):
            for c in range(n_channels):
                # Exponentially smooth coefficients toward this bin's target.
                # Non-finite β -> no update (matches EMA-toward-self no-op).
                if finite_mask[bin_idx, c]:
                    for p in range(n_poles):
                        coeffs[c, p] += alpha * (target_coeffs_all[bin_idx, c, p] - coeffs[c, p])

                # IIR filter: y[n] = x[n] - sum_p(a_k * y[n-k])
                acc = 0.0
                for p in range(n_poles):
                    acc += coeffs[c, p] * delay_lines[c, p]
                y = white_noise[i, c] - acc
                output[i, c] = y

                # Shift delay lines (newest sample into slot 0)
                for p in range(n_poles - 1, 0, -1):
                    delay_lines[c, p] = delay_lines[c, p - 1]
                delay_lines[c, 0] = y

        cumulative_samples = next_cumulative


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
    delay_lines: npt.NDArray[np.float64] | None = None
    """Filter delay lines, shape (n_channels, n_poles)."""

    coeffs: npt.NDArray[np.float64] | None = None
    """Current filter coefficients, shape (n_channels, n_poles)."""

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
        n_channels = message.data.shape[1] if message.data.ndim > 1 else 1

        # Initialize RNG
        self._state.rng = np.random.default_rng(self.settings.seed)

        # Initialize vectorized filter state
        initial_coeffs = compute_kasdin_coefficients(self.settings.initial_beta, self.settings.n_poles)
        self._state.delay_lines = np.zeros((n_channels, self.settings.n_poles), dtype=np.float64)
        self._state.coeffs = np.tile(initial_coeffs, (n_channels, 1))

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
        xp = get_namespace(message.data)
        beta_data = np.asarray(message.data, dtype=np.float64)

        # Handle 1D input
        was_1d = beta_data.ndim == 1
        if was_1d:
            beta_data = beta_data[:, np.newaxis]

        n_input_samples, n_channels = beta_data.shape
        n_poles = self.settings.n_poles

        # Get precomputed timing parameters from state
        output_gain = self._state.output_gain
        samples_per_bin = self._state.samples_per_bin
        alpha = self._state.alpha

        # Get offset from message for output
        time_axis = message.axes.get("time")
        input_offset = time_axis.offset if time_axis is not None else 0.0

        # Calculate total output samples for this chunk
        total_fractional = n_input_samples * samples_per_bin + self._state.sample_remainder
        n_output_samples = int(total_fractional)
        new_remainder = total_fractional - n_output_samples

        if n_output_samples == 0:
            # Not enough input to produce output yet
            self._state.sample_remainder = total_fractional
            empty_data = np.zeros((0, n_channels), dtype=np.float64)
            if was_1d:
                empty_data = empty_data[:, 0]
            out_data = xp.asarray(empty_data) if xp is not np else empty_data
            return replace(
                message,
                data=out_data,
                dims=message.dims,
                axes={
                    **message.axes,
                    "time": replace(time_axis, gain=output_gain, offset=input_offset),
                },
            )

        # Precompute target coefficients for all bins in batch.
        # For non-finite betas, we'll use current coeffs (EMA no-op).
        finite_mask = np.isfinite(beta_data)  # (n_input, n_channels)

        # Collect all unique finite betas and batch-compute their coefficients
        all_betas_flat = beta_data.ravel()  # (n_input * n_channels,)
        finite_flat = finite_mask.ravel()
        # Build target coefficients array: (n_input, n_channels, n_poles)
        target_coeffs_all = np.zeros((n_input_samples, n_channels, n_poles), dtype=np.float64)

        if np.any(finite_flat):
            finite_betas = all_betas_flat[finite_flat]
            finite_coeffs = compute_kasdin_coefficients_batch(finite_betas, n_poles)
            # Scatter back into the full array
            target_coeffs_all.reshape(-1, n_poles)[finite_flat] = finite_coeffs

        # Generate white noise for all output samples
        white_noise = self._state.rng.standard_normal((n_output_samples, n_channels))
        output = np.zeros_like(white_noise)

        # Run the time-varying Kasdin IIR recursion. The state arrays
        # (delay_lines, coeffs) are mutated in place so continuity is preserved
        # across chunks. The loop is sequential per channel, so it is compiled
        # with numba rather than vectorized (see _kasdin_filter_loop).
        _kasdin_filter_loop(
            output,
            white_noise,
            target_coeffs_all,
            finite_mask,
            self._state.delay_lines,  # (n_channels, n_poles), mutated in place
            self._state.coeffs,  # (n_channels, n_poles), mutated in place
            alpha,
            samples_per_bin,
            self._state.sample_remainder,
            n_output_samples,
        )

        # Update remainder for next chunk
        self._state.sample_remainder = new_remainder

        # Apply scaling
        output *= self.settings.scale

        # Handle 1D output if input was 1D
        if was_1d:
            output = output[:, 0]

        # Convert to input's array backend
        out_data = xp.asarray(output) if xp is not np else output

        return replace(
            message,
            data=out_data,
            dims=["time"] if was_1d else ["time", "ch"],
            axes={
                **message.axes,
                "time": replace(time_axis, gain=output_gain, offset=input_offset),
            },
        )


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
