"""Always-on slow 1/f (pink) baseline drift.

Adds a low-frequency, velocity-independent baseline wander to a signal so that
even at rest there is non-zero power at very low frequencies. This complements
:mod:`ezmsg.simbiophys.dynamic_colored_noise`, whose Kasdin filter runs at the
full output rate and therefore only holds a true 1/f slope down to a corner of
a few hundred Hz (at 30 kHz, ~700 Hz with 5 poles). Reaching sub-Hz drift that
way would need thousands of poles.

Instead we generate pink noise with a fixed spectral exponent (β = 1) on a
**low internal grid** (``drift_fs``, e.g. 50 Hz), where a handful of poles put
the 1/f corner well below 1 Hz, then linearly interpolate that slow signal up
to the output rate and add it. Because the drift is dominated by frequencies far
below the drift-grid Nyquist, linear interpolation is effectively exact.

The drift filter state, interpolation anchors, and phase are all carried across
chunks so the output is continuous and independent of how the stream is chunked.

Performance: the grid is generated on only ``drift_fs`` samples per second (a
handful per chunk), so the sequential IIR recursion is tiny. Upsampling to the
output rate, scaling, and adding to the input are fused into a single compiled
pass that touches the ``(n_samples, n_channels)`` result exactly once (see
:func:`_pink_drift_add`), which keeps the stage memory-bandwidth bound and
scales well to large channel counts on low-power CPUs.
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

from .dynamic_colored_noise import compute_kasdin_coefficients


def _generate_grid(white: np.ndarray, coeffs: np.ndarray, delay_lines: np.ndarray) -> np.ndarray:
    """Advance the Kasdin IIR one step per row of ``white`` (vectorised over channels).

    Used only for the one-off warm-up at reset (``white`` has a few hundred rows),
    so it stays in plain NumPy. ``delay_lines`` (shape ``(n_channels, n_poles)``)
    is mutated in place; returns the generated grid, shape ``(n_new, n_channels)``.
    """
    n_new, n_channels = white.shape
    n_poles = coeffs.shape[0]
    grid = np.empty((n_new, n_channels), dtype=np.float64)
    for k in range(n_new):
        # y[n] = x[n] - sum_p(a_p * y[n-p]); coeffs contracts the pole axis.
        y = white[k] - delay_lines @ coeffs
        # Shift delay lines (newest into slot 0); walk high->low to avoid aliasing.
        for p in range(n_poles - 1, 0, -1):
            delay_lines[:, p] = delay_lines[:, p - 1]
        delay_lines[:, 0] = y
        grid[k] = y
    return grid


@numba.jit(nopython=True, cache=True, fastmath=True)
def _pink_drift_add(
    out: np.ndarray,  # (n_out, n_channels) result, written in place: data + scale*drift
    data: np.ndarray,  # (n_out, n_channels) input signal
    white: np.ndarray,  # (n_new, n_channels) white noise for new grid samples
    coeffs: np.ndarray,  # (n_poles,) fixed Kasdin coefficients
    delay_lines: np.ndarray,  # (n_channels, n_poles) filter state, mutated in place
    prev: np.ndarray,  # (n_channels,) left interp anchor, mutated in place
    nxt: np.ndarray,  # (n_channels,) right interp anchor, mutated in place
    phase: float,  # fractional position between prev and nxt, in [0, 1)
    step: float,  # low-rate grid samples advanced per output sample
    scale: float,  # drift amplitude
) -> float:
    """Interpolate, scale, and add the slow pink drift in a single fused pass.

    Each output sample linearly interpolates the low-rate grid (``prev`` ->
    ``nxt``), scales it, and adds it to ``data``, writing ``out`` once. When
    ``phase`` crosses 1.0 a new grid sample is produced by advancing the Kasdin
    IIR one step. Returns the carried phase so the next chunk resumes seamlessly.
    """
    n_out = out.shape[0]
    n_channels = out.shape[1]
    n_poles = coeffs.shape[0]

    w = 0  # index into the pre-drawn white-noise grid
    ph = phase
    for i in range(n_out):
        # Hoist the interpolation weights so the channel loop is a plain axpy
        # (out = data + a*prev + b*nxt), which the compiler can vectorise (SIMD).
        a = scale * (1.0 - ph)
        b = scale * ph
        for c in range(n_channels):
            out[i, c] = data[i, c] + a * prev[c] + b * nxt[c]
        ph += step
        while ph >= 1.0:
            ph -= 1.0
            for c in range(n_channels):
                acc = 0.0
                for p in range(n_poles):
                    acc += coeffs[p] * delay_lines[c, p]
                y = white[w, c] - acc
                for p in range(n_poles - 1, 0, -1):
                    delay_lines[c, p] = delay_lines[c, p - 1]
                delay_lines[c, 0] = y
                prev[c] = nxt[c]
                nxt[c] = y
            w += 1
    return ph


class BaselineDriftSettings(ez.Settings):
    scale: float = 1.0
    """Amplitude of the added drift (multiplies the raw filter output)."""

    drift_fs: float = 50.0
    """Internal sample rate (Hz) at which the slow pink noise is generated.
    Lower rates push the 1/f corner lower for a given pole count; it only needs
    to sit comfortably above the drift frequencies of interest."""

    n_poles: int = 24
    """Number of IIR poles for the drift filter. With ``drift_fs`` = 50 Hz this
    places the 1/f corner near ~0.3 Hz, giving sub-Hz to few-Hz wander."""

    beta: float = 1.0
    """Spectral exponent of the drift. 1.0 is pink (1/f); kept < 2 for stability."""

    seed: int | None = None
    """Random seed for reproducibility. If None, uses system entropy."""


@processor_state
class BaselineDriftState:
    delay_lines: npt.NDArray[np.float64] | None = None
    """Kasdin filter delay lines, shape (n_channels, n_poles)."""

    coeffs: npt.NDArray[np.float64] | None = None
    """Fixed filter coefficients for ``beta``, shape (n_poles,)."""

    prev: npt.NDArray[np.float64] | None = None
    """Left interpolation anchor (current low-rate sample), shape (n_channels,)."""

    nxt: npt.NDArray[np.float64] | None = None
    """Right interpolation anchor (next low-rate sample), shape (n_channels,)."""

    phase: float = 0.0
    """Fractional position between ``prev`` and ``nxt`` for the next output sample."""

    rng: np.random.Generator | None = None
    """Random number generator."""

    step: float = 0.0
    """Low-rate grid samples advanced per output sample (drift_fs / output_fs)."""


class BaselineDriftTransformer(
    BaseStatefulTransformer[BaselineDriftSettings, AxisArray, AxisArray, BaselineDriftState]
):
    """Add always-on slow 1/f baseline drift to a signal.

    The drift is generated independently of the input values (it depends only on
    the input's shape and sample rate for timing), so it is present even when the
    upstream signal is at rest. One independent drift process is generated per
    channel of the input.
    """

    def _hash_message(self, message: AxisArray) -> int:
        time_axis = message.axes.get("time")
        gain = time_axis.gain if time_axis is not None else 0.0
        n_channels = message.data.shape[1] if message.data.ndim > 1 else 1
        return hash((n_channels, gain))

    def _reset_state(self, message: AxisArray) -> None:
        n_channels = message.data.shape[1] if message.data.ndim > 1 else 1

        self._state.rng = np.random.default_rng(self.settings.seed)
        self._state.coeffs = compute_kasdin_coefficients(self.settings.beta, self.settings.n_poles)
        self._state.delay_lines = np.zeros((n_channels, self.settings.n_poles), dtype=np.float64)

        time_axis = message.axes.get("time")
        output_gain = time_axis.gain if time_axis is not None else 1.0
        output_fs = 1.0 / output_gain
        self._state.step = self.settings.drift_fs / output_fs

        # Warm up the filter so the delay lines hold representative pink state
        # before the first output sample (avoids a startup transient from zero).
        # The last two warmup samples seed the interpolation anchors.
        n_warmup = 20 * self.settings.n_poles
        warm_white = self._state.rng.standard_normal((n_warmup + 2, n_channels))
        grid = _generate_grid(warm_white, self._state.coeffs, self._state.delay_lines)
        self._state.prev = grid[-2].copy()
        self._state.nxt = grid[-1].copy()
        self._state.phase = 0.0

    def _process(self, message: AxisArray) -> AxisArray:
        xp = get_namespace(message.data)
        data = np.asarray(message.data, dtype=np.float64)

        was_1d = data.ndim == 1
        if was_1d:
            data = data[:, np.newaxis]

        n_out, n_channels = data.shape

        # Number of new grid samples this chunk consumes. Carrying phase across
        # chunks makes this exact and chunk-order independent.
        n_steps = int(np.floor(self._state.phase + n_out * self._state.step))
        white = self._state.rng.standard_normal((n_steps, n_channels))

        # Fused interpolate + scale + add, writing the result in a single pass.
        out = np.empty_like(data)
        self._state.phase = _pink_drift_add(
            out,
            data,
            white,
            self._state.coeffs,
            self._state.delay_lines,  # mutated in place
            self._state.prev,  # mutated in place
            self._state.nxt,  # mutated in place
            self._state.phase,
            self._state.step,
            self.settings.scale,
        )

        if was_1d:
            out = out[:, 0]
        out_data = xp.asarray(out) if xp is not np else out
        return replace(message, data=out_data)


class BaselineDriftUnit(
    BaseTransformerUnit[
        BaselineDriftSettings,
        AxisArray,
        AxisArray,
        BaselineDriftTransformer,
    ]
):
    """Unit wrapper for BaselineDriftTransformer."""

    SETTINGS = BaselineDriftSettings
