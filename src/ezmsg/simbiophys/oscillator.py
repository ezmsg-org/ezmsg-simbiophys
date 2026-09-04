"""Oscillator/sinusoidal signal generators."""

import numpy as np
import numpy.typing as npt
from ezmsg.baseproc import (
    BaseClockDrivenProducer,
    BaseClockDrivenUnit,
    ClockDrivenSettings,
    ClockDrivenState,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis, replace


def freq_drift_step_std(drift_rate_per_sec: float, dt: float) -> float:
    """Per-sample random-walk std that yields a given drift rate.

    A frequency doing a zero-mean random walk with per-sample increment std
    ``s`` drifts with RMS ``s * sqrt(T/dt)`` over an interval ``T``. Choosing
    ``s = drift_rate_per_sec * sqrt(dt)`` makes the RMS drift over 1 s equal to
    ``drift_rate_per_sec`` (Hz/s), independent of sample rate or chunking.
    """
    return drift_rate_per_sec * np.sqrt(dt)


def advance_drifting_sine(
    n_samples: int,
    dt: float,
    ang_freq: np.ndarray,
    amp: np.ndarray,
    phase_state: np.ndarray,
    freq_off_state: np.ndarray,
    step_std: float,
    bound_hz: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a sinusoid whose frequency slowly wanders, via phase accumulation.

    The instantaneous frequency is ``base + offset`` where ``offset`` does a
    bounded random walk (clamped to ``+/- bound_hz``). Phase is integrated
    sample-by-sample so the waveform stays continuous even as the frequency
    changes and across chunk boundaries.

    Args:
        n_samples: Number of output samples for this chunk.
        dt: Sample period (seconds).
        ang_freq: Base angular frequency ``2*pi*f``, shape ``(1, k)``.
        amp: Amplitude, shape ``(1, k)``.
        phase_state: Carried phase from the previous chunk, shape ``(1, k)``.
        freq_off_state: Carried frequency offset (Hz), shape ``(1, k)``.
        step_std: Per-sample random-walk std (see :func:`freq_drift_step_std`);
            0 disables drift (fixed frequency).
        bound_hz: Frequency offset is clamped to ``+/- bound_hz``.
        rng: Random generator for the walk.

    Returns:
        ``(data, new_phase_state, new_freq_off_state)`` where ``data`` has shape
        ``(n_samples, k)`` and the two states have shape ``(1, k)``.
    """
    k = ang_freq.shape[1]
    if n_samples == 0:
        return np.zeros((0, k)), phase_state, freq_off_state

    if step_std > 0.0:
        inc = rng.standard_normal((n_samples, k)) * step_std
        freq_off = freq_off_state + np.cumsum(inc, axis=0)
        np.clip(freq_off, -bound_hz, bound_hz, out=freq_off)
    else:
        freq_off = np.zeros((n_samples, k))

    inst_ang = ang_freq + 2.0 * np.pi * freq_off  # (n_samples, k)
    phase = phase_state + np.cumsum(inst_ang * dt, axis=0)
    data = amp * np.sin(phase)
    new_phase_state = np.mod(phase[-1:, :], 2.0 * np.pi)
    new_freq_off_state = freq_off[-1:, :].copy()
    return data, new_phase_state, new_freq_off_state


class SpiralGeneratorSettings(ClockDrivenSettings):
    """Settings for :obj:`SpiralGenerator`.

    Generates 2D position (x, y) following a spiral pattern where both
    the radius and angle change over time.

    The parametric equations are:
        r(t) = r_mean + r_amp * sin(2*π*radial_freq*t + radial_phase)
        θ(t) = 2*π*angular_freq*t + angular_phase
        x(t) = r(t) * cos(θ(t))
        y(t) = r(t) * sin(θ(t))
    """

    r_mean: float = 150.0
    """Mean radius of the spiral."""

    r_amp: float = 50.0
    """Amplitude of the radial oscillation."""

    radial_freq: float = 0.1
    """Frequency of the radial oscillation in Hz."""

    radial_phase: float = 0.0
    """Initial phase of the radial oscillation in radians."""

    angular_freq: float = 0.25
    """Frequency of the angular rotation in Hz."""

    angular_phase: float = 0.0
    """Initial angular phase in radians."""


@processor_state
class SpiralGeneratorState(ClockDrivenState):
    """State for SpiralGenerator."""

    template: AxisArray | None = None


class SpiralProducer(BaseClockDrivenProducer[SpiralGeneratorSettings, SpiralGeneratorState]):
    """
    Generates spiral motion synchronized to clock ticks.

    Each clock tick produces a block of 2D position data (x, y) following
    a spiral pattern where both radius and angle change over time.
    """

    def _reset_state(self, time_axis: LinearAxis) -> None:
        """Initialize template."""
        ch_axis = AxisArray.CoordinateAxis(data=np.array(["x", "y"]), dims=["ch"])
        # Primed once for the stream -- see noise.py for why.
        ch_axis.fingerprint
        self._state.template = AxisArray(
            data=np.zeros((0, 2)),
            dims=["time", "ch"],
            axes={"time": time_axis, "ch": ch_axis},
            chunk_dim="time",
        )

    def _produce(self, n_samples: int, time_axis: LinearAxis) -> AxisArray:
        """Generate spiral motion for this chunk."""
        t = (np.arange(n_samples) + self._state.counter) * time_axis.gain

        # Radial component: oscillates between r_mean - r_amp and r_mean + r_amp
        r = self.settings.r_mean + self.settings.r_amp * np.sin(
            2.0 * np.pi * self.settings.radial_freq * t + self.settings.radial_phase
        )

        # Angular component: rotates at angular_freq
        theta = 2.0 * np.pi * self.settings.angular_freq * t + self.settings.angular_phase

        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        data = np.column_stack([x, y])

        return replace(
            self._state.template,
            data=data,
            axes={
                **self._state.template.axes,
                "time": time_axis,
            },
        )


class SpiralGenerator(BaseClockDrivenUnit[SpiralGeneratorSettings, SpiralProducer]):
    """
    Generates 2D spiral motion synchronized to clock ticks.

    Receives timing from INPUT_CLOCK (LinearAxis from Clock) and outputs
    2D position AxisArray (x, y) on OUTPUT_SIGNAL.

    The spiral pattern has both radius and angle varying over time:
    - Radius oscillates sinusoidally (breathing in/out)
    - Angle increases linearly (rotation)
    """

    SETTINGS = SpiralGeneratorSettings


class SinGeneratorSettings(ClockDrivenSettings):
    """Settings for :obj:`SinGenerator`."""

    n_ch: int = 1
    """Number of channels to output."""

    freq: float | npt.ArrayLike = 1.0
    """The frequency of the sinusoid, in Hz. Scalar or per-channel array."""

    amp: float | npt.ArrayLike = 1.0
    """The amplitude of the sinusoid. Scalar or per-channel array."""

    phase: float | npt.ArrayLike = 0.0
    """The initial phase of the sinusoid, in radians. Scalar or per-channel array."""

    freq_drift_rate: float = 0.0
    """Frequency drift rate in Hz per second (RMS wander over 1 s). When > 0 the
    frequency does a slow bounded random walk to emulate e.g. recording-clock
    drift. 0 (default) keeps the frequency fixed."""

    freq_drift_bound: float = 1.5
    """The drifting frequency is clamped to ``+/- freq_drift_bound`` Hz around the
    base ``freq``."""

    freq_drift_seed: int | None = None
    """Random seed for the frequency-drift walk. If None, uses system entropy."""


@processor_state
class SinGeneratorState(ClockDrivenState):
    """State for SinGenerator."""

    template: AxisArray | None = None
    # Pre-computed arrays for efficient processing, shape (1, 1) or (1, n_ch)
    ang_freq: np.ndarray | None = None  # 2*pi*freq
    amp: np.ndarray | None = None
    phase: np.ndarray | None = None
    # Frequency-drift state (only used when freq_drift_rate > 0)
    drift_rng: np.random.Generator | None = None
    drift_phase: np.ndarray | None = None  # accumulated phase, shape (1, k)
    drift_freq_off: np.ndarray | None = None  # frequency offset (Hz), shape (1, k)
    drift_step_std: float = 0.0


class SinProducer(BaseClockDrivenProducer[SinGeneratorSettings, SinGeneratorState]):
    """
    Generates sinusoidal waveforms synchronized to clock ticks.

    Each clock tick produces a block of sinusoidal data based on the
    sample rate (fs) and chunk size (n_time) settings.
    """

    def _reset_state(self, time_axis: LinearAxis) -> None:
        """Initialize template and pre-compute parameter arrays."""
        n_ch = self.settings.n_ch

        # Create template
        ch_axis = AxisArray.CoordinateAxis(data=np.arange(n_ch), dims=["ch"])
        # Primed once for the stream -- see noise.py for why.
        ch_axis.fingerprint
        self._state.template = AxisArray(
            data=np.zeros((0, n_ch)),
            dims=["time", "ch"],
            axes={"time": time_axis, "ch": ch_axis},
            chunk_dim="time",
        )

        # Convert settings to arrays and validate
        freq = np.atleast_1d(self.settings.freq)
        amp = np.atleast_1d(self.settings.amp)
        phase = np.atleast_1d(self.settings.phase)

        for name, arr in [("freq", freq), ("amp", amp), ("phase", phase)]:
            if arr.size > 1 and arr.size != n_ch:
                raise ValueError(
                    f"{name} has length {arr.size} but n_ch is {n_ch}. "
                    f"Per-channel arrays must have length equal to n_ch."
                )

        # Reshape for broadcasting: (1, n_ch) or (1, 1)
        freq = freq.reshape(1, -1) if freq.size > 1 else freq.reshape(1, 1)
        amp = amp.reshape(1, -1) if amp.size > 1 else amp.reshape(1, 1)
        phase = phase.reshape(1, -1) if phase.size > 1 else phase.reshape(1, 1)

        # Store pre-computed values
        self._state.ang_freq = 2.0 * np.pi * freq
        self._state.amp = amp
        self._state.phase = phase

        # Frequency-drift state. When drift is enabled we switch to phase
        # accumulation (in _produce) to keep the waveform continuous as the
        # frequency wanders; when disabled the fast closed-form path is used.
        if self.settings.freq_drift_rate > 0.0:
            self._state.drift_rng = np.random.default_rng(self.settings.freq_drift_seed)
            self._state.drift_phase = np.array(self._state.phase, dtype=np.float64).reshape(1, -1)
            self._state.drift_freq_off = np.zeros_like(self._state.drift_phase)
            self._state.drift_step_std = freq_drift_step_std(self.settings.freq_drift_rate, time_axis.gain)

    def _produce(self, n_samples: int, time_axis: LinearAxis) -> AxisArray:
        """Generate sinusoidal waveform for this chunk."""
        if self.settings.freq_drift_rate > 0.0:
            sin_data, self._state.drift_phase, self._state.drift_freq_off = advance_drifting_sine(
                n_samples,
                time_axis.gain,
                self._state.ang_freq,
                self._state.amp,
                self._state.drift_phase,
                self._state.drift_freq_off,
                self._state.drift_step_std,
                self.settings.freq_drift_bound,
                self._state.drift_rng,
            )
        else:
            # Fast closed-form path: amp * sin(ang_freq*t + phase)
            # t shape: (n_time,) -> (n_time, 1) for broadcasting with (1, n_ch)
            t = (np.arange(n_samples) + self._state.counter)[:, np.newaxis] * time_axis.gain
            sin_data = self._state.amp * np.sin(self._state.ang_freq * t + self._state.phase)

        # Tile if all params were scalar but n_ch > 1
        if sin_data.shape[1] < self.settings.n_ch:
            sin_data = np.tile(sin_data, (1, self.settings.n_ch))

        return replace(
            self._state.template,
            data=sin_data,
            axes={
                **self._state.template.axes,
                "time": time_axis,
            },
        )


class SinGenerator(BaseClockDrivenUnit[SinGeneratorSettings, SinProducer]):
    """
    Generates sinusoidal waveforms synchronized to clock ticks.

    Receives timing from INPUT_CLOCK (LinearAxis from Clock) and outputs
    sinusoidal AxisArray on OUTPUT_SIGNAL.
    """

    SETTINGS = SinGeneratorSettings
