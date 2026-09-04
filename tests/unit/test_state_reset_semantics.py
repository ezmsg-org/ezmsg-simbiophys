"""What each stateful stage here treats as "a different stream".

Two facts these simulators now rely on, neither of which anything else notices
if it stops being true:

* Every producer declares :attr:`AxisArray.chunk_dim`, so a downstream processor
  knows which dimension grows rather than guessing ``"time"``, and hands its
  channel axis over with the fingerprint already computed, so nobody downstream
  pays the checksum on every message.
* The stages that hold per-channel state let the axis-aware default hash decide
  when to reset, and the two that genuinely do not hold per-channel state say so
  explicitly. Getting that backwards is silent either way: too eager and a
  simulated population is redrawn mid-run, too lazy and one channel's drift is
  handed to another.
"""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.simbiophys import (
    BaselineDriftSettings,
    BaselineDriftTransformer,
    CosineEncoderSettings,
    CosineEncoderTransformer,
    DynamicColoredNoiseSettings,
    DynamicColoredNoiseTransformer,
    LineNoiseSettings,
    LineNoiseTransformer,
)
from ezmsg.simbiophys.dnss.lfp import DNSSLFPProducer, DNSSLFPSettings
from ezmsg.simbiophys.dnss.spike import DNSSSpikeProducer, DNSSSpikeSettings
from ezmsg.simbiophys.noise import WhiteNoiseProducer, WhiteNoiseSettings
from ezmsg.simbiophys.oscillator import (
    SinGeneratorSettings,
    SinProducer,
    SpiralGeneratorSettings,
    SpiralProducer,
)


def signal(labels: list[str], fs: float = 100.0, n_time: int = 16, dim: str = "time") -> AxisArray:
    return AxisArray(
        np.zeros((n_time, len(labels))),
        dims=[dim, "ch"],
        axes={
            dim: AxisArray.TimeAxis(fs=fs),
            "ch": CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key="dev",
        chunk_dim=dim,
    )


PRODUCERS = [
    ("white_noise", WhiteNoiseProducer, WhiteNoiseSettings(n_ch=4, fs=100.0), 100.0),
    ("spiral", SpiralProducer, SpiralGeneratorSettings(fs=100.0), 100.0),
    ("sin", SinProducer, SinGeneratorSettings(n_ch=4, fs=100.0), 100.0),
    ("dnss_lfp", DNSSLFPProducer, DNSSLFPSettings(fs=100.0), 100.0),
    # Spike patterns are tabulated at 30 kHz and cannot be resampled.
    ("dnss_spike", DNSSSpikeProducer, DNSSSpikeSettings(fs=30000.0), 30000.0),
]


@pytest.mark.parametrize("name,cls,settings,fs", PRODUCERS, ids=[p[0] for p in PRODUCERS])
class TestProducersDescribeTheirStream:
    def _produce(self, cls, settings, fs):
        producer = cls(settings)
        time_axis = AxisArray.TimeAxis(fs=fs, offset=0.0)
        producer._reset_state(time_axis)
        return producer._produce(10, time_axis)

    def test_declares_the_chunk_dim(self, name, cls, settings, fs):
        """Without this a windowing stage downstream has to guess, and ``"time"``
        is present-but-wrong the moment the message becomes ``(win, time, ch)``."""
        assert self._produce(cls, settings, fs).chunk_dim == "time"

    def test_hands_over_a_primed_channel_axis(self, name, cls, settings, fs):
        """The axis is built once per stream, so one checksum covers every
        message and every consumer -- including across a process boundary, where
        unpickling otherwise hands out a cold axis per message, forever."""
        msg = self._produce(cls, settings, fs)
        cold = sorted(
            dim
            for dim, axis in msg.axes.items()
            if isinstance(axis, CoordinateAxis) and "_fingerprint" not in axis.__dict__
        )
        assert not cold, f"axes published without a fingerprint: {cold}"


class TestPerChannelStateResetsOnRelabel:
    """Same channel *count*, different channels. The delay lines and drift
    anchors are one-per-channel, so reusing them hands each new channel the
    history of whoever used to sit in its column."""

    @pytest.mark.parametrize(
        "cls,settings",
        [
            (BaselineDriftTransformer, BaselineDriftSettings(seed=0)),
            (DynamicColoredNoiseTransformer, DynamicColoredNoiseSettings(seed=0)),
        ],
        ids=["baseline_drift", "dynamic_colored_noise"],
    )
    def test_a_relabel_at_a_fixed_count_is_a_new_stream(self, cls, settings):
        proc = cls(settings)
        assert proc._hash_message(signal(["a", "b"])) != proc._hash_message(signal(["c", "d"]))

    @pytest.mark.parametrize(
        "cls,settings",
        [
            (BaselineDriftTransformer, BaselineDriftSettings(seed=0)),
            (DynamicColoredNoiseTransformer, DynamicColoredNoiseSettings(seed=0)),
        ],
        ids=["baseline_drift", "dynamic_colored_noise"],
    )
    def test_a_longer_chunk_is_the_same_stream(self, cls, settings):
        """The chunk dimension is excluded, so ordinary chunk-size jitter must
        not throw away a warmed-up filter."""
        proc = cls(settings)
        assert proc._hash_message(signal(["a", "b"], n_time=16)) == proc._hash_message(signal(["a", "b"], n_time=64))


class TestLineNoiseIsCommonMode:
    """Its state is all ``(1, 1)`` and broadcasts, so it deliberately ignores
    what the default hash cares most about."""

    def test_a_relabel_does_not_restart_the_phase(self):
        proc = LineNoiseTransformer(LineNoiseSettings(freq=60.0))
        assert proc._hash_message(signal(["a", "b"])) == proc._hash_message(signal(["c", "d", "e"]))

    def test_a_sample_rate_change_does(self):
        proc = LineNoiseTransformer(LineNoiseSettings(freq=60.0))
        assert proc._hash_message(signal(["a", "b"], fs=100.0)) != proc._hash_message(signal(["a", "b"], fs=500.0))

    def test_it_reads_the_declared_chunk_dim_not_the_name_time(self):
        """A stream that grows along ``samp`` has no ``time`` axis at all; the
        old hard-coded lookup silently fell back to a period of zero."""
        proc = LineNoiseTransformer(LineNoiseSettings(freq=60.0))
        slow = signal(["a", "b"], fs=100.0, dim="samp")
        fast = signal(["a", "b"], fs=500.0, dim="samp")
        assert proc._hash_message(slow) != proc._hash_message(fast)

        proc(slow)
        assert proc.state.dt == pytest.approx(1 / 100.0)


class TestCosineEncoderKeepsItsPopulation:
    """The tuning parameters come from settings -- a file, or a seeded draw --
    and never from the message."""

    def test_an_upstream_relabel_does_not_redraw(self):
        # seed=None deliberately: with a fixed seed a redraw reproduces the same
        # preferred directions and the assertion below could not tell the two
        # apart. Here a reset is visible.
        proc = CosineEncoderTransformer(CosineEncoderSettings(output_ch=8, seed=None))
        polar = signal(["mag", "ang"])
        proc(polar)
        pd_before = np.array(proc.state.pd, copy=True)

        proc(signal(["magnitude", "angle"]))
        np.testing.assert_array_equal(proc.state.pd, pd_before)

    def test_the_output_channel_axis_is_primed_and_reused(self):
        proc = CosineEncoderTransformer(CosineEncoderSettings(output_ch=8, seed=42))
        first = proc(signal(["mag", "ang"]))
        assert "_fingerprint" in first.axes["ch"].__dict__
        # The same object every message, which is what lets a downstream
        # processor settle its own hash on one pointer comparison.
        assert proc(signal(["mag", "ang"])).axes["ch"] is first.axes["ch"]
