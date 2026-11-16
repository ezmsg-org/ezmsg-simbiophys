"""What each stateful stage here treats as "a different stream".

Two facts these simulators now rely on, neither of which anything else notices
if it stops being true:

* Every producer declares :attr:`AxisArray.stream_dim`, so a downstream processor
  knows which dimension grows rather than guessing ``"time"``, and hands its
  channel axis over with the fingerprint already computed, so nobody downstream
  pays the checksum on every message.
* The stages that hold per-channel state let the axis-aware default hash decide
  when to reset, and the two that genuinely do not hold per-channel state say so
  explicitly. Getting that backwards is silent either way: too eager and a
  simulated population is redrawn mid-run, too lazy and one channel's drift is
  handed to another.
"""

import dataclasses
import logging
import sys

import ezmsg.baseproc.stateful
import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, compute_config_id

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
from ezmsg.simbiophys.noise import PinkNoiseProducer, PinkNoiseSettings, WhiteNoiseProducer, WhiteNoiseSettings
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
        stream_dim=dim,
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

    def test_declares_the_stream_dim(self, name, cls, settings, fs):
        """Without this a windowing stage downstream has to guess, and ``"time"``
        is present-but-wrong the moment the message becomes ``(win, time, ch)``."""
        assert self._produce(cls, settings, fs).stream_dim == "time"

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
        """The stream dimension is excluded, so ordinary chunk-size jitter must
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

    def test_it_reads_the_declared_stream_dim_not_the_name_time(self):
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


# name, producer, settings, changes that alter the layout, a change that keeps it
STAMPED = [
    (
        "white_noise",
        WhiteNoiseProducer,
        WhiteNoiseSettings(n_ch=4, fs=100.0, n_time=10),
        [{"n_ch": 5}, {"fs": 200.0}],
        {"scale": 2.0},
    ),
    # A generator followed by a filter: the filter must carry the generator's id and segment through.
    (
        "pink_noise",
        PinkNoiseProducer,
        PinkNoiseSettings(n_ch=4, fs=100.0, n_time=10, cutoff=20.0),
        [{"n_ch": 5}, {"fs": 200.0}],
        {"scale": 2.0},
    ),
    ("spiral", SpiralProducer, SpiralGeneratorSettings(fs=100.0, n_time=10), [{"fs": 200.0}], {"r_mean": 1.0}),
    (
        "sin",
        SinProducer,
        SinGeneratorSettings(n_ch=4, fs=100.0, n_time=10),
        [{"n_ch": 5}, {"fs": 200.0}],
        {"freq": 2.0},
    ),
    (
        "dnss_lfp",
        DNSSLFPProducer,
        DNSSLFPSettings(fs=100.0, n_time=10),
        [{"n_ch": 5}, {"fs": 200.0}],
        {"mode": "pedestal_norm"},
    ),
    ("dnss_spike", DNSSSpikeProducer, DNSSSpikeSettings(n_ch=4, n_time=10), [{"n_ch": 5}], {"mode": "ideal"}),
]

# Neither 1 / fs nor n_time / fs for any setting above: a template built from the clock tick
# would carry this gain into its config_id while every message says 1 / fs.
TICK = 0.03


def run(producer, n=4):
    return [producer(AxisArray.LinearAxis(gain=TICK, offset=i * TICK)) for i in range(n)]


def generator_class(cls):
    # PinkNoise is WhiteNoise followed by a filter; the generator stage stamps the id.
    return WhiteNoiseProducer if cls is PinkNoiseProducer else cls


@pytest.mark.parametrize("name,cls,settings,layout_changes,same_layout", STAMPED, ids=[p[0] for p in STAMPED])
class TestProducersStampConfigIdAndSegment:
    def _stream(self, cls, settings, layout_changes):
        """Messages per configuration: the initial one, then after each layout change."""
        producer = cls(settings)
        streams = [run(producer)]
        for change in layout_changes:
            settings = dataclasses.replace(settings, **change)
            producer.update_settings(settings)
            streams.append(run(producer))
        return streams

    def test_one_declared_id_per_layout(self, name, cls, settings, layout_changes, same_layout, monkeypatch):
        module = sys.modules[generator_class(cls).__module__]
        calls = []
        monkeypatch.setattr(module, "compute_config_id", lambda m: calls.append(m) or compute_config_id(m))

        streams = self._stream(cls, settings, layout_changes)
        for msgs in streams:
            fs = 1 / msgs[0].axes["time"].gain
            assert fs in (settings.fs, *(c.get("fs") for c in layout_changes))
            assert len({m.config_id for m in msgs}) == 1
            for m in msgs:
                assert m.config_id and m.config_id == compute_config_id(m)
        assert len({msgs[0].config_id for msgs in streams}) == len(streams)
        # Once per configuration, not per message.
        assert len(calls) == len(streams)

    def test_output_time_axis_is_the_sample_rate_not_the_tick(self, name, cls, settings, layout_changes, same_layout):
        assert run(cls(settings))[0].axes["time"].gain == pytest.approx(1 / settings.fs)

    def test_a_change_that_keeps_the_layout_keeps_the_id(self, name, cls, settings, layout_changes, same_layout):
        producer = cls(settings)
        before = run(producer)
        producer.update_settings(dataclasses.replace(settings, **same_layout))
        after = run(producer)
        assert after[0].config_id == before[0].config_id
        # It still resets, which is a discontinuity in the signal.
        assert {m.segment for m in before} == {0}
        assert {m.segment for m in after} == {1}

    def test_each_reset_starts_the_next_segment(self, name, cls, settings, layout_changes, same_layout):
        streams = self._stream(cls, settings, layout_changes)
        assert [{m.segment for m in msgs} for msgs in streams] == [{i} for i in range(len(streams))]

    def test_two_fresh_instances_both_start_at_segment_0(self, name, cls, settings, layout_changes, same_layout):
        assert {m.segment for m in run(cls(settings)) + run(cls(settings))} == {0}

    def test_the_output_check_finds_nothing_to_correct(
        self, name, cls, settings, layout_changes, same_layout, monkeypatch, caplog
    ):
        monkeypatch.setattr(ezmsg.baseproc.stateful, "_warned_unannounced", set())
        with caplog.at_level(logging.WARNING, logger="ezmsg.baseproc.stateful"):
            streams = self._stream(cls, settings, layout_changes)
        assert not caplog.records
        assert all(m.config_id == compute_config_id(m) for msgs in streams for m in msgs)
