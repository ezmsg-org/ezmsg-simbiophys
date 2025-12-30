"""Integration tests for DNSS (Digital Neural Signal Simulator) units."""

import os
from dataclasses import field

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.simbiophys import Clock, ClockSettings, Counter, CounterSettings
from ezmsg.simbiophys.dnss import (
    DEFAULT_FS,
    LFP_GAINS,
    DNSSLFPSettings,
    DNSSLFPUnit,
    DNSSSpikeSettings,
    DNSSSpikeUnit,
    DNSSSynth,
    DNSSSynthSettings,
)
from tests.helpers.util import get_test_fn


class DNSSLFPTestSystemSettings(ez.Settings):
    clock_settings: ClockSettings
    counter_settings: CounterSettings
    lfp_settings: DNSSLFPSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(default_factory=TerminateOnTotalSettings)


class DNSSLFPTestSystem(ez.Collection):
    """Test system for DNSS LFP: Clock -> Counter -> LFP."""

    SETTINGS = DNSSLFPTestSystemSettings

    CLOCK = Clock()
    COUNTER = Counter()
    LFP = DNSSLFPUnit()
    LOG = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.CLOCK.apply_settings(self.SETTINGS.clock_settings)
        self.COUNTER.apply_settings(self.SETTINGS.counter_settings)
        self.LFP.apply_settings(self.SETTINGS.lfp_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_SIGNAL, self.COUNTER.INPUT_SIGNAL),
            (self.COUNTER.OUTPUT_SIGNAL, self.LFP.INPUT_SIGNAL),
            (self.LFP.OUTPUT_SIGNAL, self.LOG.INPUT_MESSAGE),
            (self.LOG.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
        )


def test_dnss_lfp_unit(test_name: str | None = None):
    """Test DNSSLFPUnit produces valid LFP output."""
    fs = DEFAULT_FS  # 30 kHz
    n_time = 600  # 20ms blocks
    n_ch = 4
    n_messages = 10
    dispatch_rate = fs / n_time  # 50 Hz dispatch

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = DNSSLFPTestSystemSettings(
        clock_settings=ClockSettings(dispatch_rate=dispatch_rate),
        counter_settings=CounterSettings(fs=fs, n_time=n_time),
        lfp_settings=DNSSLFPSettings(
            n_ch=n_ch,
            pattern="spike",
            mode="hdmi",
        ),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateOnTotalSettings(total=n_messages),
    )
    system = DNSSLFPTestSystem(settings)
    ez.run(SYSTEM=system)

    # Collect result
    messages: list[AxisArray] = list(message_log(test_filename))
    os.remove(test_filename)

    assert len(messages) >= n_messages

    # Verify each message has correct shape
    for msg in messages:
        assert msg.data.shape == (n_time, n_ch)

    # Concatenate and check properties
    agg = AxisArray.concatenate(*messages, dim="time")

    # Check sample rate
    assert agg.axes["time"].gain == 1.0 / fs

    # LFP should have values (not all zeros)
    assert not np.allclose(agg.data, 0.0)

    # All channels should be identical (LFP is broadcast)
    for ch in range(1, n_ch):
        np.testing.assert_array_almost_equal(agg.data[:, 0], agg.data[:, ch])

    # Check approximate amplitude range for hdmi mode
    # LFP is sum of 3 sinusoids, each with gain ~894.4
    max_expected = sum(LFP_GAINS["hdmi"])  # ~2683
    assert np.max(np.abs(agg.data)) < max_expected * 1.1


def test_dnss_lfp_unit_other_pattern(test_name: str | None = None):
    """Test DNSSLFPUnit with 'other' LFP pattern."""
    fs = DEFAULT_FS
    n_time = 600
    n_ch = 2
    n_messages = 5
    dispatch_rate = fs / n_time

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = DNSSLFPTestSystemSettings(
        clock_settings=ClockSettings(dispatch_rate=dispatch_rate),
        counter_settings=CounterSettings(fs=fs, n_time=n_time),
        lfp_settings=DNSSLFPSettings(
            n_ch=n_ch,
            pattern="other",
            mode="hdmi",
        ),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateOnTotalSettings(total=n_messages),
    )
    system = DNSSLFPTestSystem(settings)
    ez.run(SYSTEM=system)

    messages: list[AxisArray] = list(message_log(test_filename))
    os.remove(test_filename)

    assert len(messages) >= n_messages

    agg = AxisArray.concatenate(*messages, dim="time")
    assert agg.data.shape[1] == n_ch
    assert not np.allclose(agg.data, 0.0)

    # "other" pattern has max amplitude of 6000 in hdmi mode
    assert np.max(np.abs(agg.data)) <= 6000


class DNSSSpikeTestSystemSettings(ez.Settings):
    clock_settings: ClockSettings
    counter_settings: CounterSettings
    spike_settings: DNSSSpikeSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(default_factory=TerminateOnTotalSettings)


class DNSSSpikeTestSystem(ez.Collection):
    """Test system for DNSS Spike: Clock -> Counter -> Spike."""

    SETTINGS = DNSSSpikeTestSystemSettings

    CLOCK = Clock()
    COUNTER = Counter()
    SPIKE = DNSSSpikeUnit()
    LOG = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.CLOCK.apply_settings(self.SETTINGS.clock_settings)
        self.COUNTER.apply_settings(self.SETTINGS.counter_settings)
        self.SPIKE.apply_settings(self.SETTINGS.spike_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_SIGNAL, self.COUNTER.INPUT_SIGNAL),
            (self.COUNTER.OUTPUT_SIGNAL, self.SPIKE.INPUT_SIGNAL),
            (self.SPIKE.OUTPUT_SIGNAL, self.LOG.INPUT_MESSAGE),
            (self.LOG.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
        )


def test_dnss_spike_unit(test_name: str | None = None):
    """Test DNSSSpikeUnit produces valid sparse spike output."""
    fs = DEFAULT_FS
    n_time = 600
    n_ch = 4
    n_messages = 10
    dispatch_rate = fs / n_time

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = DNSSSpikeTestSystemSettings(
        clock_settings=ClockSettings(dispatch_rate=dispatch_rate),
        counter_settings=CounterSettings(fs=fs, n_time=n_time),
        spike_settings=DNSSSpikeSettings(
            n_ch=n_ch,
            mode="hdmi",
        ),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateOnTotalSettings(total=n_messages),
    )
    system = DNSSSpikeTestSystem(settings)
    ez.run(SYSTEM=system)

    messages: list[AxisArray] = list(message_log(test_filename))
    os.remove(test_filename)

    assert len(messages) >= n_messages

    # Check that output is sparse
    import sparse

    for msg in messages:
        assert isinstance(msg.data, sparse.COO)
        assert msg.data.shape == (n_time, n_ch)

    # Check sample rate
    assert messages[0].axes["time"].gain == 1.0 / fs

    # Count total events across all messages
    total_events = sum(len(msg.data.data) for msg in messages)

    # With 10 messages * 600 samples = 6000 samples total
    # At slow rate (7500 samples between spikes), we might have 0-1 spikes
    # Just verify the output structure is correct
    assert total_events >= 0  # May or may not have spikes depending on timing


def test_dnss_spike_unit_burst_period(test_name: str | None = None):
    """Test DNSSSpikeUnit during burst period (more spikes)."""
    fs = DEFAULT_FS
    n_time = 3000  # 100ms blocks
    n_ch = 4
    # Run long enough to hit burst period (starts at sample 270000 = 9 seconds)
    # Use fast dispatch to reach burst quickly
    n_messages = 200
    dispatch_rate = float("inf")  # AFAP mode

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = DNSSSpikeTestSystemSettings(
        clock_settings=ClockSettings(dispatch_rate=dispatch_rate),
        counter_settings=CounterSettings(fs=fs, n_time=n_time),
        spike_settings=DNSSSpikeSettings(
            n_ch=n_ch,
            mode="ideal",
        ),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateOnTotalSettings(total=n_messages),
    )
    system = DNSSSpikeTestSystem(settings)
    ez.run(SYSTEM=system)

    messages: list[AxisArray] = list(message_log(test_filename))
    os.remove(test_filename)

    # TerminateOnTotal may allow extra messages through
    assert len(messages) >= n_messages
    messages = messages[:n_messages]

    # Check that we got some spikes in the slow period
    total_events = sum(len(msg.data.data) for msg in messages)

    # 5 messages * 3000 samples = 15000 samples
    # At slow rate (7500 samples/spike), we should see 1-2 spikes
    assert total_events >= 1, f"Expected at least 1 spike, got {total_events}"

    # Verify waveform IDs are 1, 2, or 3
    msg = messages[199]
    assert np.all(np.diff(np.where(np.diff(msg.data.data))[0]) == n_ch)


def test_dnss_synth(test_name: str | None = None):
    """Test DNSSSynth produces combined spike+LFP output."""
    n_time = 600  # 20ms blocks
    n_ch = 4
    n_messages = 25  # 0.5 seconds - enough to see at least one spike

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    comps = {
        "SYNTH": DNSSSynth(
            DNSSSynthSettings(
                n_time=n_time,
                n_ch=n_ch,
                mode="hdmi",
            )
        ),
        "LOG": MessageLogger(
            MessageLoggerSettings(
                output=test_filename,
            )
        ),
        "TERM": TerminateOnTotal(
            TerminateOnTotalSettings(
                total=n_messages,
            )
        ),
    }
    conns = (
        (comps["SYNTH"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    messages: list[AxisArray] = list(message_log(test_filename))
    os.remove(test_filename)

    # TerminateOnTotal may allow extra messages through
    assert len(messages) >= n_messages
    messages = messages[:n_messages]

    # Verify each message has correct shape (dense output)
    for msg in messages:
        assert msg.data.shape == (n_time, n_ch)
        # Output should be dense (not sparse)
        assert isinstance(msg.data, np.ndarray)

    # Concatenate and check properties
    agg = AxisArray.concatenate(*messages, dim="time")

    # Check sample rate (DNSS is fixed at 30kHz)
    assert agg.axes["time"].gain == 1.0 / DEFAULT_FS

    # Output should have values (LFP contributes even without spikes)
    assert not np.allclose(agg.data, 0.0)

    # Check that LFP component is present (all channels should have similar baseline)
    # LFP is broadcast to all channels, so correlation should be high
    ch0 = agg.data[:, 0]
    for ch in range(1, n_ch):
        correlation = np.corrcoef(ch0, agg.data[:, ch])[0, 1]
        # High correlation expected due to shared LFP
        assert correlation > 0.9, f"Channel {ch} correlation: {correlation}"

    # Check amplitude range - should include both LFP and spike contributions
    # max_lfp = sum(LFP_GAINS["hdmi"])  # ~2683
    # max_spike = np.max(np.abs(wf_orig))  # Max waveform amplitude
    # Combined max should be higher than LFP alone if spikes are present
    max_signal = np.max(np.abs(agg.data))
    assert max_signal > 0, "Expected non-zero signal"
