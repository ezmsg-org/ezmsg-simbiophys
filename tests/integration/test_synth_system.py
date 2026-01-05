"""Integration tests for ezmsg.simbiophys signal generator systems."""

import os
from dataclasses import field

import ezmsg.core as ez
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.simbiophys import (
    EEGSynth,
    EEGSynthSettings,
)
from tests.helpers.util import get_test_fn


class EEGSynthSettingsTest(ez.Settings):
    synth_settings: EEGSynthSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(default_factory=TerminateOnTotalSettings)


class EEGSynthIntegrationTest(ez.Collection):
    SETTINGS = EEGSynthSettingsTest

    SOURCE = EEGSynth()
    SINK = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.SOURCE.apply_settings(self.SETTINGS.synth_settings)
        self.SINK.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.SOURCE.OUTPUT_SIGNAL, self.SINK.INPUT_MESSAGE),
            (self.SINK.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
        )


def test_eegsynth_system(
    test_name: str | None = None,
):
    # Just a quick test to make sure the system runs. We aren't checking validity of values or anything.
    fs = 500.0
    n_time = 100  # samples per block. dispatch_rate = fs / n_time
    target_dur = 2.0
    target_messages = int(target_dur * fs / n_time)

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = EEGSynthSettingsTest(
        synth_settings=EEGSynthSettings(
            fs=fs,
            n_time=n_time,
            alpha_freq=10.5,
            n_ch=8,
        ),
        log_settings=MessageLoggerSettings(
            output=test_filename,
        ),
        term_settings=TerminateOnTotalSettings(
            total=target_messages,
        ),
    )

    system = EEGSynthIntegrationTest(settings)
    ez.run(SYSTEM=system)

    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)
    agg = AxisArray.concatenate(*messages, dim="time")
    assert agg.axes["time"].gain == 1 / fs
    # EEGSynth outputs 2D array (time, ch) from transformers
    assert agg.data.ndim == 2
    assert agg.data.shape[1] == 8  # n_ch
