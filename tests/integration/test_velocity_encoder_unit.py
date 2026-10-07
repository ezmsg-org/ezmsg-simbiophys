"""The velocity systems declare their output unit on every message."""

import os
from dataclasses import field

import ezmsg.core as ez
from ezmsg.baseproc import Clock, ClockSettings
from ezmsg.sigproc.diff import DiffSettings, DiffUnit
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.simbiophys.oscillator import SpiralGenerator, SpiralGeneratorSettings
from ezmsg.simbiophys.system.velocity2ecephys import VelocityEncoder, VelocityEncoderSettings
from tests.helpers.util import get_test_fn

CURSOR_FS = 50.0


class EncoderUnitTestSettings(ez.Settings):
    encoder_settings: VelocityEncoderSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(default_factory=TerminateOnTotalSettings)


class EncoderUnitTest(ez.Collection):
    SETTINGS = EncoderUnitTestSettings

    CLOCK = Clock()
    SPIRAL = SpiralGenerator()
    DIFF = DiffUnit()
    ENCODER = VelocityEncoder()
    SINK = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.CLOCK.apply_settings(ClockSettings(dispatch_rate=CURSOR_FS))
        self.SPIRAL.apply_settings(SpiralGeneratorSettings(fs=CURSOR_FS, r_mean=150.0, r_amp=150.0))
        self.DIFF.apply_settings(DiffSettings(axis="time", scale_by_fs=True))
        self.ENCODER.apply_settings(self.SETTINGS.encoder_settings)
        self.SINK.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_SIGNAL, self.SPIRAL.INPUT_CLOCK),
            (self.SPIRAL.OUTPUT_SIGNAL, self.DIFF.INPUT_SIGNAL),
            (self.DIFF.OUTPUT_SIGNAL, self.ENCODER.INPUT_SIGNAL),
            (self.ENCODER.OUTPUT_SIGNAL, self.SINK.INPUT_MESSAGE),
            (self.SINK.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
        )


def test_velocity_encoder_output_is_microvolts(test_name: str | None = None):
    test_filename = get_test_fn(test_name)
    settings = EncoderUnitTestSettings(
        # Line noise on, so the unit is shown to survive every stage after Add.
        encoder_settings=VelocityEncoderSettings(output_fs=3_000.0, output_ch=8, line_noise_freq=60.0),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateOnTotalSettings(total=20),
    )
    ez.run(SYSTEM=EncoderUnitTest(settings))

    messages: list[AxisArray] = list(message_log(test_filename))
    os.remove(test_filename)
    assert messages
    assert all(msg.attrs.get("unit") == "microvolts" for msg in messages)
