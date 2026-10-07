"""Declare the physical unit of a simulated signal on its messages.

The systems in this package manufacture their amplitudes in a fixed unit (the
DNSS spike templates are microvolts and the LFP scale is chosen to match), but
none of the generators along the way know it. This stage writes it to the
message's ``unit`` attr at the end of each system so consumers -- e.g. an LSL
outlet, which publishes attrs as stream metadata -- can report it.
"""

import ezmsg.core as ez
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class SignalUnitSettings(ez.Settings):
    unit: str
    """Value written to the output's ``unit`` attr."""


class SignalUnitTransformer(BaseTransformer[SignalUnitSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        if message.attrs.get("unit") == self.settings.unit:
            return message
        return replace(message, attrs={**message.attrs, "unit": self.settings.unit})


class SignalUnit(BaseTransformerUnit[SignalUnitSettings, AxisArray, AxisArray, SignalUnitTransformer]):
    SETTINGS = SignalUnitSettings
