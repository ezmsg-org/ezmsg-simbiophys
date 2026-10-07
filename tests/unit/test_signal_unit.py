"""Unit tests for ezmsg.simbiophys.system._signal_unit."""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys.system._signal_unit import SignalUnitSettings, SignalUnitTransformer


def _msg(attrs: dict) -> AxisArray:
    return AxisArray(
        np.zeros((10, 2)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=1000.0)},
        attrs=attrs,
    )


def test_sets_unit_and_keeps_other_attrs():
    msg = _msg({"lsl_source_id": "abc"})
    out = SignalUnitTransformer(SignalUnitSettings(unit="microvolts"))(msg)
    assert out.attrs == {"lsl_source_id": "abc", "unit": "microvolts"}
    assert "unit" not in msg.attrs
    assert out.data is msg.data


def test_replaces_a_different_unit():
    out = SignalUnitTransformer(SignalUnitSettings(unit="microvolts"))(_msg({"unit": "volts"}))
    assert out.attrs["unit"] == "microvolts"
