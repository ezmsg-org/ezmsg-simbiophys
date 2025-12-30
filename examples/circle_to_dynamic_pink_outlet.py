"""Circular motion to velocity-modulated LFP, streamed over LSL.

This example generates a simulated cursor moving in a circle, computes its
velocity, encodes the velocity into LFP-like colored noise, and streams the
result over Lab Streaming Layer (LSL).

Pipeline::

    Clock -> Counter -> SinGenerator (circle) -> Diff (velocity)
          -> Velocity2LFP -> LSLOutlet

The circular motion produces smoothly varying velocity vectors that sweep
through all directions. The Velocity2LFP system converts this into multi-channel
colored noise where the spectral properties are modulated by velocity magnitude
and direction.

This is useful for:
    - Testing LFP processing pipelines with known ground truth
    - Validating velocity decoding algorithms
    - Demonstrating the ezmsg-simbiophys velocity encoding system

Usage::

    uv run python circle_to_dynamic_pink_outlet.py --help
    uv run python circle_to_dynamic_pink_outlet.py --output-fs 1000 --output-ch 64

Args:
    graph_addr: Address for ezmsg graph server (ip:port). Empty to disable.
    cursor_fs: Simulated cursor position update rate in Hz.
    output_fs: Output sampling rate in Hz.
    output_ch: Number of output channels.
    seed: Random seed for reproducibility.

See Also:
    :mod:`ezmsg.simbiophys.system.velocity2lfp`: The LFP encoding system.
    :func:`mouse_to_lsl_full`: Similar example using real mouse input.
"""

import ezmsg.core as ez
import numpy as np
import typer
from ezmsg.baseproc import Clock, ClockSettings, Counter, CounterSettings
from ezmsg.lsl.outlet import LSLOutletSettings, LSLOutletUnit
from ezmsg.sigproc.diff import DiffSettings, DiffUnit

from ezmsg.simbiophys.oscillator import SinGenerator, SinGeneratorSettings
from ezmsg.simbiophys.system.velocity2lfp import Velocity2LFP, Velocity2LFPSettings

GRAPH_IP = "127.0.0.1"
GRAPH_PORT = 25978


def main(
    graph_addr: str = ":".join((GRAPH_IP, str(GRAPH_PORT))),
    cursor_fs: float = 100.0,
    output_fs: float = 30_000.0,
    output_ch: int = 256,
    seed: int = 6767,
):
    if not graph_addr:
        graph_addr = None
    else:
        graph_ip, graph_port = graph_addr.split(":")
        graph_port = int(graph_port)
        graph_addr = (graph_ip, graph_port)

    comps = {
        "CLOCK": Clock(ClockSettings(dispatch_rate=cursor_fs)),
        "COUNTER": Counter(CounterSettings(fs=cursor_fs)),
        "OSCILLATOR": SinGenerator(
            SinGeneratorSettings(
                n_ch=2,  # x,y
                freq=0.25,  # 1/4 Hz = 4 second period
                amp=200.0,  # radius 200 pixels
                phase=[np.pi / 2, 0.0],  # [x, y]: cos = sin + π/2, counterclockwise from (200, 0)
            )
        ),
        "DIFF": DiffUnit(DiffSettings(axis="time", scale_by_fs=True)),
        "VEL2LFP": Velocity2LFP(
            Velocity2LFPSettings(
                output_fs=output_fs,
                output_ch=output_ch,
                seed=seed,
            )
        ),
        "SINK": LSLOutletUnit(
            LSLOutletSettings(
                stream_name="CircleModulatedPinkNoise",
                stream_type="EEG",
            )
        ),
    }
    conns = (
        (comps["CLOCK"].OUTPUT_SIGNAL, comps["COUNTER"].INPUT_SIGNAL),
        (comps["COUNTER"].OUTPUT_SIGNAL, comps["OSCILLATOR"].INPUT_SIGNAL),
        (comps["OSCILLATOR"].OUTPUT_SIGNAL, comps["DIFF"].INPUT_SIGNAL),
        (comps["DIFF"].OUTPUT_SIGNAL, comps["VEL2LFP"].INPUT_SIGNAL),
        (comps["VEL2LFP"].OUTPUT_SIGNAL, comps["SINK"].INPUT_SIGNAL),
    )

    ez.run(
        components=comps,
        connections=conns,
        graph_address=graph_addr,
    )


if __name__ == "__main__":
    typer.run(main)
