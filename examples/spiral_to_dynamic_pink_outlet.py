"""Spiral motion to velocity-modulated LFP, streamed over LSL.

This example generates a simulated cursor moving in a spiral pattern, computes
its velocity, encodes the velocity into LFP-like colored noise, and streams the
result over Lab Streaming Layer (LSL).

Pipeline::

    Clock -> SpiralGenerator -> Diff (velocity) -> CART2POL -> Velocity2LFP -> LSLOutlet

The spiral motion produces varying velocity vectors where both the magnitude
(speed) and direction change over time. The SpiralGenerator creates a pattern
where:
    - The radius oscillates sinusoidally (breathing in/out)
    - The angle increases linearly (rotation)

This provides richer dynamics than circular motion for testing the velocity
encoding system, as velocity is non-zero and varying even when the cursor
"pauses" at the turning points of the radial oscillation.

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
import typer
from ezmsg.baseproc import Clock, ClockSettings
from ezmsg.lsl.outlet import LSLOutletSettings, LSLOutletUnit
from ezmsg.sigproc.coordinatespaces import CoordinateMode, CoordinateSpaces, CoordinateSpacesSettings
from ezmsg.sigproc.diff import DiffSettings, DiffUnit

from ezmsg.simbiophys.oscillator import SpiralGenerator, SpiralGeneratorSettings
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
        "SPIRAL": SpiralGenerator(
            SpiralGeneratorSettings(
                fs=cursor_fs,
                r_mean=150.0,  # Mean radius 150 pixels
                r_amp=150.0,  # Radius oscillates +/- 50 pixels (100-200 range)
                radial_freq=0.1,  # Radial breathing at 0.1 Hz (10 second period)
                angular_freq=0.25,  # Rotation at 0.25 Hz (4 second period)
            )
        ),
        "DIFF": DiffUnit(DiffSettings(axis="time", scale_by_fs=True)),
        # DIFF Output is [[vx, vy]] pixels/sec with varying magnitude
        "COORDS": CoordinateSpaces(CoordinateSpacesSettings(mode=CoordinateMode.CART2POL, axis="ch")),
        # COORDS Output is [[magnitude, angle]] polar velocity
        "VEL2LFP": Velocity2LFP(
            Velocity2LFPSettings(
                output_fs=output_fs,
                output_ch=output_ch,
                max_velocity=472.0,
                seed=seed,
            )
        ),
        "SINK": LSLOutletUnit(
            LSLOutletSettings(
                stream_name="SpiralModulatedPinkNoise",
                stream_type="EEG",
            )
        ),
    }
    conns = (
        (comps["CLOCK"].OUTPUT_SIGNAL, comps["SPIRAL"].INPUT_CLOCK),
        (comps["SPIRAL"].OUTPUT_SIGNAL, comps["DIFF"].INPUT_SIGNAL),
        (comps["DIFF"].OUTPUT_SIGNAL, comps["COORDS"].INPUT_SIGNAL),
        (comps["COORDS"].OUTPUT_SIGNAL, comps["VEL2LFP"].INPUT_SIGNAL),
        (comps["VEL2LFP"].OUTPUT_SIGNAL, comps["SINK"].INPUT_SIGNAL),
    )

    ez.run(
        components=comps,
        connections=conns,
        graph_address=graph_addr,
    )


if __name__ == "__main__":
    typer.run(main)
