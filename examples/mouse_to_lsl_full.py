"""Mouse velocity to simulated ecephys, streamed over LSL.

This example captures real mouse cursor movements, computes velocity, encodes
it into simulated extracellular electrophysiology (spikes + LFP), and streams
the result over Lab Streaming Layer (LSL).

Pipeline::

    Clock -> MousePoller -> Diff (velocity) -> VelocityEncoder -> LSLOutlet

The VelocityEncoder produces realistic ecephys signals where:
    - Spike firing rates are cosine-tuned to velocity direction
    - LFP spectral properties are modulated by velocity magnitude and angle

This is useful for:
    - Testing BCI decoders with real-time cursor control
    - Developing closed-loop neural interface simulations
    - Demonstrating velocity encoding for educational purposes

Usage::

    uv run python mouse_to_lsl_full.py --help
    uv run python mouse_to_lsl_full.py --output-fs 30000 --output-ch 256

Requirements:
    - ezmsg-peripheraldevice (for MousePoller)
    - ezmsg-lsl (for LSLOutlet)

Args:
    graph_addr: Address for ezmsg graph server (ip:port). Empty to disable.
    cursor_fs: Mouse polling rate in Hz.
    output_fs: Output sampling rate in Hz.
    output_ch: Number of output channels.
    seed: Random seed for reproducibility.

See Also:
    :mod:`ezmsg.simbiophys.system.velocity2ecephys`: The encoding system.
    :func:`circle_to_dynamic_pink_outlet`: Similar example with simulated input.
"""

import ezmsg.core as ez
import typer
from ezmsg.baseproc import Clock, ClockSettings
from ezmsg.lsl.outlet import LSLOutletSettings, LSLOutletUnit
from ezmsg.peripheraldevice.mouse import MousePoller, MousePollerSettings
from ezmsg.sigproc.diff import DiffSettings, DiffUnit

from ezmsg.simbiophys.system.velocity2ecephys import VelocityEncoder, VelocityEncoderSettings

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
        "SOURCE": MousePoller(MousePollerSettings()),
        "DIFF": DiffUnit(
            DiffSettings(
                axis="time",
                scale_by_fs=True,  # pixels per second
            )
        ),
        "ENCODER": VelocityEncoder(
            VelocityEncoderSettings(
                output_fs=output_fs,
                output_ch=output_ch,
                seed=seed,
            )
        ),
        "SINK": LSLOutletUnit(LSLOutletSettings(stream_name="MouseModulatedRaw", stream_type="ECEPhys")),
    }
    conns = (
        (comps["CLOCK"].OUTPUT_SIGNAL, comps["SOURCE"].INPUT_SIGNAL),
        (comps["SOURCE"].OUTPUT_SIGNAL, comps["DIFF"].INPUT_SIGNAL),
        (comps["DIFF"].OUTPUT_SIGNAL, comps["ENCODER"].INPUT_SIGNAL),
        (comps["ENCODER"].OUTPUT_SIGNAL, comps["SINK"].INPUT_SIGNAL),
    )
    ez.run(
        components=comps,
        connections=conns,
        graph_address=graph_addr,
    )


if __name__ == "__main__":
    typer.run(main)
