"""Mouse velocity source - publishes mouse cursor velocity to a topic.

This script captures mouse cursor position, computes velocity, and publishes
it to a topic. Run velocity_encoder_server.py separately to consume the
velocity and produce ecephys output.

Pipeline::

    Clock -> MousePoller -> Diff (velocity) -> "velocity" topic

Usage::

    # Terminal 1: Start the encoder server
    uv run python velocity_encoder_server.py

    # Terminal 2: Start this velocity source
    uv run python mouse_velocity_source.py

Requirements:
    - ezmsg-peripheraldevice (for MousePoller)

Args:
    graph_addr: Address for ezmsg graph server (ip:port).
    cursor_fs: Mouse polling rate in Hz.
    velocity_topic: Topic name to publish velocity to.

See Also:
    :func:`velocity_encoder_server`: The encoder that consumes this velocity.
    :func:`spiral_velocity_source`: Alternative input using simulated spiral.
"""

import ezmsg.core as ez
import typer
from ezmsg.baseproc import Clock, ClockSettings
from ezmsg.peripheraldevice.mouse import MousePoller, MousePollerSettings
from ezmsg.sigproc.diff import DiffSettings, DiffUnit

GRAPH_IP = "127.0.0.1"
GRAPH_PORT = 25978


def main(
    graph_addr: str = ":".join((GRAPH_IP, str(GRAPH_PORT))),
    cursor_fs: float = 50.0,
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
    }
    conns = (
        (comps["CLOCK"].OUTPUT_SIGNAL, comps["SOURCE"].INPUT_CLOCK),
        (comps["SOURCE"].OUTPUT_SIGNAL, comps["DIFF"].INPUT_SIGNAL),
        # Publish velocity to topic (consumed by encoder server)
        (comps["DIFF"].OUTPUT_SIGNAL, "ENCODER/INPUT_SIGNAL"),
    )

    ez.run(
        components=comps,
        connections=conns,
        graph_address=graph_addr,
    )


if __name__ == "__main__":
    typer.run(main)
