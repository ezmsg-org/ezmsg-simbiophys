import ezmsg.core as ez
import typer
from ezmsg.baseproc import Clock, ClockSettings
from ezmsg.lsl.outlet import LSLOutletSettings, LSLOutletUnit
from ezmsg.sigproc.diff import DiffSettings, DiffUnit

from ezmsg.simbiophys.oscillator import SpiralGenerator, SpiralGeneratorSettings
from ezmsg.simbiophys.system.velocity2ecephys import VelocityEncoder, VelocityEncoderSettings

GRAPH_IP = "127.0.0.1"
GRAPH_PORT = 25978


def main(
    graph_addr: str = ":".join((GRAPH_IP, str(GRAPH_PORT))),
    cursor_fs: float = 50.0,
    output_fs: float = 30_000.0,
    output_ch: int = 128,
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
                r_amp=150.0,  # Radius oscillates +/- 150 pixels (0-300 range)
                radial_freq=0.1,  # Radial breathing at 0.1 Hz (10 second period)
                angular_freq=0.25,  # Rotation at 0.25 Hz (4 second period)
            )
        ),
        "DIFF": DiffUnit(DiffSettings(axis="time", scale_by_fs=True)),
        "ENCODER": VelocityEncoder(
            VelocityEncoderSettings(
                output_fs=output_fs,
                output_ch=output_ch,
                seed=seed,
            )
        ),
        "SINK": LSLOutletUnit(
            LSLOutletSettings(
                stream_name="VelocityModulatedECEPhys",
                stream_type="ECEPhys",
            )
        ),
    }
    conns = (
        (comps["CLOCK"].OUTPUT_SIGNAL, comps["SPIRAL"].INPUT_CLOCK),
        (comps["SPIRAL"].OUTPUT_SIGNAL, comps["DIFF"].INPUT_SIGNAL),
        (comps["DIFF"].OUTPUT_SIGNAL, comps["ENCODER"].INPUT_SIGNAL),
        (comps["ENCODER"].OUTPUT_SIGNAL, comps["SINK"].INPUT_SIGNAL),
    )

    ez.run(components=comps, connections=conns, graph_address=graph_addr, process_components=(comps["ENCODER"],))


if __name__ == "__main__":
    typer.run(main)
