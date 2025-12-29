"""DNSS Synthesizer - Combined spike and LFP signal generation."""

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import Clock, ClockSettings, Counter, CounterSettings
from ezmsg.event.kernel import ArrayKernel, MultiKernel
from ezmsg.event.kernel_insert import SparseKernelInserterSettings, SparseKernelInserterUnit
from ezmsg.sigproc.math.add import Add
from ezmsg.util.messages.axisarray import AxisArray

from .lfp import DNSSLFPSettings, DNSSLFPUnit
from .spike import FS, DNSSSpikeSettings, DNSSSpikeUnit
from .wfs import wf_orig


def _create_dnss_multikernel() -> MultiKernel:
    """
    Create a MultiKernel with DNSS spike waveforms.

    The DNSS has 3 spike waveforms (identified as 1, 2, 3 in the sparse event data).
    Each waveform is 41 samples long.

    Returns:
        MultiKernel mapping waveform IDs 1, 2, 3 to their respective waveforms.
    """
    # wf_orig is (3, 41) with integer values representing microvolts
    # Convert to float for the kernel
    kernels = {}
    for i in range(3):
        # Waveform IDs in the generator are 1-indexed (1, 2, 3)
        waveform = wf_orig[i].astype(np.float64)
        kernels[i + 1] = ArrayKernel(waveform)
    return MultiKernel(kernels)


class DNSSSynthSettings(ez.Settings):
    """Settings for DNSS Synthesizer."""

    n_time: int = 600
    """Number of samples per block (default: 600 = 20ms at 30kHz)."""

    n_ch: int = 256
    """Number of channels."""

    lfp_pattern: str = "spike"
    """LFP pattern: "spike" or "other"."""

    mode: str = "hdmi"
    """Mode: "hdmi" reproduces HDMI bugs, "pedestal_norm" or "pedestal_wide" for analog."""


class DNSSSynth(ez.Collection):
    """
    DNSS Signal Synthesizer.

    A Collection that generates combined DNSS spike and LFP signals.
    Uses a common Clock to synchronize:
    - Spike generation (sparse events with waveform insertion)
    - LFP generation (sum of sinusoids)

    The final output is the sum of spike waveforms and LFP signal.

    Network flow:
        Clock -> Counter -> {SpikeGenerator, LFPGenerator}
        SpikeGenerator -> KernelInserter -> Add.A
        LFPGenerator -> Add.B
        Add -> OUTPUT
    """

    SETTINGS = DNSSSynthSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    # Clock produces timestamps at the block rate
    CLOCK = Clock()

    # Counter converts timestamps to AxisArray with timing metadata
    COUNTER = Counter()

    # Spike path: produces sparse events, then inserts waveforms
    SPIKE = DNSSSpikeUnit()
    KERNEL_INSERT = SparseKernelInserterUnit()

    # LFP path: produces dense LFP signal
    LFP = DNSSLFPUnit()

    # Combine spike waveforms and LFP
    ADD = Add()

    def configure(self) -> None:
        # Calculate dispatch rate for blocks (DNSS is fixed at 30kHz)
        dispatch_rate = FS / self.SETTINGS.n_time

        self.CLOCK.apply_settings(ClockSettings(dispatch_rate=dispatch_rate))

        self.COUNTER.apply_settings(
            CounterSettings(
                n_time=self.SETTINGS.n_time,
                fs=FS,
            )
        )

        self.SPIKE.apply_settings(
            DNSSSpikeSettings(
                n_ch=self.SETTINGS.n_ch,
                mode=self.SETTINGS.mode,
            )
        )

        # Create MultiKernel with DNSS waveforms
        multikernel = _create_dnss_multikernel()
        self.KERNEL_INSERT.apply_settings(
            SparseKernelInserterSettings(
                kernel=multikernel,
                scale_by_value=False,  # Waveform ID used for selection, not scaling
            )
        )

        self.LFP.apply_settings(
            DNSSLFPSettings(
                n_ch=self.SETTINGS.n_ch,
                pattern=self.SETTINGS.lfp_pattern,
                mode=self.SETTINGS.mode,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            # Clock drives Counter
            (self.CLOCK.OUTPUT_SIGNAL, self.COUNTER.INPUT_SIGNAL),
            # Counter fans out to both Spike and LFP generators
            (self.COUNTER.OUTPUT_SIGNAL, self.SPIKE.INPUT_SIGNAL),
            (self.COUNTER.OUTPUT_SIGNAL, self.LFP.INPUT_SIGNAL),
            # Spike path: insert waveforms
            (self.SPIKE.OUTPUT_SIGNAL, self.KERNEL_INSERT.INPUT_SIGNAL),
            # Combine spike waveforms and LFP
            (self.KERNEL_INSERT.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_A),
            (self.LFP.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_B),
            # Output
            (self.ADD.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
