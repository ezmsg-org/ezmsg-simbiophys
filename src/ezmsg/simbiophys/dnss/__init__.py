"""DNSS (Digital Neural Signal Simulator) signal generation."""

from .lfp import (
    FS,
    LFP_FREQS,
    LFP_GAINS,
    LFP_PERIOD,
    LFP_SAMPLE_SHIFTS,
    OTHER_PERIOD,
    DNSSLFPGenerator,
    DNSSLFPProducer,
    DNSSLFPSettings,
    DNSSLFPTransformer,
    DNSSLFPTransformerSettings,
    DNSSLFPTransformerState,
    DNSSLFPUnit,
    lfp_generator,
)
from .spike import (
    FULL_PERIOD,
    GAP_BURST,
    INT_BURST,
    INT_SLOW,
    N_BURST_SPIKES,
    N_SLOW_SPIKES,
    SAMPS_BURST,
    SAMPS_SLOW,
    DNSSSpikeGenerator,
    DNSSSpikeProducer,
    DNSSSpikeSettings,
    DNSSSpikeTransformer,
    DNSSSpikeTransformerSettings,
    DNSSSpikeTransformerState,
    DNSSSpikeUnit,
    spike_event_generator,
)
from .synth import (
    DNSSSynth,
    DNSSSynthSettings,
)
from .wfs import wf_orig

__all__ = [
    # LFP constants
    "FS",
    "LFP_FREQS",
    "LFP_GAINS",
    "LFP_PERIOD",
    "LFP_SAMPLE_SHIFTS",
    "OTHER_PERIOD",
    # LFP classes
    "DNSSLFPGenerator",
    "DNSSLFPProducer",
    "DNSSLFPSettings",
    "DNSSLFPTransformer",
    "DNSSLFPTransformerSettings",
    "DNSSLFPTransformerState",
    "DNSSLFPUnit",
    "lfp_generator",
    # Spike constants
    "FULL_PERIOD",
    "GAP_BURST",
    "INT_BURST",
    "INT_SLOW",
    "N_BURST_SPIKES",
    "N_SLOW_SPIKES",
    "SAMPS_BURST",
    "SAMPS_SLOW",
    # Spike classes
    "DNSSSpikeGenerator",
    "DNSSSpikeProducer",
    "DNSSSpikeSettings",
    "DNSSSpikeTransformer",
    "DNSSSpikeTransformerSettings",
    "DNSSSpikeTransformerState",
    "DNSSSpikeUnit",
    "spike_event_generator",
    # Synth classes
    "DNSSSynth",
    "DNSSSynthSettings",
    # Waveforms
    "wf_orig",
]
