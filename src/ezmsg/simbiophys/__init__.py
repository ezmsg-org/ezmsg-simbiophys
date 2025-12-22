"""ezmsg-simbiophys: Signal simulation and synthesis for ezmsg."""

from .__version__ import __version__ as __version__

# Clock
from .clock import (
    Clock,
    ClockProducer,
    ClockSettings,
    ClockState,
)

# Cosine Tuning
from .cosine_tuning import (
    CosineTuningParams,
    CosineTuningSettings,
    CosineTuningState,
    CosineTuningTransformer,
    CosineTuningUnit,
)

# Counter
from .counter import (
    Counter,
    CounterSettings,
    CounterTransformer,
    CounterTransformerState,
)

# DNSS (Digital Neural Signal Simulator)
from .dnss import (
    # LFP
    DNSSLFPSettings,
    DNSSLFPTransformer,
    DNSSLFPUnit,
    # Spike
    DNSSSpikeSettings,
    DNSSSpikeTransformer,
    DNSSSpikeUnit,
)

# Dynamic Colored Noise
from .dynamic_colored_noise import (
    ColoredNoiseFilterState,
    DynamicColoredNoiseSettings,
    DynamicColoredNoiseState,
    DynamicColoredNoiseTransformer,
    DynamicColoredNoiseUnit,
    compute_kasdin_coefficients,
)

# EEG
from .eeg import (
    EEGSynth,
    EEGSynthSettings,
)

# Noise
from .noise import (
    PinkNoise,
    PinkNoiseSettings,
    PinkNoiseTransformer,
    WhiteNoise,
    WhiteNoiseSettings,
    WhiteNoiseTransformer,
)

# Oscillator
from .oscillator import (
    SinGenerator,
    SinGeneratorSettings,
    SinTransformer,
)

__all__ = [
    # Version
    "__version__",
    # Clock
    "Clock",
    "ClockProducer",
    "ClockSettings",
    "ClockState",
    # Counter
    "Counter",
    "CounterSettings",
    "CounterTransformer",
    "CounterTransformerState",
    # Oscillator
    "SinGenerator",
    "SinGeneratorSettings",
    "SinTransformer",
    # Noise
    "PinkNoise",
    "PinkNoiseSettings",
    "PinkNoiseTransformer",
    "WhiteNoise",
    "WhiteNoiseSettings",
    "WhiteNoiseTransformer",
    # EEG
    "EEGSynth",
    "EEGSynthSettings",
    # Cosine Tuning
    "CosineTuningParams",
    "CosineTuningSettings",
    "CosineTuningState",
    "CosineTuningTransformer",
    "CosineTuningUnit",
    # Dynamic Colored Noise
    "ColoredNoiseFilterState",
    "DynamicColoredNoiseSettings",
    "DynamicColoredNoiseState",
    "DynamicColoredNoiseTransformer",
    "DynamicColoredNoiseUnit",
    "compute_kasdin_coefficients",
    # DNSS LFP
    "DNSSLFPSettings",
    "DNSSLFPTransformer",
    "DNSSLFPUnit",
    # DNSS Spike
    "DNSSSpikeSettings",
    "DNSSSpikeTransformer",
    "DNSSSpikeUnit",
]
