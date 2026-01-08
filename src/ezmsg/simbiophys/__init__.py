"""ezmsg-simbiophys: Signal simulation and synthesis for ezmsg."""

# Clock and Counter (from ezmsg.baseproc)
from ezmsg.baseproc import (
    Clock,
    ClockProducer,
    ClockSettings,
    ClockState,
    Counter,
    CounterSettings,
    CounterTransformer,
    CounterTransformerState,
)

from .__version__ import __version__ as __version__

# Cosine Encoder
from .cosine_encoder import (
    CosineEncoderSettings,
    CosineEncoderState,
    CosineEncoderTransformer,
    CosineEncoderUnit,
)

# DNSS (Digital Neural Signal Simulator)
from .dnss import (
    # LFP
    DNSSLFPProducer,
    DNSSLFPSettings,
    DNSSLFPUnit,
    # Spike
    DNSSSpikeProducer,
    DNSSSpikeSettings,
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
    PinkNoiseProducer,
    PinkNoiseSettings,
    WhiteNoise,
    WhiteNoiseProducer,
    WhiteNoiseSettings,
    WhiteNoiseState,
)

# Oscillator
from .oscillator import (
    SinGenerator,
    SinGeneratorSettings,
    SinGeneratorState,
    SinProducer,
    SpiralGenerator,
    SpiralGeneratorSettings,
    SpiralGeneratorState,
    SpiralProducer,
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
    "SinGeneratorState",
    "SinProducer",
    "SpiralGenerator",
    "SpiralGeneratorSettings",
    "SpiralGeneratorState",
    "SpiralProducer",
    # Noise
    "PinkNoise",
    "PinkNoiseProducer",
    "PinkNoiseSettings",
    "WhiteNoise",
    "WhiteNoiseProducer",
    "WhiteNoiseSettings",
    "WhiteNoiseState",
    # EEG
    "EEGSynth",
    "EEGSynthSettings",
    # Cosine Encoder
    "CosineEncoderSettings",
    "CosineEncoderState",
    "CosineEncoderTransformer",
    "CosineEncoderUnit",
    # Dynamic Colored Noise
    "ColoredNoiseFilterState",
    "DynamicColoredNoiseSettings",
    "DynamicColoredNoiseState",
    "DynamicColoredNoiseTransformer",
    "DynamicColoredNoiseUnit",
    "compute_kasdin_coefficients",
    # DNSS LFP
    "DNSSLFPProducer",
    "DNSSLFPSettings",
    "DNSSLFPUnit",
    # DNSS Spike
    "DNSSSpikeProducer",
    "DNSSSpikeSettings",
    "DNSSSpikeUnit",
]
