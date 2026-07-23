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

# Baseline Drift
from .baseline_drift import (
    BaselineDriftSettings,
    BaselineDriftState,
    BaselineDriftTransformer,
    BaselineDriftUnit,
)

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
    DynamicColoredNoiseSettings,
    DynamicColoredNoiseState,
    DynamicColoredNoiseTransformer,
    DynamicColoredNoiseUnit,
    compute_kasdin_coefficients,
    compute_kasdin_coefficients_batch,
)

# EEG
from .eeg import (
    EEGSynth,
    EEGSynthSettings,
)

# Line Noise
from .line_noise import (
    LineNoiseSettings,
    LineNoiseState,
    LineNoiseTransformer,
    LineNoiseUnit,
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
    # Line Noise
    "LineNoiseSettings",
    "LineNoiseState",
    "LineNoiseTransformer",
    "LineNoiseUnit",
    # Cosine Encoder
    "CosineEncoderSettings",
    "CosineEncoderState",
    "CosineEncoderTransformer",
    "CosineEncoderUnit",
    # Baseline Drift
    "BaselineDriftSettings",
    "BaselineDriftState",
    "BaselineDriftTransformer",
    "BaselineDriftUnit",
    # Dynamic Colored Noise
    "DynamicColoredNoiseSettings",
    "DynamicColoredNoiseState",
    "DynamicColoredNoiseTransformer",
    "DynamicColoredNoiseUnit",
    "compute_kasdin_coefficients",
    "compute_kasdin_coefficients_batch",
    # DNSS LFP
    "DNSSLFPProducer",
    "DNSSLFPSettings",
    "DNSSLFPUnit",
    # DNSS Spike
    "DNSSSpikeProducer",
    "DNSSSpikeSettings",
    "DNSSSpikeUnit",
]
