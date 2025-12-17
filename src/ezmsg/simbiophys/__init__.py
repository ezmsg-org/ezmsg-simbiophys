"""ezmsg-simbiophys: Signal simulation and synthesis for ezmsg."""

from .__version__ import __version__ as __version__

# Clock
from .clock import (
    Clock,
    ClockProducer,
    ClockSettings,
    ClockState,
    aclock,
    clock,
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
    CounterProducer,
    CounterSettings,
    CounterState,
    acounter,
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
    NoiseSettings,
    PinkNoise,
    PinkNoiseProducer,
    PinkNoiseSettings,
    RandomGenerator,
    RandomGeneratorSettings,
    RandomTransformer,
    WhiteNoise,
    WhiteNoiseProducer,
    WhiteNoiseSettings,
)

# Oscillator
from .oscillator import (
    Oscillator,
    OscillatorProducer,
    OscillatorSettings,
    SinGenerator,
    SinGeneratorSettings,
    SinTransformer,
    sin,
)

__all__ = [
    # Version
    "__version__",
    # Clock
    "Clock",
    "ClockProducer",
    "ClockSettings",
    "ClockState",
    "aclock",
    "clock",
    # Counter
    "Counter",
    "CounterProducer",
    "CounterSettings",
    "CounterState",
    "acounter",
    # Oscillator
    "Oscillator",
    "OscillatorProducer",
    "OscillatorSettings",
    "SinGenerator",
    "SinGeneratorSettings",
    "SinTransformer",
    "sin",
    # Noise
    "NoiseSettings",
    "PinkNoise",
    "PinkNoiseProducer",
    "PinkNoiseSettings",
    "RandomGenerator",
    "RandomGeneratorSettings",
    "RandomTransformer",
    "WhiteNoise",
    "WhiteNoiseProducer",
    "WhiteNoiseSettings",
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
]
