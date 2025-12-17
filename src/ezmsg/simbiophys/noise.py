"""Noise signal generators."""

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import (
    BaseTransformer,
    BaseTransformerUnit,
    CompositeProducer,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ._base import BaseCounterFirstProducerUnit
from .counter import CounterProducer, CounterSettings


class RandomGeneratorSettings(ez.Settings):
    loc: float = 0.0
    """loc argument for :obj:`numpy.random.normal`"""

    scale: float = 1.0
    """scale argument for :obj:`numpy.random.normal`"""


class RandomTransformer(BaseTransformer[RandomGeneratorSettings, AxisArray, AxisArray]):
    """
    Replaces input data with random data and returns the result.
    """

    def __init__(self, *args, settings: RandomGeneratorSettings | None = None, **kwargs):
        super().__init__(*args, settings=settings, **kwargs)

    def _process(self, message: AxisArray) -> AxisArray:
        random_data = np.random.normal(size=message.shape, loc=self.settings.loc, scale=self.settings.scale)
        return replace(message, data=random_data)


class RandomGenerator(
    BaseTransformerUnit[
        RandomGeneratorSettings,
        AxisArray,
        AxisArray,
        RandomTransformer,
    ]
):
    SETTINGS = RandomGeneratorSettings


class NoiseSettings(ez.Settings):
    """
    See :obj:`CounterSettings` and :obj:`RandomGeneratorSettings`.
    """

    n_time: int  # Number of samples to output per block
    fs: float  # Sampling rate of signal output in Hz
    n_ch: int = 1  # Number of channels to output
    dispatch_rate: float | str | None = None
    """(Hz), 'realtime', or 'ext_clock'"""
    loc: float = 0.0  # DC offset
    scale: float = 1.0  # Scale (in standard deviations)


WhiteNoiseSettings = NoiseSettings


class WhiteNoiseProducer(CompositeProducer[NoiseSettings, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: NoiseSettings,
    ) -> dict[str, CounterProducer | RandomTransformer]:
        return {
            "counter": CounterProducer(
                CounterSettings(
                    n_time=settings.n_time,
                    fs=settings.fs,
                    n_ch=settings.n_ch,
                    dispatch_rate=settings.dispatch_rate,
                    mod=None,
                )
            ),
            "random": RandomTransformer(
                RandomGeneratorSettings(
                    loc=settings.loc,
                    scale=settings.scale,
                )
            ),
        }


class WhiteNoise(BaseCounterFirstProducerUnit[NoiseSettings, AxisArray, AxisArray, WhiteNoiseProducer]):
    """chains a :obj:`Counter` and :obj:`RandomGenerator`."""

    SETTINGS = NoiseSettings


PinkNoiseSettings = NoiseSettings


class PinkNoiseProducer(CompositeProducer[PinkNoiseSettings, AxisArray]):
    @staticmethod
    def _initialize_processors(
        settings: PinkNoiseSettings,
    ) -> dict[str, WhiteNoiseProducer]:
        # Import here to allow simbiophys to work without sigproc dependency if PinkNoise is not used
        from ezmsg.sigproc.butterworthfilter import ButterworthFilterSettings, ButterworthFilterTransformer

        return {
            "white_noise": WhiteNoiseProducer(settings=settings),
            "filter": ButterworthFilterTransformer(
                settings=ButterworthFilterSettings(
                    axis="time",
                    order=1,
                    cutoff=settings.fs * 0.01,  # Hz
                )
            ),
        }


class PinkNoise(BaseCounterFirstProducerUnit[NoiseSettings, AxisArray, AxisArray, PinkNoiseProducer]):
    """chains :obj:`WhiteNoise` and :obj:`ButterworthFilter`."""

    SETTINGS = NoiseSettings
