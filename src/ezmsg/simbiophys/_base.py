"""Base classes for counter-based producers."""

import traceback
import typing

import ezmsg.core as ez
from ezmsg.baseproc import BaseProducerUnit, MessageInType, MessageOutType, ProducerType, SettingsType
from ezmsg.baseproc.util.profile import profile_subpub

from .counter import CounterProducer


class BaseCounterFirstProducerUnit(
    BaseProducerUnit[SettingsType, MessageOutType, ProducerType],
    typing.Generic[SettingsType, MessageInType, MessageOutType, ProducerType],
):
    """
    Base class for units whose primary processor is a composite producer with a CounterProducer as the first
    processor (producer) in the chain.
    """

    INPUT_SIGNAL = ez.InputStream(MessageInType)

    def create_producer(self):
        super().create_producer()

        def recurse_get_counter(proc) -> CounterProducer:
            if hasattr(proc, "_procs"):
                return recurse_get_counter(list(proc._procs.values())[0])
            return proc

        self._counter = recurse_get_counter(self.producer)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, _: ez.Flag):
        if self.producer.settings.dispatch_rate == "ext_clock":
            out = await self.producer.__acall__()
            yield self.OUTPUT_SIGNAL, out

    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        try:
            counter_state = self._counter.state
            while True:
                # Once-only, enter the generator loop
                await counter_state.new_generator.wait()
                counter_state.new_generator.clear()

                if self.producer.settings.dispatch_rate == "ext_clock":
                    # We shouldn't even be here. Cycle around and wait on the event again.
                    continue

                # We are not using an external clock. Run the generator.
                while not counter_state.new_generator.is_set():
                    out = await self.producer.__acall__()
                    yield self.OUTPUT_SIGNAL, out
        except Exception:
            ez.logger.info(traceback.format_exc())
