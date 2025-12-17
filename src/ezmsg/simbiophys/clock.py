"""Clock generator for timing control."""

import asyncio
import time
import typing
from dataclasses import field

import ezmsg.core as ez
from ezmsg.baseproc import BaseProducerUnit, BaseStatefulProducer, processor_state


class ClockSettings(ez.Settings):
    """Settings for clock generator."""

    dispatch_rate: float | str | None = None
    """Dispatch rate in Hz, 'realtime', or None for external clock"""


@processor_state
class ClockState:
    """State for clock generator."""

    t_0: float = field(default_factory=time.time)  # Start time
    n_dispatch: int = 0  # Number of dispatches


class ClockProducer(BaseStatefulProducer[ClockSettings, ez.Flag, ClockState]):
    """
    Produces clock ticks at specified rate.
    Can be used to drive periodic operations.
    """

    def _reset_state(self) -> None:
        """Reset internal state."""
        self._state.t_0 = time.time()
        self._state.n_dispatch = 0

    def __call__(self) -> ez.Flag:
        """Synchronous clock production. We override __call__ (which uses run_coroutine_sync)
        to avoid async overhead."""
        if self._hash == -1:
            self._reset_state()
            self._hash = 0

        if isinstance(self.settings.dispatch_rate, (int, float)):
            # Manual dispatch_rate. (else it is 'as fast as possible')
            target_time = self.state.t_0 + (self.state.n_dispatch + 1) / self.settings.dispatch_rate
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)

        self.state.n_dispatch += 1
        return ez.Flag()

    async def _produce(self) -> ez.Flag:
        """Generate next clock tick."""
        if isinstance(self.settings.dispatch_rate, (int, float)):
            # Manual dispatch_rate. (else it is 'as fast as possible')
            target_time = self.state.t_0 + (self.state.n_dispatch + 1) / self.settings.dispatch_rate
            now = time.time()
            if target_time > now:
                await asyncio.sleep(target_time - now)

        self.state.n_dispatch += 1
        return ez.Flag()


def aclock(dispatch_rate: float | None) -> ClockProducer:
    """
    Construct an async generator that yields events at a specified rate.

    Returns:
        A :obj:`ClockProducer` object.
    """
    return ClockProducer(ClockSettings(dispatch_rate=dispatch_rate))


clock = aclock
"""
Alias for :obj:`aclock` expected by synchronous methods. `ClockProducer` can be used in sync or async.
"""


class Clock(
    BaseProducerUnit[
        ClockSettings,  # SettingsType
        ez.Flag,  # MessageType
        ClockProducer,  # ProducerType
    ]
):
    SETTINGS = ClockSettings

    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        # Override so we can not to yield if out is False-like
        while True:
            out = await self.producer.__acall__()
            if out:
                yield self.OUTPUT_SIGNAL, out
