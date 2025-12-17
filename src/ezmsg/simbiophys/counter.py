"""Counter generator for sample counting and timing."""

import asyncio
import time
import traceback
import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import BaseProducerUnit, BaseStatefulProducer, processor_state
from ezmsg.util.messages.axisarray import AxisArray


class CounterSettings(ez.Settings):
    """
    Settings for :obj:`Counter`.
    See :obj:`acounter` for a description of the parameters.
    """

    n_time: int
    """Number of samples to output per block."""

    fs: float
    """Sampling rate of signal output in Hz"""

    n_ch: int = 1
    """Number of channels to synthesize"""

    dispatch_rate: float | str | None = None
    """
    Message dispatch rate (Hz), 'realtime', 'ext_clock', or None (fast as possible)
     Note: if dispatch_rate is a float then time offsets will be synthetic and the
     system will run faster or slower than wall clock time.
    """

    mod: int | None = None
    """If set to an integer, counter will rollover"""


@processor_state
class CounterState:
    """
    State for counter generator.
    """

    counter_start: int = 0
    """next sample's first value"""

    n_sent: int = 0
    """number of samples sent"""

    clock_zero: float | None = None
    """time of first sample"""

    timer_type: str = "unspecified"
    """
    "realtime" | "ext_clock" | "manual" | "unspecified"
    """

    new_generator: asyncio.Event | None = None
    """
    Event to signal the counter has been reset.
    """


class CounterProducer(BaseStatefulProducer[CounterSettings, AxisArray, CounterState]):
    """Produces incrementing integer blocks as AxisArray."""

    @classmethod
    def get_message_type(cls, dir: str) -> typing.Optional[type[AxisArray]]:
        if dir == "in":
            return None
        elif dir == "out":
            return AxisArray
        else:
            raise ValueError(f"Invalid direction: {dir}. Use 'in' or 'out'.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.settings.dispatch_rate, str) and self.settings.dispatch_rate not in [
            "realtime",
            "ext_clock",
        ]:
            raise ValueError(f"Unknown dispatch_rate: {self.settings.dispatch_rate}")
        self._reset_state()
        self._hash = 0

    def _reset_state(self) -> None:
        """Reset internal state."""
        self._state.counter_start = 0
        self._state.n_sent = 0
        self._state.clock_zero = time.time()
        if self.settings.dispatch_rate is not None:
            if isinstance(self.settings.dispatch_rate, str):
                self._state.timer_type = self.settings.dispatch_rate.lower()
            else:
                self._state.timer_type = "manual"
        if self._state.new_generator is None:
            self._state.new_generator = asyncio.Event()
        # Set the event to indicate that the state has been reset.
        self._state.new_generator.set()

    async def _produce(self) -> AxisArray:
        """Generate next counter block."""
        # 1. Prepare counter data
        block_samp = np.arange(self.state.counter_start, self.state.counter_start + self.settings.n_time)[:, np.newaxis]
        if self.settings.mod is not None:
            block_samp %= self.settings.mod
        block_samp = np.tile(block_samp, (1, self.settings.n_ch))

        # 2. Sleep if necessary. 3. Calculate time offset.
        if self._state.timer_type == "realtime":
            n_next = self.state.n_sent + self.settings.n_time
            t_next = self.state.clock_zero + n_next / self.settings.fs
            await asyncio.sleep(t_next - time.time())
            offset = t_next - self.settings.n_time / self.settings.fs
        elif self._state.timer_type == "manual":
            # manual dispatch rate
            n_disp_next = 1 + self.state.n_sent / self.settings.n_time
            t_disp_next = self.state.clock_zero + n_disp_next / self.settings.dispatch_rate
            await asyncio.sleep(t_disp_next - time.time())
            offset = self.state.n_sent / self.settings.fs
        elif self._state.timer_type == "ext_clock":
            #  ext_clock -- no sleep. Assume this is called at appropriate intervals.
            offset = time.time()
        else:
            # Was "unspecified"
            offset = self.state.n_sent / self.settings.fs

        # 4. Create output AxisArray
        # Note: We can make this a bit faster by preparing a template for self._state
        result = AxisArray(
            data=block_samp,
            dims=["time", "ch"],
            axes={
                "time": AxisArray.TimeAxis(fs=self.settings.fs, offset=offset),
                "ch": AxisArray.CoordinateAxis(
                    data=np.array([f"Ch{_}" for _ in range(self.settings.n_ch)]),
                    dims=["ch"],
                ),
            },
            key="acounter",
        )

        # 5. Update state
        self.state.counter_start = block_samp[-1, 0] + 1
        self.state.n_sent += self.settings.n_time

        return result


def acounter(
    n_time: int,
    fs: float | None,
    n_ch: int = 1,
    dispatch_rate: float | str | None = None,
    mod: int | None = None,
) -> CounterProducer:
    """
    Construct an asynchronous generator to generate AxisArray objects at a specified rate
    and with the specified sampling rate.

    NOTE: This module uses asyncio.sleep to delay appropriately in realtime mode.
    This method of sleeping/yielding execution priority has quirky behavior with
    sub-millisecond sleep periods which may result in unexpected behavior (e.g.
    fs = 2000, n_time = 1, realtime = True -- may result in ~1400 msgs/sec)

    Returns:
        An asynchronous generator.
    """
    return CounterProducer(CounterSettings(n_time=n_time, fs=fs, n_ch=n_ch, dispatch_rate=dispatch_rate, mod=mod))


class Counter(
    BaseProducerUnit[
        CounterSettings,  # SettingsType
        AxisArray,  # MessageOutType
        CounterProducer,  # ProducerType
    ]
):
    """Generates monotonically increasing counter. Unit for :obj:`CounterProducer`."""

    SETTINGS = CounterSettings
    INPUT_CLOCK = ez.InputStream(ez.Flag)

    @ez.subscriber(INPUT_CLOCK)
    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def on_clock(self, _: ez.Flag):
        if self.producer.settings.dispatch_rate == "ext_clock":
            out = await self.producer.__acall__()
            yield self.OUTPUT_SIGNAL, out

    @ez.publisher(BaseProducerUnit.OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        """
        Generate counter output.
        This is an infinite loop, but we will likely only enter the loop once if we are self-timed,
        and twice if we are using an external clock.

        When using an internal clock, we enter the loop, and wait for the event which should have
        been reset upon initialization then we immediately clear, then go to the internal loop
        that will async call __acall__ to let the internal timer determine when to produce an output.

        When using an external clock, we enter the loop, and wait for the event which should have been
        reset upon initialization then we immediately clear, then we hit `continue` to loop back around
        and wait for the event to be set again -- potentially forever. In this case, it is expected that
        `on_clock` will be called to produce the output.
        """
        try:
            while True:
                # Once-only, enter the generator loop
                await self.producer.state.new_generator.wait()
                self.producer.state.new_generator.clear()

                if self.producer.settings.dispatch_rate == "ext_clock":
                    # We shouldn't even be here. Cycle around and wait on the event again.
                    continue

                # We are not using an external clock. Run the generator.
                while not self.producer.state.new_generator.is_set():
                    out = await self.producer.__acall__()
                    yield self.OUTPUT_SIGNAL, out
        except Exception:
            ez.logger.info(traceback.format_exc())
