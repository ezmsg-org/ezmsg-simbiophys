"""
Example usage of ezmsg-simbiophys package.
This is just a placeholder to test importing. Actual usage example to come.
"""

import asyncio
import importlib
import typing

import ezmsg.core as ez

# Import your units from the package
# from ezmsg.simbiophys import MyUnit


class ExampleSettings(ez.Settings):
    """Settings for ExampleUnit."""

    message: str = "Hello from ezmsg-simbiophys!"


class ExampleUnit(ez.Unit):
    """Example ezmsg Unit demonstrating basic patterns."""

    SETTINGS = ExampleSettings

    INPUT = ez.InputStream(str)
    OUTPUT = ez.OutputStream(str)

    @ez.subscriber(INPUT)
    @ez.publisher(OUTPUT)
    async def on_message(self, message: str) -> typing.AsyncGenerator:
        """Process incoming messages."""
        result = f"{self.SETTINGS.message} Received: {message}"
        yield self.OUTPUT, result


async def main():
    """Run the example."""
    print("ezmsg-simbiophys loaded successfully!")
    print(f"Version: {importlib.import_module('ezmsg.simbiophys').__version__}")

    # Example: Create and run a simple system
    # system = ExampleSystem()
    # await ez.run(SYSTEM=system)


if __name__ == "__main__":
    asyncio.run(main())
