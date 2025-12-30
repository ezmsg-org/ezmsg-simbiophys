Mouse Velocity to Simulated Ecephys
====================================

This guide walks through the ``mouse_to_lsl_full.py`` example, which captures
real mouse cursor movements and encodes them into simulated extracellular
electrophysiology (ecephys) signals streamed over Lab Streaming Layer (LSL).

Overview
--------

The example demonstrates a complete pipeline for generating realistic neural
signals that encode cursor velocity:

.. code-block:: text

    Clock -> MousePoller -> Diff -> VelocityEncoder -> LSLOutlet

The output contains both spike waveforms and LFP-like background activity,
making it suitable for testing brain-computer interface (BCI) decoders.

Prerequisites
-------------

Install the required packages:

.. code-block:: bash

    uv add ezmsg-simbiophys ezmsg-peripheraldevice ezmsg-lsl

Running the Example
-------------------

Basic usage:

.. code-block:: bash

    cd examples
    uv run python mouse_to_lsl_full.py

With custom parameters:

.. code-block:: bash

    uv run python mouse_to_lsl_full.py \
        --cursor-fs 100 \
        --output-fs 30000 \
        --output-ch 256

Command-Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

``--graph-addr``
    Address for the ezmsg graph server (ip:port). Set empty to disable.
    Default: ``127.0.0.1:25978``

``--cursor-fs``
    Mouse polling rate in Hz. Default: ``100.0``

``--output-fs``
    Output sampling rate in Hz. Default: ``30000.0``

``--output-ch``
    Number of output channels. Default: ``256``

``--seed``
    Random seed for reproducibility. Default: ``6767``

Pipeline Components
-------------------

Clock
~~~~~

Drives the system at the specified cursor polling rate (default 100 Hz).

MousePoller
~~~~~~~~~~~

Captures the current cursor position (x, y) at each clock tick. Requires
the ``ezmsg-peripheraldevice`` package.

Diff
~~~~

Computes velocity from position by differentiating along the time axis.
With ``scale_by_fs=True``, output is in pixels per second.

VelocityEncoder
~~~~~~~~~~~~~~~

The core encoding system from ``ezmsg.simbiophys.system.velocity2ecephys``.
It runs two parallel branches:

**Spike Branch:**

1. Convert velocity to polar coordinates (magnitude, angle)
2. Apply cosine tuning to generate firing rates
3. Generate Poisson spike events
4. Insert realistic spike waveforms

**LFP Branch:**

1. Convert velocity to polar coordinates
2. Scale to spectral exponent (beta) range
3. Generate dynamically-modulated colored noise
4. Mix across output channels

The branches are summed to produce the final output.

LSLOutlet
~~~~~~~~~

Streams the encoded signal over LSL with stream name ``MouseModulatedRaw``
and type ``ECEPhys``.

Receiving the Stream
--------------------

Use any LSL-compatible receiver to capture the stream:

**Python (pylsl):**

.. code-block:: python

    from pylsl import StreamInlet, resolve_stream

    streams = resolve_stream('name', 'MouseModulatedRaw')
    inlet = StreamInlet(streams[0])

    while True:
        sample, timestamp = inlet.pull_sample()
        print(f"t={timestamp:.3f}: {sample[:5]}...")  # First 5 channels

**MATLAB:**

.. code-block:: matlab

    lib = lsl_loadlib();
    result = lsl_resolve_byprop(lib, 'name', 'MouseModulatedRaw');
    inlet = lsl_inlet(result{1});

    while true
        [sample, ts] = inlet.pull_sample();
        disp(sample(1:5));
    end

Visualizing with ezmsg-tools
----------------------------

Use the signal monitor to visualize the stream:

.. code-block:: bash

    # Terminal 1: Start the graph server
    uv run ezmsg serve --port 25978

    # Terminal 2: Run the example
    uv run python mouse_to_lsl_full.py

    # Terminal 3: Open the signal monitor
    cd ../ezmsg-tools
    uv run python -m ezmsg.tools.sigmon.main

Click on nodes in the graph to view their output signals.

Modifying the Example
---------------------

Using a Different Source
~~~~~~~~~~~~~~~~~~~~~~~~

Replace ``MousePoller`` with other input sources:

- **Gamepad:** Use ``ezmsg-peripheraldevice`` gamepad support
- **Simulated circle:** See ``circle_to_dynamic_pink_outlet.py``
- **Task parser:** Extract velocity from behavioral task data

Changing the Output
~~~~~~~~~~~~~~~~~~~

Replace ``LSLOutlet`` with other sinks:

- **File logging:** Use ``ezmsg.util.log.Log``
- **Debug output:** Use ``ezmsg.util.debug.DebugLog``
- **Custom processing:** Connect to your own ezmsg unit

See Also
--------

- :doc:`circle_to_lfp` - Similar example with simulated circular motion
- :mod:`ezmsg.simbiophys.system.velocity2ecephys` - API documentation
