Circular Motion to Velocity-Modulated LFP
==========================================

This guide walks through the ``circle_to_dynamic_pink_outlet.py`` example,
which generates a simulated cursor moving in a circle and encodes its velocity
into LFP-like colored noise streamed over Lab Streaming Layer (LSL).

Overview
--------

The example demonstrates velocity-to-LFP encoding with a predictable,
synthetic input:

.. code-block:: text

    Clock -> Counter -> SinGenerator -> Diff -> Velocity2LFP -> LSLOutlet

The circular motion produces smoothly varying velocity vectors that sweep
through all directions, providing known ground truth for testing decoders.

Prerequisites
-------------

Install the required packages:

.. code-block:: bash

    uv add ezmsg-simbiophys ezmsg-lsl

Running the Example
-------------------

Basic usage:

.. code-block:: bash

    cd examples
    uv run python circle_to_dynamic_pink_outlet.py

With custom parameters:

.. code-block:: bash

    uv run python circle_to_dynamic_pink_outlet.py \
        --cursor-fs 100 \
        --output-fs 30000 \
        --output-ch 256

Command-Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

``--graph-addr``
    Address for the ezmsg graph server (ip:port). Set empty to disable.
    Default: ``127.0.0.1:25978``

``--cursor-fs``
    Simulated cursor update rate in Hz. Default: ``100.0``

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

Generates timing signals at the specified rate (default 100 Hz).

Counter
~~~~~~~

Converts clock signals to integer sample counts, used for computing
sinusoidal positions.

SinGenerator
~~~~~~~~~~~~

Generates circular motion by outputting two sinusoids with a 90-degree
phase difference:

.. code-block:: python

    SinGenerator(SinGeneratorSettings(
        n_ch=2,           # x, y coordinates
        freq=0.25,        # 0.25 Hz = 4 second period
        amp=200.0,        # 200 pixel radius
        phase=[np.pi/2, 0.0],  # cos, sin -> counterclockwise circle
    ))

This produces a cursor position that traces a circle with:

- Period: 4 seconds
- Radius: 200 pixels
- Direction: counterclockwise starting at (200, 0)

Diff
~~~~

Differentiates position to get velocity. With ``scale_by_fs=True``, the
output is in pixels per second.

For circular motion at radius *r* and angular frequency *ω*, the velocity
magnitude is constant: *v = r × ω = 200 × 2π/4 ≈ 314* pixels/second.

Velocity2LFP
~~~~~~~~~~~~

Encodes velocity into LFP-like colored noise:

1. **Polar conversion:** Transform (vx, vy) to (magnitude, angle)
2. **Scale to beta:** Map magnitude and angle to spectral exponent range
3. **Colored noise:** Generate 1/f^β noise with β modulated by velocity
4. **Spatial mixing:** Project 2 noise sources to output channels

The result is multi-channel colored noise where spectral properties vary
with cursor velocity.

LSLOutlet
~~~~~~~~~

Streams the output over LSL with name ``CircleModulatedPinkNoise`` and
type ``EEG``.

Understanding the Encoding
--------------------------

Velocity to Beta Mapping
~~~~~~~~~~~~~~~~~~~~~~~~

The velocity components are mapped to spectral exponents (β):

- **Magnitude:** 0-314 px/s → β = 0.5-2.0
- **Angle:** 0-2π → β = 0.5-2.0

This creates two noise sources with different spectral characteristics that
vary as the cursor moves.

Spectral Exponent Effects
~~~~~~~~~~~~~~~~~~~~~~~~~

The spectral exponent β controls the noise color:

- **β = 0:** White noise (flat spectrum)
- **β = 1:** Pink noise (1/f, equal power per octave)
- **β = 2:** Brown noise (1/f², random walk)

As the cursor moves faster, one noise source becomes more "red" (higher β).
As the direction changes, the other source's spectral properties shift.

Spatial Mixing
~~~~~~~~~~~~~~

The two noise sources are projected onto output channels using a mixing
matrix based on sinusoidal weights with random perturbations:

.. code-block:: python

    weights = np.array([
        np.sin(2 * np.pi * ch_idx / output_ch),  # Source 1 weights
        np.cos(2 * np.pi * ch_idx / output_ch),  # Source 2 weights
    ]) + 0.3 * rng.standard_normal((2, output_ch))

This creates spatially-varying patterns where different channels have
different mixtures of the two velocity-modulated sources.

Verifying the Output
--------------------

The circular motion provides predictable ground truth:

1. **Constant velocity magnitude:** ~314 pixels/second
2. **Linearly varying angle:** 0 to 2π over 4 seconds
3. **Periodic behavior:** Pattern repeats every 4 seconds

You can verify the encoding by:

1. Recording the LSL stream
2. Computing spectral features from the output
3. Checking that spectral properties correlate with the known velocity pattern

Example Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from pylsl import StreamInlet, resolve_stream

    # Capture one period (4 seconds)
    streams = resolve_stream('name', 'CircleModulatedPinkNoise')
    inlet = StreamInlet(streams[0])

    samples = []
    for _ in range(int(4 * 30000)):  # 4 seconds at 30 kHz
        sample, _ = inlet.pull_sample()
        samples.append(sample)

    data = np.array(samples)

    # Compute spectrum for each second
    from scipy import signal
    for i in range(4):
        segment = data[i*30000:(i+1)*30000, 0]  # First channel
        f, psd = signal.welch(segment, fs=30000)
        # Compare spectral slope across segments

See Also
--------

- :doc:`mouse_to_ecephys` - Real mouse input with full ecephys output
- :mod:`ezmsg.simbiophys.system.velocity2lfp` - API documentation
- :mod:`ezmsg.simbiophys.dynamic_colored_noise` - Colored noise generator
