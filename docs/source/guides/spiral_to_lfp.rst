Spiral Motion to Velocity-Modulated LFP
=======================================

This guide walks through the ``spiral_to_dynamic_pink_outlet.py`` example,
which generates a simulated cursor moving in a spiral pattern and encodes its
velocity into LFP-like colored noise streamed over Lab Streaming Layer (LSL).

Overview
--------

The example demonstrates velocity-to-LFP encoding with a predictable,
synthetic input:

.. code-block:: text

    Clock -> SpiralGenerator -> Diff -> CART2POL -> Velocity2LFP -> LSLOutlet

The spiral motion produces smoothly varying velocity vectors that sweep
through all directions while also varying in magnitude, providing known
ground truth for testing decoders.

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
    uv run python spiral_to_dynamic_pink_outlet.py

With custom parameters:

.. code-block:: bash

    uv run python spiral_to_dynamic_pink_outlet.py \
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

SpiralGenerator
~~~~~~~~~~~~~~~

Generates spiral 2-dimensional motion where both radius and angle vary over time:

.. code-block:: python

    SpiralGenerator(SpiralGeneratorSettings(
        r_mean=150.0,        # Mean radius
        r_amp=50.0,          # Amplitude of radial oscillation
        radial_freq=0.1,     # Radial oscillation frequency (Hz)
        angular_freq=0.25,   # Angular rotation frequency (Hz)
    ))

The parametric equations are:

- ``r(t) = r_mean + r_amp * sin(2*pi*radial_freq*t)``
- ``theta(t) = 2*pi*angular_freq*t``
- ``x(t) = r(t) * cos(theta(t))``
- ``y(t) = r(t) * sin(theta(t))``

This creates a "breathing" spiral where the cursor rotates while moving
in and out from the center.


Diff
~~~~

Differentiates position to get velocity. With ``scale_by_fs=True``, the
output is in pixels per second.

CART2POL (CoordinateSpaces)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts Cartesian velocity (vx, vy) to polar coordinates (magnitude, angle).
This transformation is done once upstream and shared if both spike and LFP
encoding are used (via VelocityEncoder).

Velocity2LFP
~~~~~~~~~~~~

Encodes polar velocity into LFP-like colored noise using a cosine tuning model:

1. **Cosine encoder:** Each of ``n_lfp_sources`` (default 8) has a random
   preferred direction. The spectral exponent beta is computed as:
   ``beta = baseline + modulation * magnitude * cos(angle - pd)``
2. **Clip:** Ensures beta values stay within valid range [0, 2]
3. **Colored noise:** Generate 1/f^β noise with β dynamically modulated per source
4. **Spatial mixing:** Project n_lfp_sources onto output_ch channels using
   sinusoidal mixing patterns

The result is multi-channel colored noise where spectral properties vary
with cursor velocity direction and magnitude.

LSLOutlet
~~~~~~~~~

Streams the output over LSL with name ``SpiralModulatedPinkNoise`` and
type ``EEG``.

Understanding the Encoding
--------------------------

Cosine Tuning Model
~~~~~~~~~~~~~~~~~~~

Each of the ``n_lfp_sources`` (default 8) has a randomly assigned preferred
direction. The spectral exponent beta for each source is computed using
a cosine tuning model:

.. code-block:: python

    beta = baseline + modulation * magnitude * cos(angle - pd)

With default settings:

- ``baseline = 1.0`` (pink noise at rest)
- ``modulation = 1.0 / max_velocity`` (scales with velocity)
- ``max_velocity = 315.0`` (pixels/second)

When moving at maximum velocity in a source's preferred direction,
beta reaches 2.0 (brown noise). When moving opposite to the preferred
direction, beta reaches 0.0 (white noise). The output is clipped to [0, 2].

Spectral Exponent Effects
~~~~~~~~~~~~~~~~~~~~~~~~~

The spectral exponent β controls the noise color:

- **β = 0:** White noise (flat spectrum)
- **β = 1:** Pink noise (1/f, equal power per octave)
- **β = 2:** Brown noise (1/f², random walk)

Each source responds differently to velocity direction based on its
preferred direction, creating a rich mixture of spectral characteristics.

Spatial Mixing
~~~~~~~~~~~~~~

The ``n_lfp_sources`` noise sources are projected onto ``output_ch`` channels
using a mixing matrix based on sinusoidal weights at different spatial
frequencies, plus random perturbations:

.. code-block:: python

    weights = np.zeros((n_sources, output_ch))
    for i in range(n_sources):
        freq = (i + 1) / n_sources
        phase = 2 * np.pi * i / n_sources
        weights[i, :] = np.sin(2 * np.pi * freq * ch_idx / output_ch + phase)
    weights += 0.3 * rng.standard_normal((n_sources, output_ch))

This creates spatially-varying patterns where different channels have
different mixtures of the velocity-modulated sources, mimicking the
spatial spread of LFP signals across electrode arrays.

Verifying the Output
--------------------

The spiral motion provides predictable ground truth:

1. **Varying velocity magnitude:** Oscillates due to radial breathing
2. **Linearly varying angle:** Rotates at ``angular_freq`` Hz
3. **Periodic behavior:** Angular period = 1/angular_freq seconds (4s default),
   radial period = 1/radial_freq seconds (10s default)

You can verify the encoding by:

1. Recording the LSL stream
2. Computing spectral features from the output
3. Checking that spectral properties correlate with the known velocity pattern

Example Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from pylsl import StreamInlet, resolve_stream

    # Capture one angular period (4 seconds with default settings)
    streams = resolve_stream('name', 'SpiralModulatedPinkNoise')
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
- :mod:`ezmsg.simbiophys.cosine_encoder` - Cosine tuning encoder
- :mod:`ezmsg.simbiophys.dynamic_colored_noise` - Colored noise generator
- :mod:`ezmsg.simbiophys.oscillator` - SpiralGenerator and SinGenerator
