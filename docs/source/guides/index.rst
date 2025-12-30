Guides
======

Step-by-step tutorials for using ezmsg-simbiophys to generate simulated
neural signals.

.. toctree::
   :maxdepth: 2

   mouse_to_ecephys
   circle_to_lfp

Overview
--------

These guides demonstrate how to use the velocity encoding systems to generate
realistic neural signals for testing brain-computer interface (BCI) applications.

**Mouse to Ecephys** (:doc:`mouse_to_ecephys`)
    Capture real mouse movements and encode them into simulated extracellular
    electrophysiology containing both spikes and LFP.

**Circle to LFP** (:doc:`circle_to_lfp`)
    Generate a simulated cursor moving in a circle and encode its velocity
    into LFP-like colored noise with known ground truth.

Getting Started
---------------

All examples stream their output over Lab Streaming Layer (LSL), making it
easy to receive and process the data in other applications.

Prerequisites
~~~~~~~~~~~~~

.. code-block:: bash

    # Core package
    uv add ezmsg-simbiophys

    # For LSL streaming
    uv add ezmsg-lsl

    # For mouse input (mouse_to_ecephys only)
    uv add ezmsg-peripheraldevice

Running Examples
~~~~~~~~~~~~~~~~

Examples are located in the ``examples/`` directory:

.. code-block:: bash

    cd ezmsg-simbiophys/examples
    uv run python mouse_to_lsl_full.py
    # or
    uv run python circle_to_dynamic_pink_outlet.py

Use ``--help`` to see available options:

.. code-block:: bash

    uv run python mouse_to_lsl_full.py --help
