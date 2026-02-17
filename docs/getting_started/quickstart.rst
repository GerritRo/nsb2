Quick Start
===========

This guide walks through the basic workflow for simulating night sky background
with nsb2.

Running a Simulation
--------------------

The main entry point is the :class:`~nsb2.core.pipeline.Pipeline` class, which
combines emission sources, an atmospheric model, and a telescope instrument:

.. code-block:: python

   from nsb2.core.pipeline import Pipeline

   # Create a pipeline with sources, atmosphere, and instrument
   pipeline = Pipeline(sources, atmosphere, instrument)

   # Run the simulation for a given observation
   results = pipeline.predict(observation)

Example Notebooks
-----------------

The best way to get started is with the example notebooks:

- **HESS-I Tutorial** — Complete walkthrough using the H.E.S.S. Phase I telescope
- **LST Tutorial** — Simulation setup for the CTAO Large-Sized Telescope

These notebooks are available in the ``examples/`` directory of the repository.

.. code-block:: bash

   cd examples
   jupyter notebook HESSI_Tutorial.ipynb
