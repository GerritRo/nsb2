Installation
============

Requirements
------------

nsb2 requires **Python 3.11 or later**.

Basic Installation
------------------

Install directly from the git repository:

.. code-block:: bash

   pip install git+https://github.com/GerritRo/nsb2.git

Development Installation
------------------------

For development or to access example notebooks, clone the repository:

.. code-block:: bash

   git clone https://github.com/GerritRo/nsb2.git
   cd nsb2
   pip install -e ".[dev]"

This installs additional development dependencies:

- ``pytest`` for running tests
- ``mypy`` for type checking
- ``ruff`` for code linting
- ``commitizen`` for conventional commits

Building Documentation
----------------------

To build this documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.

Dependencies
------------

nsb2 depends on:

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Package
     - Purpose
   * - `NumPy <https://numpy.org/>`_
     - Array operations
   * - `SciPy <https://scipy.org/>`_
     - Interpolation and integration
   * - `Astropy <https://www.astropy.org/>`_
     - Coordinates, units, and time
   * - `healpy <https://healpy.readthedocs.io/>`_
     - HEALPix sky pixelization
   * - `scikit-learn <https://scikit-learn.org/>`_
     - BallTree spatial queries
   * - `dust_extinction <https://dust-extinction.readthedocs.io/>`_
     - Dust extinction models
