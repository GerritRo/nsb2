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

Using nsb2 with ctapipe
-----------------------

nsb2's dependency bounds are chosen to overlap those of
`ctapipe <https://ctapipe.readthedocs.io/>`_, so the two install into a
single environment without a resolver conflict:

.. code-block:: bash

   pip install ctapipe git+https://github.com/GerritRo/nsb2.git

Note that ctapipe itself requires **Python 3.12 or later**, which is
stricter than nsb2's own floor.  A CI job installs both together and runs
the nsb2 test suite in that environment, so the combination stays working.

Development Installation
------------------------

For development or to access example notebooks, clone the repository:

.. code-block:: bash

   git clone https://github.com/GerritRo/nsb2.git
   cd nsb2
   pip install -e ".[dev]"
   pre-commit install

This installs additional development dependencies:

- ``pytest`` for running tests
- ``mypy`` for type checking
- ``ruff`` for code linting and formatting
- ``pre-commit`` for running both automatically before each commit
- ``commitizen`` for conventional commits

Narrower extras are available if you do not need the whole toolchain:
``nsb2[tests]`` installs just the test requirements, and ``nsb2[docs]``
just the documentation ones.

Running the Tests
-----------------

.. code-block:: bash

   pytest

Tests live next to the code they cover, in ``nsb2/<subpackage>/tests/``.
Tests that download reference data from STScI, Zenodo or the SVO Filter
Profile Service are marked ``remote_data`` and are skipped by default; run
them explicitly with:

.. code-block:: bash

   pytest -m remote_data

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
