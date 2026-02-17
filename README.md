# nsb2

![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Simulating Night Sky Background for Imaging Air Cherenkov Telescopes**

nsb2 is a Python library for calculating the effect of night sky background (NSB) on telescopes with large pixel areas, namely imaging air Cherenkov telescopes (IACTs). It includes models from theory and experimental data for:

- Star light
- Zodiacal light
- Moon light
- Diffuse galactic light
- Airglow

It also models atmospheric effects such as Rayleigh scattering, Mie scattering, single scattering albedo, and aerosol concentration.

Telescope models can be created from raytracing data or using simple radial PSF models. Both forward and backward propagation of light through telescope/atmosphere/source layers is supported.

## Features

- Multiple NSB emission models (stars, zodiacal, moon, airglow)
- Atmospheric scattering and absorption modeling
- Telescope response from raytracing data or radial PSF
- Forward and backward light propagation
- Computational graph created at initialization
- Interface with ctapipe for telescope modeling and plotting

## Installation

```bash
pip install git+https://github.com/GerritRo/nsb2.git
```

For development:
```bash
git clone https://github.com/GerritRo/nsb2.git
cd nsb2
pip install -e ".[dev]"
```

## Quick Start

```python
from nsb2.core.pipeline import Pipeline
from nsb2.core.instrument import Instrument

# Set up a telescope instrument
instrument = Instrument.from_config("HESSI")

# Create a simulation pipeline
pipeline = Pipeline(instrument)

# Run the simulation
result = pipeline.run()
```

See the [example notebooks](examples/) for detailed tutorials on HESS-I and LST telescopes.

## Documentation

Full documentation is available at: **https://gerritro.github.io/nsb2/**

### Building Documentation Locally

```bash
pip install -e ".[docs]"
cd docs && make html
```

## License

BSD-3-Clause License - see [LICENSE](LICENSE) for details.

## Citation

If you use nsb2 in a scientific publication, please cite this repository:

```bibtex
@misc{roellinghoff_2025_nsb2,
    author = {Roellinghoff, Gerrit},
    title = {nsb2: Simulating Night Sky Background for IACTs},
    url = {https://github.com/GerritRo/nsb2},
    year = {2025}
}
```

Please also cite the origin of all telescope, atmospheric, and emission models you use in your work.
