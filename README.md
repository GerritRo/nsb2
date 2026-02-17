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
- Airglow

It also models atmospheric effects such as Rayleigh scattering, Mie scattering, single scattering albedo, and aerosol concentration.

Telescope models can be created from raytracing data. It automatically decides the render direction for source types and supports mapping catalog data via HEALPix for efficient in-scattering lookups.

## Features

- Multiple NSB emission models (stars, zodiacal, moon, airglow)
- Atmospheric scattering and absorption modeling
- Telescope response from raytracing data
- Forward and backward light propagation
- Interface with ctapipe for plotting

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
Please also cite the original NSB paper using nsb2:

```bibtex
@article{roellinghoff_2025_advanced,
  title={Advanced modelling of the night sky background light for imaging atmospheric Cherenkov telescopes},
  author={Roellinghoff, Gerrit and Spencer, Samuel T and Funk, Stefan},
  journal={Astronomy \& Astrophysics},
  volume={698},
  pages={A212},
  year={2025},
  publisher={EDP Sciences}
}
```

Please also cite the origin of all telescope, atmospheric, and emission models you use in your work.
