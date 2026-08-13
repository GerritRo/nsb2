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

## Acknowledgements

If you used the Gaia star catalogue or sky map, include:

> This work presents results from the European Space Agency (ESA) space mission Gaia. Gaia
> data are being processed by the Gaia Data Processing and Analysis Consortium (DPAC).
> Funding for the DPAC is provided by national institutions, in particular the institutions
> participating in the Gaia MultiLateral Agreement (MLA). The Gaia mission website is
> https://www.cosmos.esa.int/gaia. The Gaia archive website is https://archives.esac.esa.int/gaia.

Any prediction involving a scattered light path, or a catalogue binned to a sky map, uses
[HEALPix](https://healpix.sourceforge.net) via healpy:

> Some of the results in this paper have been derived using the healpy and HEALPix package.

If you use a Bandpass via the SVO filter service, you should include:

> This research has made use of the SVO Filter Profile Service
> (http://svo2.cab.inta-csic.es/theory/fps/) supported from the Spanish MINECO through grant
> AYA2017-84089 and described in
> [Rodrigo et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012ivoa.rept.1015R/abstract) and
> [Rodrigo et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020sea..confE.182R/abstract).


See the [acknowledgements page](https://gerritro.github.io/nsb2/about/acknowledgements.html)
for what to cite alongside each, and for the reference data behind the individual emission models.
