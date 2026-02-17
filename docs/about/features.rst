Features
========

Emission Models
---------------

nsb2 includes models for the major components of night sky background:

- **Star light** - GAIA Catalog-based stellar emission using spectral templates
  (Pickles 1998 atlas), with BallTree spatial queries for efficient FOV matching.
- **Zodiacal light** - Zodiacal light model based on Leinert et al. (1998)
  tabulated data, interpolated across ecliptic coordinates.
- **Moon light** - Lunar scattered light model based on Noll et al. (2013)
  ROLO data, accounting for lunar phase and angular separation.
- **Airglow** - Atmospheric airglow emission based on ESO SkyCalc models
  (Noll et al. 2012), including solar activity dependence.

Atmospheric Modeling
--------------------

- **Rayleigh scattering** - Wavelength-dependent molecular scattering
- **Mie scattering** - Aerosol scattering with configurable profiles
- **Molecular Absorption** - Interpolated absorption data

Instruments
---------------------

- **Raytracing-based response** - Import response functions from optical
  raytracing simulations (e.g., from IACTrace)
- **Pre-built configurations** - Included configurations for H.E.S.S. and CTAO
  telescopes (LST, MST)

Pipeline Architecture
---------------------

- **Direct and scattered light** - Support for both direct and scattered
  light paths through the atmosphere, with flexible solvers.
- **Traceable by source** - Visualize contributions from each component 
  independently.
- **Efficient compilation** - Precomputations such as LUT builds or balltree
  creating is done in model compile step.

Data & Interfaces
-----------------

- **Bundled calibration data** - Reference spectra, scattering tables, and
  instrument responses included in the package
- **Astropy integration** - Native use of astropy coordinates, units, and
  time representations
