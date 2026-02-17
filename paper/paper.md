---
title: 'nsb2: Simulating Night Sky Background for IACTs'
tags:
  - Python
  - astronomy
  - gamma rays
  - simulation
  - imaging air cherenkov telescopes
authors:
  - name: Gerrit Roellinghoff
    orcid: 0000-0002-9824-9597
    equal-contrib: true
    corresponding: true
    affiliation: "1"
affiliations:
 - name: Erlangen Centre for Astroparticle Physics, Friedrich-Alexander-Universität Erlangen
   index: 1
date: 25 March 2025
bibliography: paper.bib
---

# Summary
Imaging Air Cherenkov Telescopes (IACTs) are widely used in gamma-ray astronomy 
to capture light emission caused by superluminal charged particles on 
nanosecond timescales. Optical light from other light sources in the night sky
(such as stars) pose an irreducible background for IACTs. `nsb2` is a Python3 
package that enables the simulation of this background given telescope parameters 
such as effective aperture and bandpass, which can be derived from ray-tracing 
simulations. `nsb2` includes a variety of light sources and atmospheric models 
common to optical astronomy. It can treat both atmospheric extinction and scattering 
of light in a modular manner, enabling the parallel computation of different 
observation scenarios. The modular approach enables seamless inclusion of custom
atmosphere, light source and telescope models.

# Statement of need
Night sky background (NSB) is the main background for IACTs besides electronical noise.
As of March 2025, there is no public tool in existance besides `nsb2` that enables the 
simulation of pixel-wise NSB for IACTs based on realistic data. The ability to simulate
NSB for different instrument types and locations provides the astronomical community 
with a new tool for instrument design, instrument monitoring and observation planning.
Using it to augment the simulation of IACT event images can reduce the mismatch between 
simulation and real data, a problem acutely felt in modern machine learning based event 
reconstruction techniques. Upcoming IACT observatories such as CTAO, LACT and 
ASTRI Mini-Array can use `nsb2` to evaluate the susceptibility of their data processing
pipeline to different NSB conditions and test NSB based algorithms (such as pointing 
reconstruction) on realistic data. 

`nsb2` has already been evaluated on data taken by the H.E.S.S. telescopes (cite??) and 
used to calculate expected NSB rates for CTAO telescope designs (cite??). The integration 
with Astropy and ctapipe means `nsb2` seamlessly integrates with common software used in 
gamma-ray astronomy, enabling fast adoption in the field.

# Method
`nsb2` simulations work by defining three components of a chain: An instrument, an atmosphere, 
and an emitter. These are bundled in a Method, which determines if the propagation of light 
is done explicitly or if intermediate look-up-tables are computed. Methods can be bundle in 
a MultiMethod object, which simplifies simultaneous creation of Method objects and the 
evaluation thereof.

Instruments consist ouf of a collection of pixels and an instrument bandpass. For each pixel, 
an effective apterture object is calculated. This describes the effective aperture of the pixel
to a point source at infinity as a 2D function in telescope coordinates (a spherical 
coordinate system with (0,0) being aligned with center of the telescope field of view). Given a 
location on earth, rotation of the telescope around its optical axis and a pointing direction 
in local AltAz coordinates at a time T, the instrument can query a source object. This works in 
two ways: For the direct line-of-sight contribution from a source with atmospheric extinction 
the source is queried within the field of view. For the indirect contribution of light sources 
due to in-scattering, the source is queried over the visible hemisphere.

Atmospheres support two modes: Atmospheric extinction depending on the (alt, az) coordinates 
of the source, and atmospheric in-scattering, depending on the projected position of instrument 
pixels on the sky (alt_pixel, az_pixel) and the position of the sources contributing 
(alt_source, az_source). Both of these accept an array of wavelengths lambda, for which they compute
the relative contribution.

Emitters can be based on a catalog (a collection of spherical coordinates with an associated
data vector d), a map (a HEALPix representation of d for each tile) or a function (returning 
d for a pair of spherical coordinates). The emission spectra is based on a spectral grid of 
(lambda, d_0, d_1, ..., d_N), with wavelength lambda. When queried by an instrument observation, 
an emitter queries the catalog/map/function for vectors d and linearly interpolates 
the spectral grid over finite values in d. It returns a spectral grid object of dimension N_lambda + N_nf,
where N_nf is equal to the amount of non finite values in d, for each queried emitter. 
This spectral grid is further weighted with the bandpass of the instrument. 



# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures


# Acknowledgements


# References