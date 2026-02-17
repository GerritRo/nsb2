Introduction
============

nsb2 is a Python library for simulating the night sky background (NSB) as seen
by imaging air Cherenkov telescopes (IACTs). It provides a modular framework for
combining emission models, atmospheric effects, and telescope responses into a
complete simulation pipeline.

What is nsb2?
-------------

nsb2 provides a computational framework for:

- **NSB emission modeling**: Simulate multiple sources of night sky background
  including stars, zodiacal light, moonlight and airglow.

- **Atmospheric effects**: Model Rayleigh and Mie scattering, absorption, and
  single-scattering albedo through the atmosphere.

- **Telescope response**: Characterize how IACTS respond to background light,
  using raytraced pixel reponses.

- **Pipeline simulation**: Combine different types of pipelines and solvers, 
  be it explicit evaluation or look-up tables.

Target Audience
---------------

This library is designed for observation planning for IACTs, generating training 
data for machine learning algorithms and instrument monitoring. For a more 
complex implementation, take a look at `NYX <https://github.com/GerritRo/nyx/>`_

The library assumes familiarity with Python, NumPy-style array programming, and
basic atmospheric optics concepts.

License
-------

nsb2 is released under the BSD-3-Clause license. See the
`LICENSE <https://github.com/GerritRo/nsb2/blob/main/LICENSE>`_
file for details.

Citation
--------

If you use nsb2 in your research, please cite::

   @misc{roellinghoff_2025_nsb2,
       author = {Roellinghoff, Gerrit; Spencer, Samuel},
       title = {nsb2: Simulating Night Sky Background for IACTs},
       url = {https://github.com/GerritRo/nsb2},
       year = {2025}
   }

Please also cite the origin of all telescope, atmospheric, and emission models
you use in your work.
