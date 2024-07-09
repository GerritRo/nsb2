<!-- language: lang-none -->
                   __   ___ 
       ____  _____/ /_ |__ \
      / __ \/ ___/ __ \__/ /
     / / / (__  ) /_/ / __/ 
    /_/ /_/____/_.___/____/


# nsb2: Simulating Night Sky Background for IACTs

nsb2 is a python library for calculating the effect of night sky background (NSB) on telescopes with large pixel areas, namely imaging air cherenkov telescopes (IACTs). It includes models from theory / experimental data for star light, zodiacal light, moon light, diffuse galactic light and airglow. It also models atmospheric effects, such as rayleigh scattering, mie scattering, single scattering albedo and aerosol concentration. 

nsb2 offers an interface with ctapipe, enabling adhoc simulation of simple telescope models from ctapipe geometries as well as making use of ctapipe plotting capability. Depending on the complexity of the starfield simulated, nsb2 can simulate NSB on the order of ~1sec per field.

# Installation
This package can be installed via pip via:

pip install git+https://git.ecap.work/groellinghoff/nsb2.git

nsb2 requires blacksky, histlite, numpy, scikit-learn, astropy and healpy.

Some of the supplementary data (fits file and gaia catalogs) are large and not on this git repo. They can either be created/downloaded manually (see the blacksky examples on how to create a gaia catalog/map). Alternatively, you can get the larger files directly from me.

## Optional dependencies
For plotting, nsb2 relies on ctapipe and matplotlib.