<!-- language: lang-none -->
                   __   ___ 
       ____  _____/ /_ |__ \
      / __ \/ ___/ __ \__/ /
     / / / (__  ) /_/ / __/ 
    /_/ /_/____/_.___/____/


# nsb2: Simulating Night Sky Background for IACTs

This is a refactored and reworked version of Matthias Buecheles [nsb](https://pypi.org/project/nsb/) package. It includes the following improvements:

- Per Pixel NSB rates
- Modularity + Support for other IACTs
- More physical
- Significantly faster

# Installation
This package can be installed via pip via:

pip install git+https://git.ecap.work/groellinghoff/nsb2.git

Some of the supplementary data (fits file and gaia catalogs) are large and not on this git repo. They can either be created/downloaded manually (see the blacksky examples on how to create a gaia catalog/map). Alternatively, you can get the larger files directly from me.
