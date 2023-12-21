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

# Tutorial for Developers:
## Installation
A list of necessary packages can be found in nsb_dev.def, which is a singularity container definition file. The easiest way to interact with the package is to compile this into a container:

singularity build nsb_dev.sif nsb_def.def

## Executing Code
Enter the singularity container via

singularity shell nsb_dev.sif

You also need to add the nsb2 folder to your PYTHONPATH via

export <path-to-nsb-2> $PYTHONPATH
