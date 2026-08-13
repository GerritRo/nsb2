.. _acknowledgements:

****************
Acknowledgements
****************

nsb2 is built on survey data and software whose authors ask to be
acknowledged. Those requests pass through to you: if you publish results
produced with nsb2, please include the statements below for the models and
features you used, alongside the citations listed in :ref:`bibliography`.

Gaia
====

The bundled star catalogue and integrated sky map
(:func:`~nsb2.emitter.stars.from_gaia_dr3_catalog` and
:func:`~nsb2.emitter.stars.from_gaia_dr3_map`) are derived from Gaia DR3, and
the passbands they are calibrated against are the Gaia DR3 passbands.  The
European Space Agency asks that work using Gaia data carry the following
acknowledgement:

   This work presents results from the European Space Agency (ESA) space
   mission Gaia. Gaia data are being processed by the Gaia Data Processing
   and Analysis Consortium (DPAC). Funding for the DPAC is provided by
   national institutions, in particular the institutions participating in
   the Gaia MultiLateral Agreement (MLA). The Gaia mission website is
   https://www.cosmos.esa.int/gaia. The Gaia archive website is
   https://archives.esac.esa.int/gaia.

Please also cite the Gaia mission [GaiaMission2016]_ and the data release
[GaiaDR3]_.

HEALPix and healpy
==================

nsb2 uses the HEALPix [Gorski2005]_ pixelisation through healpy
[Zonca2019]_ to bin catalogues onto the sphere
(:meth:`~nsb2.core.sources.CatalogSource.to_map`,
:class:`~nsb2.core.sources.HEALPixSource`) and to sample the hemisphere for
in-scattering (:class:`~nsb2.core.lightpath.ScatteredPath`).  Any prediction
involving a scattered light path uses it. The HEALPix project asks for:

   Some of the results in this paper have been derived using the healpy and
   HEALPix package.

with a footnote at first mention pointing to https://healpix.sourceforge.net.

SVO Filter Profile Service
==========================

Passbands fetched with :meth:`~nsb2.core.spectral.Bandpass.from_SVO` come from
the Spanish Virtual Observatory Filter Profile Service, including the Gaia DR3
passbands the star catalogues are calibrated against. Acknowledge via:

   This research has made use of the SVO Filter Profile Service
   (http://svo2.cab.inta-csic.es/theory/fps/) supported from the Spanish
   MINECO through grant AYA2017-84089 and described in
   `Rodrigo et al. (2012)
   <https://ui.adsabs.harvard.edu/abs/2012ivoa.rept.1015R/abstract>`_ and
   `Rodrigo et al. (2020)
   <https://ui.adsabs.harvard.edu/abs/2020sea..confE.182R/abstract>`_.

Those two references are [Rodrigo2012]_ and [Rodrigo2020]_.

Reference data
==============

The emission and instrument models carry their own attribution. Each is
cited in the docstring of the function that implements it, and collected in
:ref:`bibliography`.

Please cite the origin of every telescope, atmospheric and emission model you
use, not only nsb2 itself.
