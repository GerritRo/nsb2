"""Tests for the photometry module."""

from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest

from nsb2.core.photometry import PicklesTRDSAtlas1998
from nsb2.core.spectral import SpectralGrid


class TestPicklesTRDSAtlas1998:
    def test_returns_spectral_grid(self):
        sg = PicklesTRDSAtlas1998()
        assert isinstance(sg, SpectralGrid)

    def test_wavelength_in_angstrom(self):
        sg = PicklesTRDSAtlas1998()
        assert sg.wvl.unit == u.angstrom

    def test_wavelength_range_optical(self):
        sg = PicklesTRDSAtlas1998()
        assert sg.wvl.min() < 4000 * u.angstrom
        assert sg.wvl.max() > 8000 * u.angstrom

    def test_flux_unit(self):
        sg = PicklesTRDSAtlas1998()
        expected = u.erg / u.angstrom / u.s / u.cm**2
        assert sg.flx.unit.is_equivalent(expected)

    def test_multiple_spectra(self):
        """Pickles atlas has multiple stellar spectra as columns."""
        sg = PicklesTRDSAtlas1998()
        assert sg.flx.ndim >= 2
        assert sg.flx.shape[-1] > 1

    def test_no_parameter_dimensions(self):
        sg = PicklesTRDSAtlas1998()
        assert len(sg.points) == 0


class TestSolarSpectrumRieke2008Mocked:
    def test_returns_wvl_and_flux(self):
        from nsb2.core.photometry import SolarSpectrumRieke2008

        mock_hdul = MagicMock()
        mock_hdul.__getitem__ = lambda self, idx: MagicMock(
            data={
                "WAVELENGTH": np.linspace(1000, 30000, 200),
                "FLUX": np.ones(200) * 1.5e-8,
            }
        )
        with patch("nsb2.core.photometry.download_file", return_value="/fake/path"):
            with patch("nsb2.core.photometry.fits.open", return_value=mock_hdul):
                wvl, flx = SolarSpectrumRieke2008()
        assert wvl.unit == u.angstrom
        assert flx.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.angstrom)
        assert len(wvl) == 200

    def test_wvl_and_flux_same_length(self):
        from nsb2.core.photometry import SolarSpectrumRieke2008

        mock_hdul = MagicMock()
        mock_hdul.__getitem__ = lambda self, idx: MagicMock(
            data={
                "WAVELENGTH": np.linspace(2000, 20000, 150),
                "FLUX": np.ones(150) * 2.0e-8,
            }
        )
        with patch("nsb2.core.photometry.download_file", return_value="/fake/path"):
            with patch("nsb2.core.photometry.fits.open", return_value=mock_hdul):
                wvl, flx = SolarSpectrumRieke2008()
        assert len(wvl) == len(flx)
