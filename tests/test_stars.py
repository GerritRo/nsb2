"""Tests for the stars emitter module."""

from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest

from nsb2.core.sources import CatalogSource, HEALPixSource
from tests.conftest import make_bandpass, make_spectral_grid


def _make_fake_gaia_catalog(n=20):
    """Create a minimal structured array mimicking a Gaia DR3 catalog."""
    dtype = np.dtype(
        [
            ("ra", float),
            ("dec", float),
            ("phot_g_mean_mag", float),
            ("phot_rp_mean_mag", float),
            ("phot_bp_mean_mag", float),
        ]
    )
    data = np.zeros(n, dtype=dtype)
    data["ra"] = np.linspace(0, 300, n)
    data["dec"] = np.zeros(n)
    data["phot_g_mean_mag"] = 12.0
    data["phot_rp_mean_mag"] = 11.5
    data["phot_bp_mean_mag"] = 12.5
    return data


class TestFromGaiaSupplCatalog:
    @patch("nsb2.core.sources.create_color_grid")
    @patch("nsb2.emitter.stars.Bandpass.from_SVO")
    def test_returns_catalog_source(self, mock_svo, mock_ccg):
        mock_svo.return_value = make_bandpass()
        mock_ccg.return_value = make_spectral_grid()
        from nsb2.emitter.stars import from_gaia_suppl_catalog

        result = from_gaia_suppl_catalog()
        assert isinstance(result, CatalogSource)

    @patch("nsb2.core.sources.create_color_grid")
    @patch("nsb2.emitter.stars.Bandpass.from_SVO")
    def test_has_stars(self, mock_svo, mock_ccg):
        mock_svo.return_value = make_bandpass()
        mock_ccg.return_value = make_spectral_grid()
        from nsb2.emitter.stars import from_gaia_suppl_catalog

        result = from_gaia_suppl_catalog()
        assert len(result.coords) > 0

    @patch("nsb2.core.sources.create_color_grid")
    @patch("nsb2.emitter.stars.Bandpass.from_SVO")
    def test_name_is_set(self, mock_svo, mock_ccg):
        mock_svo.return_value = make_bandpass()
        mock_ccg.return_value = make_spectral_grid()
        from nsb2.emitter.stars import from_gaia_suppl_catalog

        result = from_gaia_suppl_catalog()
        assert result.name != ""


class TestFromGaiaDR3Catalog:
    @patch("nsb2.core.sources.create_color_grid")
    @patch("nsb2.emitter.stars.Bandpass.from_SVO")
    @patch("nsb2.emitter.stars.download_file")
    def test_returns_catalog_source(self, mock_download, mock_svo, mock_ccg, tmp_path):
        fake_gaia = _make_fake_gaia_catalog(n=20)
        npy_file = tmp_path / "fake_gaia.npy"
        np.save(npy_file, fake_gaia)
        mock_download.return_value = str(npy_file)
        mock_svo.return_value = make_bandpass()
        mock_ccg.return_value = make_spectral_grid()
        from nsb2.emitter.stars import from_gaia_dr3_catalog

        result = from_gaia_dr3_catalog()
        assert isinstance(result, CatalogSource)

    @patch("nsb2.core.sources.create_color_grid")
    @patch("nsb2.emitter.stars.Bandpass.from_SVO")
    @patch("nsb2.emitter.stars.download_file")
    def test_has_expected_count(self, mock_download, mock_svo, mock_ccg, tmp_path):
        fake_gaia = _make_fake_gaia_catalog(n=15)
        npy_file = tmp_path / "fake_gaia.npy"
        np.save(npy_file, fake_gaia)
        mock_download.return_value = str(npy_file)
        mock_svo.return_value = make_bandpass()
        mock_ccg.return_value = make_spectral_grid()
        from nsb2.emitter.stars import from_gaia_dr3_catalog

        result = from_gaia_dr3_catalog()
        assert len(result.coords) == 15


class TestFromGaiaDR3Map:
    @patch("nsb2.core.sources.create_color_grid")
    @patch("nsb2.emitter.stars.Bandpass.from_SVO")
    @patch("nsb2.emitter.stars.download_file")
    def test_returns_healpix_source(self, mock_download, mock_svo, mock_ccg, tmp_path):
        import healpy as hp

        nside = 4
        npix = hp.nside2npix(nside)
        # mag_map[0]=G, [1]=BP, [2]=RP
        mag_map = np.full((3, npix), 12.0)
        npy_file = tmp_path / "fake_mag_map.npy"
        np.save(npy_file, mag_map)
        mock_download.return_value = str(npy_file)
        mock_svo.return_value = make_bandpass()
        mock_ccg.return_value = make_spectral_grid()
        from nsb2.emitter.stars import from_gaia_dr3_map

        result = from_gaia_dr3_map()
        assert isinstance(result, HEALPixSource)
