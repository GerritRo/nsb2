"""Tests for the code paths that fetch external reference data.

The downloads themselves are exercised in ``test_remote_emitters.py``; here
``download_file`` is replaced with a local file so the parsing, integration
and error handling around it can be checked offline.  These are the paths
that broke when STScI retired a CALSPEC revision, so they are worth pinning.
"""

from urllib.error import HTTPError

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from nsb2.conftest import stub_download
from nsb2.core.photometry import SolarSpectrumRieke2008
from nsb2.core.spectral import CALSPEC_URL, Bandpass


def _write_spectrum_fits(path, wavelength_angstrom, flux):
    """Write a CALSPEC-shaped FITS table with WAVELENGTH and FLUX columns."""
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="WAVELENGTH", format="E", array=wavelength_angstrom),
            fits.Column(name="FLUX", format="E", array=flux),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
    return path


class TestVegazero:
    def test_integrates_the_reference_spectrum(self, monkeypatch, tmp_path):
        """A flat spectrum gives the analytic photon-weighted integral."""
        wvl = np.linspace(4000.0, 6000.0, 201)
        path = _write_spectrum_fits(tmp_path / "vega.fits", wvl, np.ones_like(wvl))
        stub_download(monkeypatch, "nsb2.core.spectral", path)

        bandpass = Bandpass(np.linspace(400, 600, 50) * u.nm, np.ones(50))
        # integral of lambda * 1 * 1 dlambda over [4000, 6000] angstrom
        expected = 0.5 * (6000.0**2 - 4000.0**2)
        assert bandpass.vegazero.value == pytest.approx(expected, rel=1e-4)

    def test_carries_the_wavelength_unit(self, monkeypatch, tmp_path):
        wvl = np.linspace(4000.0, 6000.0, 51)
        path = _write_spectrum_fits(tmp_path / "vega.fits", wvl, np.ones_like(wvl))
        stub_download(monkeypatch, "nsb2.core.spectral", path)

        bandpass = Bandpass(np.linspace(400, 600, 20) * u.nm, np.ones(20))
        assert bandpass.vegazero.unit.is_equivalent(u.erg / u.s / u.cm**2 * u.angstrom)

    def test_is_cached_after_the_first_access(self, monkeypatch, tmp_path):
        wvl = np.linspace(4000.0, 6000.0, 51)
        path = _write_spectrum_fits(tmp_path / "vega.fits", wvl, np.ones_like(wvl))
        calls = []

        def counting_download(url, *args, **kwargs):
            calls.append(url)
            return str(path)

        stub_download(monkeypatch, "nsb2.core.spectral", counting_download)
        bandpass = Bandpass(np.linspace(400, 600, 20) * u.nm, np.ones(20))
        _ = bandpass.vegazero
        _ = bandpass.vegazero
        assert len(calls) == 1

    def test_retired_revision_gives_an_actionable_error(self, monkeypatch):
        """STScI removes superseded CALSPEC revisions; a 404 must explain that."""

        def gone(url, *args, **kwargs):
            raise HTTPError(url, 404, "Not Found", {}, None)

        stub_download(monkeypatch, "nsb2.core.spectral", gone)
        bandpass = Bandpass(np.linspace(400, 600, 20) * u.nm, np.ones(20))
        with pytest.raises(RuntimeError, match="VEGA_CALSPEC_FILE"):
            _ = bandpass.vegazero

    def test_error_message_names_the_file_and_url(self, monkeypatch):
        def gone(url, *args, **kwargs):
            raise HTTPError(url, 404, "Not Found", {}, None)

        stub_download(monkeypatch, "nsb2.core.spectral", gone)
        bandpass = Bandpass(np.linspace(400, 600, 20) * u.nm, np.ones(20))
        with pytest.raises(RuntimeError, match="alpha_lyr_stis"):
            _ = bandpass.vegazero
        with pytest.raises(RuntimeError, match=CALSPEC_URL):
            _ = Bandpass(np.linspace(400, 600, 20) * u.nm, np.ones(20)).vegazero

    def test_other_http_errors_propagate_unchanged(self, monkeypatch):
        """Only a 404 means a retired revision; a 500 is a server problem."""

        def broken(url, *args, **kwargs):
            raise HTTPError(url, 500, "Server Error", {}, None)

        stub_download(monkeypatch, "nsb2.core.spectral", broken)
        bandpass = Bandpass(np.linspace(400, 600, 20) * u.nm, np.ones(20))
        with pytest.raises(HTTPError) as excinfo:
            _ = bandpass.vegazero
        assert excinfo.value.code == 500

    def test_respects_an_overridden_revision(self, monkeypatch, tmp_path):
        """The module constant is the documented override point."""
        wvl = np.linspace(4000.0, 6000.0, 51)
        path = _write_spectrum_fits(tmp_path / "vega.fits", wvl, np.ones_like(wvl))
        requested = []

        def record(url, *args, **kwargs):
            requested.append(url)
            return str(path)

        stub_download(monkeypatch, "nsb2.core.spectral", record)
        monkeypatch.setattr(
            "nsb2.core.spectral.VEGA_CALSPEC_FILE", "alpha_lyr_stis_999.fits"
        )
        _ = Bandpass(np.linspace(400, 600, 20) * u.nm, np.ones(20)).vegazero
        assert requested[0].endswith("alpha_lyr_stis_999.fits")


def _write_filter_votable(path, wavelengths, transmissions):
    """Write an SVO-shaped VOTable of wavelength and transmission."""
    rows = "\n".join(
        f"   <TR><TD>{w}</TD><TD>{t}</TD></TR>"
        for w, t in zip(wavelengths, transmissions, strict=True)
    )
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VOTABLE version="1.1" xmlns="http://www.ivoa.net/xml/VOTable/v1.1">\n'
        " <RESOURCE><TABLE>\n"
        '  <FIELD name="Wavelength" datatype="double" unit="Angstrom"/>\n'
        '  <FIELD name="Transmission" datatype="double"/>\n'
        f"  <DATA><TABLEDATA>\n{rows}\n  </TABLEDATA></DATA>\n"
        " </TABLE></RESOURCE>\n"
        "</VOTABLE>\n"
    )
    return path


class TestBandpassFromSVO:
    def test_parses_a_votable(self, monkeypatch, tmp_path):
        """The SVO service returns a VOTable of wavelength and transmission."""
        wavelengths = np.linspace(4000, 6000, 11)
        votable = _write_filter_votable(
            tmp_path / "filter.xml", wavelengths, np.ones(11)
        )
        stub_download(monkeypatch, "nsb2.core.spectral", votable)

        bandpass = Bandpass.from_SVO("GAIA/GAIA3.G")
        assert bandpass.min == 4000 * u.angstrom
        assert bandpass.max == 6000 * u.angstrom
        assert bandpass(np.array([5000]) * u.angstrom)[0] == pytest.approx(1.0)

    def test_transmission_is_zero_outside_the_tabulated_range(
        self, monkeypatch, tmp_path
    ):
        wavelengths = np.linspace(4000, 6000, 11)
        votable = _write_filter_votable(
            tmp_path / "filter.xml", wavelengths, np.ones(11)
        )
        stub_download(monkeypatch, "nsb2.core.spectral", votable)

        bandpass = Bandpass.from_SVO("GAIA/GAIA3.G")
        assert bandpass(np.array([3000, 7000]) * u.angstrom) == pytest.approx(0.0)

    def test_passes_the_filter_id_in_the_url(self, monkeypatch, tmp_path):
        votable = _write_filter_votable(
            tmp_path / "filter.xml", np.linspace(4000, 6000, 11), np.ones(11)
        )
        requested = []

        def record(url, *args, **kwargs):
            requested.append(url)
            return str(votable)

        stub_download(monkeypatch, "nsb2.core.spectral", record)
        Bandpass.from_SVO("OSN/Johnson.V")
        assert requested[0].endswith("OSN/Johnson.V")


class TestSolarSpectrumRieke2008:
    def test_returns_wavelength_and_flux_with_units(self, monkeypatch, tmp_path):
        wvl = np.linspace(3000.0, 10000.0, 71)
        path = _write_spectrum_fits(tmp_path / "solar.fits", wvl, np.ones_like(wvl))
        stub_download(monkeypatch, "nsb2.core.photometry", path)

        wavelength, flux = SolarSpectrumRieke2008()
        assert wavelength.unit == u.angstrom
        assert flux.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.angstrom)
        assert wavelength.shape == flux.shape == (71,)

    def test_values_survive_closing_the_file(self, monkeypatch, tmp_path):
        """The arrays must be copies, not memory-mapped views of a closed file."""
        wvl = np.linspace(3000.0, 10000.0, 71)
        path = _write_spectrum_fits(tmp_path / "solar.fits", wvl, np.full(71, 3.5))
        stub_download(monkeypatch, "nsb2.core.photometry", path)

        _, flux = SolarSpectrumRieke2008()
        assert np.all(flux.value == pytest.approx(3.5))
