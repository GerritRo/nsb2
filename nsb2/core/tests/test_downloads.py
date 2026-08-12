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


def _stub_http_error(monkeypatch, code):
    """Make every download raise an HTTP error with the given status."""

    def failing(url, *args, **kwargs):
        raise HTTPError(url, code, "Failed", {}, None)

    stub_download(monkeypatch, "nsb2.core.spectral", failing)


def _top_hat(n=20):
    return Bandpass(np.linspace(400, 600, n) * u.nm, np.ones(n))


@pytest.fixture
def stubbed_calspec(monkeypatch, tmp_path):
    """Serve a flat reference spectrum and record the URLs asked for."""
    wvl = np.linspace(4000.0, 6000.0, 201)
    path = _write_spectrum_fits(tmp_path / "vega.fits", wvl, np.ones_like(wvl))
    requested = []

    def record(url, *args, **kwargs):
        requested.append(url)
        return str(path)

    stub_download(monkeypatch, "nsb2.core.spectral", record)
    return requested


class TestVegazero:
    def test_integrates_the_reference_spectrum_once(self, stubbed_calspec):
        """A flat spectrum gives the analytic photon-weighted integral.

        The cached property must not download it again on second access.
        """
        bandpass = Bandpass(np.linspace(400, 600, 50) * u.nm, np.ones(50))
        # integral of lambda * 1 * 1 dlambda over [4000, 6000] angstrom
        expected = 0.5 * (6000.0**2 - 4000.0**2)
        assert bandpass.vegazero.value == pytest.approx(expected, rel=1e-4)
        assert bandpass.vegazero.unit.is_equivalent(u.erg / u.s / u.cm**2 * u.angstrom)
        assert len(stubbed_calspec) == 1

    def test_respects_an_overridden_revision(self, stubbed_calspec, monkeypatch):
        """The module constant is the documented override point.

        The file it names is what ends up being requested.
        """
        monkeypatch.setattr(
            "nsb2.core.spectral.VEGA_CALSPEC_FILE", "alpha_lyr_stis_999.fits"
        )
        _ = _top_hat().vegazero
        assert stubbed_calspec[0].startswith(CALSPEC_URL)
        assert stubbed_calspec[0].endswith("alpha_lyr_stis_999.fits")

    def test_retired_revision_gives_an_actionable_error(self, monkeypatch):
        """STScI removes superseded CALSPEC revisions.

        A 404 must explain that, and name the constant, the file and the URL
        to look at.
        """
        _stub_http_error(monkeypatch, 404)
        for pattern in ("VEGA_CALSPEC_FILE", "alpha_lyr_stis", CALSPEC_URL):
            with pytest.raises(RuntimeError, match=pattern):
                _ = _top_hat().vegazero

    def test_other_http_errors_propagate_unchanged(self, monkeypatch):
        """Only a 404 means a retired revision; a 500 is a server problem."""
        _stub_http_error(monkeypatch, 500)
        with pytest.raises(HTTPError) as excinfo:
            _ = _top_hat().vegazero
        assert excinfo.value.code == 500


class TestBandpassFromSVO:
    def test_parses_a_votable(self, monkeypatch, tmp_path):
        """The SVO service returns a VOTable of wavelength and transmission.

        It is keyed by the filter id passed through in the URL.
        """
        wavelengths = np.linspace(4000, 6000, 11)
        rows = "\n".join(f"   <TR><TD>{w}</TD><TD>1.0</TD></TR>" for w in wavelengths)
        path = tmp_path / "filter.xml"
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
        requested = []

        def record(url, *args, **kwargs):
            requested.append(url)
            return str(path)

        stub_download(monkeypatch, "nsb2.core.spectral", record)

        bandpass = Bandpass.from_SVO("OSN/Johnson.V")
        assert requested[0].endswith("OSN/Johnson.V")
        assert bandpass.min == 4000 * u.angstrom
        assert bandpass.max == 6000 * u.angstrom
        assert bandpass(np.array([5000]) * u.angstrom)[0] == pytest.approx(1.0)
        assert bandpass(np.array([3000, 7000]) * u.angstrom) == pytest.approx(0.0)


class TestSolarSpectrumRieke2008:
    def test_returns_wavelength_and_flux_that_outlive_the_file(
        self, monkeypatch, tmp_path
    ):
        """The arrays must be copies, not memory-mapped views of a closed file."""
        wvl = np.linspace(3000.0, 10000.0, 71)
        path = _write_spectrum_fits(tmp_path / "solar.fits", wvl, np.full(71, 3.5))
        stub_download(monkeypatch, "nsb2.core.photometry", path)

        wavelength, flux = SolarSpectrumRieke2008()
        assert wavelength.unit == u.angstrom
        assert flux.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.angstrom)
        assert wavelength.shape == flux.shape == (71,)
        assert np.all(flux.value == pytest.approx(3.5))
