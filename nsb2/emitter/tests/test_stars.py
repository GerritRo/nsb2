import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from nsb2.conftest import calibrated_bandpass, stub_download
from nsb2.core.sources import CatalogSource, HEALPixSource
from nsb2.core.spectral import SpectralGrid
from nsb2.emitter import stars
from nsb2.emitter.stars import (
    GAIA_SPLIT_MAGNITUDE,
    MAX_RP_BP,
    _rp_bp_color,
    from_gaia_dr3_catalog,
    from_gaia_dr3_map,
)

NSIDE = 4
NPIX = 12 * NSIDE**2


def _synthetic_catalog(n=40, seed=1):
    """A structured array with the fields from_gaia_dr3_catalog reads."""
    rng = np.random.default_rng(seed)
    catalog = np.zeros(
        n,
        dtype=[
            ("ra", "f8"),
            ("dec", "f8"),
            ("phot_g_mean_mag", "f8"),
            ("phot_bp_mean_mag", "f8"),
            ("phot_rp_mean_mag", "f8"),
        ],
    )
    catalog["ra"] = rng.uniform(0, 360, n)
    catalog["dec"] = rng.uniform(-90, 90, n)
    catalog["phot_g_mean_mag"] = rng.uniform(2, GAIA_SPLIT_MAGNITUDE, n)
    catalog["phot_bp_mean_mag"] = catalog["phot_g_mean_mag"] + rng.uniform(0, 1, n)
    catalog["phot_rp_mean_mag"] = catalog["phot_g_mean_mag"] - rng.uniform(0, 1, n)
    return catalog


def _synthetic_map(seed=2):
    """A (3, npix) array of G, BP and RP magnitudes per HEALPix cell."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(GAIA_SPLIT_MAGNITUDE, 20, NPIX)
    return np.stack([g, g + rng.uniform(0, 1, NPIX), g - rng.uniform(0, 1, NPIX)])


def _stub_photometry(monkeypatch):
    """Replace the SVO passbands and template library with local stand-ins."""
    monkeypatch.setattr(
        stars.Bandpass,
        "from_SVO",
        classmethod(lambda cls, *a, **k: calibrated_bandpass()),
    )
    # A four-template library keeps create_color_grid fast.
    monkeypatch.setattr(
        stars,
        "PicklesTRDSAtlas1998",
        lambda: SpectralGrid(
            [],
            np.linspace(3000, 11000, 60) * u.angstrom,
            np.ones((60, 4)) * u.erg / u.angstrom / u.s / u.cm**2,
        ),
    )


@pytest.fixture
def offline_gaia(monkeypatch, tmp_path):
    """Serve synthetic Gaia products and stub the SVO passband downloads."""
    catalog_path = tmp_path / "gaiadr3.npy"
    map_path = tmp_path / "gaia_mag15plus.npy"
    np.save(catalog_path, _synthetic_catalog())
    np.save(map_path, _synthetic_map())

    def fake_download(url, *args, **kwargs):
        return str(map_path if "mag15plus" in url else catalog_path)

    stub_download(monkeypatch, "nsb2.emitter.stars", fake_download)
    _stub_photometry(monkeypatch)


class TestRpBpColor:
    def test_is_a_difference_that_is_gap_filled_and_clipped(self):
        np.testing.assert_allclose(
            _rp_bp_color(np.array([5.0, 6.0]), np.array([5.2, 6.4])), [-0.2, -0.4]
        )
        assert _rp_bp_color(np.array([10.0]), np.array([5.0]))[0] == pytest.approx(
            MAX_RP_BP
        )
        filled = _rp_bp_color(np.array([5.0, np.nan, 9.0]), np.full(3, 5.0))
        assert filled[1] == pytest.approx(0.0)
        assert np.all(
            np.isfinite(_rp_bp_color(np.array([np.inf, 4.0]), np.full(2, 4.0)))
        )

        rng = np.random.default_rng(0)
        assert np.all(
            _rp_bp_color(rng.uniform(0, 20, 200), rng.uniform(0, 20, 200)) <= MAX_RP_BP
        )

    def test_does_not_modify_its_inputs(self):
        rp = np.array([5.0, np.nan])
        bp = np.array([np.nan, 6.0])
        rp_before, bp_before = rp.copy(), bp.copy()
        _rp_bp_color(rp, bp)
        np.testing.assert_array_equal(rp, rp_before, strict=False)
        np.testing.assert_array_equal(bp, bp_before, strict=False)


class TestFromGaiaDr3Catalog:
    def test_builds_a_point_source_catalogue(self, offline_gaia):
        """Brightness weight is 10**(-0.4 m), so brighter stars weigh more."""
        catalog = from_gaia_dr3_catalog()
        assert isinstance(catalog, CatalogSource)
        assert len(catalog.coords) == 40
        assert catalog.name == f"GaiaDR3_G<{GAIA_SPLIT_MAGNITUDE}"
        assert catalog.coords.frame.name == "icrs"
        assert catalog.weight.unit == u.dimensionless_unscaled
        assert np.all(catalog.weight.value > 0)

        magnitudes = _synthetic_catalog()["phot_g_mean_mag"]
        brightest, faintest = np.argmin(magnitudes), np.argmax(magnitudes)
        assert catalog.weight[brightest, 0] > catalog.weight[faintest, 0]

    def test_is_queryable_and_can_be_binned_into_a_map(
        self, offline_gaia, observation, instrument
    ):
        catalog = from_gaia_dr3_catalog()
        catalog.build_balltree()
        field, refs = catalog.query_direct(
            observation, instrument.pixel_coords(observation), instrument.pixel_radii()
        )
        assert len(refs.indices) == instrument.n_pixels
        assert field.radiance_field is False
        assert isinstance(catalog.to_map(nside=8), HEALPixSource)


class TestFromGaiaDr3Map:
    def test_builds_a_radiance_map(self, offline_gaia):
        """Magnitudes are divided by the cell solid angle, giving 1/sr.

        Integrating that radiance back over the sky recovers the total flux.
        """
        healpix = from_gaia_dr3_map()
        assert isinstance(healpix, HEALPixSource)
        assert healpix.weight.shape[-1] == NPIX
        assert healpix.name == f"GaiaDR3_G>{GAIA_SPLIT_MAGNITUDE}"
        assert healpix.frame == "icrs"
        assert healpix.weight.unit.is_equivalent(1 / u.sr)

        total = (healpix.weight * (4 * np.pi / NPIX * u.sr)).sum()
        expected = (10 ** (-0.4 * _synthetic_map()[0])).sum()
        assert float(total.to_value(u.dimensionless_unscaled)) == pytest.approx(
            expected
        )

    def test_is_queryable_end_to_end(self, offline_gaia, observation):
        field = from_gaia_dr3_map().query_scattered(observation, nside=NSIDE)
        assert field.radiance_field is True
        assert np.all(field.coords.alt.rad > 0)


class TestBundledCatalogues:
    def test_fetches_the_three_gaia_bands(self, monkeypatch):
        requested = []
        monkeypatch.setattr(
            stars.Bandpass,
            "from_SVO",
            classmethod(lambda cls, name, *a, **k: requested.append(name)),
        )
        stars._gaia_bandpasses()
        assert requested == ["GAIA/GAIA3.G", "GAIA/GAIA3.Gbp", "GAIA/GAIA3.Grp"]

    def test_supplementary_catalogue_comes_from_the_bundled_xhip_table(
        self, monkeypatch
    ):
        """The supplementary catalogue ships with nsb2; only SVO is remote."""
        _stub_photometry(monkeypatch)
        catalog = stars.from_gaia_suppl_catalog()
        assert isinstance(catalog, CatalogSource)
        assert catalog.name == "XHIP_Gaia_Suppl"
        assert isinstance(catalog.coords, SkyCoord)
        assert len(catalog.coords) > 0
