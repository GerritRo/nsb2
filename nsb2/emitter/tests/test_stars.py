"""Tests for :mod:`nsb2.emitter.stars`.

The real Gaia products are 2.4 GB, so the catalogue and map factories are
exercised against small synthetic arrays with the same structure, injected by
replacing the module's ``download_file``.  That covers the assembly logic,
the photometric-catalogue constructors and the magnitude-to-radiance
conversion without any network access.
"""

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


class TestRpBpColor:
    def test_plain_difference(self):
        rp = np.array([5.0, 6.0])
        bp = np.array([5.2, 6.4])
        np.testing.assert_allclose(_rp_bp_color(rp, bp), [-0.2, -0.4])

    def test_clips_at_the_red_end(self):
        """No template is redder than MAX_RP_BP, so the index is capped."""
        result = _rp_bp_color(np.array([10.0]), np.array([5.0]))
        assert result[0] == pytest.approx(MAX_RP_BP)

    def test_fills_non_finite_with_the_brightest_magnitude(self):
        rp = np.array([5.0, np.nan, 9.0])
        bp = np.array([5.0, 5.0, 5.0])
        result = _rp_bp_color(rp, bp)
        # The NaN entry is replaced by min(rp) = 5.0, giving a colour of 0.
        assert result[1] == pytest.approx(0.0)

    def test_handles_infinities(self):
        result = _rp_bp_color(np.array([np.inf, 4.0]), np.array([4.0, 4.0]))
        assert np.all(np.isfinite(result))

    def test_does_not_modify_its_inputs(self):
        """The style guide forbids mutating arguments."""
        rp = np.array([5.0, np.nan])
        bp = np.array([np.nan, 6.0])
        rp_before, bp_before = rp.copy(), bp.copy()
        _rp_bp_color(rp, bp)
        np.testing.assert_array_equal(rp, rp_before, strict=False)
        np.testing.assert_array_equal(bp, bp_before, strict=False)

    def test_never_exceeds_the_cap(self):
        rng = np.random.default_rng(0)
        result = _rp_bp_color(rng.uniform(0, 20, 200), rng.uniform(0, 20, 200))
        assert np.all(result <= MAX_RP_BP)


class TestFromGaiaDr3Catalog:
    def test_builds_a_catalog_source(self, offline_gaia):
        catalog = from_gaia_dr3_catalog()
        assert isinstance(catalog, CatalogSource)
        assert len(catalog.coords) == 40

    def test_name_records_the_magnitude_split(self, offline_gaia):
        assert from_gaia_dr3_catalog().name == f"GaiaDR3_G<{GAIA_SPLIT_MAGNITUDE}"

    def test_coordinates_are_icrs(self, offline_gaia):
        assert from_gaia_dr3_catalog().coords.frame.name == "icrs"

    def test_weights_are_flux_ratios(self, offline_gaia):
        """Brightness weight is 10**(-0.4 m), so brighter stars weigh more."""
        catalog = from_gaia_dr3_catalog()
        assert catalog.weight.unit == u.dimensionless_unscaled
        assert np.all(catalog.weight.value > 0)

    def test_brighter_stars_get_larger_weights(self, offline_gaia):
        catalog = from_gaia_dr3_catalog()
        source = _synthetic_catalog()
        brightest = np.argmin(source["phot_g_mean_mag"])
        faintest = np.argmax(source["phot_g_mean_mag"])
        assert catalog.weight[brightest, 0] > catalog.weight[faintest, 0]

    def test_is_queryable_end_to_end(self, offline_gaia, observation, instrument):
        catalog = from_gaia_dr3_catalog()
        catalog.build_balltree()
        field, refs = catalog.query_direct(
            observation, instrument.pixel_coords(observation), instrument.pixel_radii()
        )
        assert len(refs.indices) == instrument.n_pixels
        assert field.radiance_field is False

    def test_can_be_binned_into_a_map(self, offline_gaia):
        assert isinstance(from_gaia_dr3_catalog().to_map(nside=8), HEALPixSource)


class TestFromGaiaDr3Map:
    def test_builds_a_healpix_source(self, offline_gaia):
        healpix = from_gaia_dr3_map()
        assert isinstance(healpix, HEALPixSource)
        assert healpix.weight.shape[-1] == NPIX

    def test_name_records_the_magnitude_split(self, offline_gaia):
        assert from_gaia_dr3_map().name == f"GaiaDR3_G>{GAIA_SPLIT_MAGNITUDE}"

    def test_weight_is_a_radiance(self, offline_gaia):
        """Magnitudes are divided by the cell solid angle, giving 1/sr."""
        assert from_gaia_dr3_map().weight.unit.is_equivalent(1 / u.sr)

    def test_radiance_scales_with_cell_solid_angle(self, offline_gaia):
        """Integrating the radiance over the sky recovers the total flux."""
        healpix = from_gaia_dr3_map()
        pixel_area = 4 * np.pi / NPIX * u.sr
        total = (healpix.weight * pixel_area).sum()
        expected = (10 ** (-0.4 * _synthetic_map()[0])).sum()
        assert float(total.to_value(u.dimensionless_unscaled)) == pytest.approx(
            expected
        )

    def test_is_defined_in_icrs(self, offline_gaia):
        assert from_gaia_dr3_map().frame == "icrs"

    def test_is_queryable_end_to_end(self, offline_gaia, observation):
        field = from_gaia_dr3_map().query_scattered(observation, nside=NSIDE)
        assert field.radiance_field is True
        assert np.all(field.coords.alt.rad > 0)


class TestGaiaBandpasses:
    def test_fetches_the_three_gaia_bands(self, monkeypatch):
        requested = []
        monkeypatch.setattr(
            stars.Bandpass,
            "from_SVO",
            classmethod(lambda cls, name, *a, **k: requested.append(name)),
        )
        stars._gaia_bandpasses()
        assert requested == ["GAIA/GAIA3.G", "GAIA/GAIA3.Gbp", "GAIA/GAIA3.Grp"]


class TestFromGaiaSupplCatalog:
    def test_builds_from_the_bundled_xhip_table(self, monkeypatch):
        """The supplementary catalogue ships with nsb2; only SVO is remote."""
        monkeypatch.setattr(
            stars.Bandpass,
            "from_SVO",
            classmethod(lambda cls, *a, **k: calibrated_bandpass()),
        )
        monkeypatch.setattr(
            stars,
            "PicklesTRDSAtlas1998",
            lambda: SpectralGrid(
                [],
                np.linspace(3000, 11000, 60) * u.angstrom,
                np.ones((60, 4)) * u.erg / u.angstrom / u.s / u.cm**2,
            ),
        )
        catalog = stars.from_gaia_suppl_catalog()
        assert isinstance(catalog, CatalogSource)
        assert catalog.name == "XHIP_Gaia_Suppl"
        assert len(catalog.coords) > 0
        assert isinstance(catalog.coords, SkyCoord)
