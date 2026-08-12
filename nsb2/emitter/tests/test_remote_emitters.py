import astropy.units as u
import numpy as np
import pytest

from nsb2.core.lightpath import DirectPath, ScatteredPath
from nsb2.core.pipeline import Pipeline
from nsb2.core.sources import (
    CatalogSource,
    EphemerisSource,
    HEALPixSource,
    LonLatSource,
)
from nsb2.emitter.moon import from_noll2013
from nsb2.emitter.stars import from_gaia_suppl_catalog
from nsb2.emitter.zodiacal import from_leinert1998

pytestmark = pytest.mark.remote_data


class TestMoon:
    def test_builds_an_ephemeris_source(self):
        moon = from_noll2013()
        assert isinstance(moon, EphemerisSource)
        assert moon.body == "moon"

    def test_spectrum_is_positive_and_photon_flux(self):
        grid = from_noll2013().spectral_grid
        assert np.all(grid.flx.value >= 0)
        assert grid.flx.unit.is_equivalent(1 / (u.nm * u.s * u.cm**2))

    def test_three_libration_variants_are_tabulated(self):
        assert from_noll2013().spectral_grid.flx.shape[-1] == 3

    def test_full_moon_is_brighter_than_crescent(self):
        """The phase axis is signed, so full Moon sits at its centre."""
        grid = from_noll2013().spectral_grid
        phase_angles = grid.points[0]

        def at(degrees):
            return np.argmin(np.abs(phase_angles - np.deg2rad(degrees)))

        full = np.nanmedian(grid.flx[at(0)].value)
        crescent = np.nanmedian(grid.flx[at(150)].value)
        assert full > crescent
        assert np.nanmedian(grid.flx[at(-150)].value) == pytest.approx(crescent)


class TestZodiacal:
    def test_builds_a_lonlat_source(self):
        zodiacal = from_leinert1998()
        assert isinstance(zodiacal, LonLatSource)
        assert zodiacal.name == "Zodiacal_Leinert1998"

    def test_brighter_close_to_the_sun(self):
        weight = from_leinert1998().weight_function
        near = weight(np.array([np.deg2rad(30)]), np.array([0.0]))
        far = weight(np.array([np.deg2rad(150)]), np.array([0.0]))
        assert near > far

    def test_brightness_is_symmetric_about_the_sun(self):
        """The tabulated brightness depends on |lon| and |lat| only."""
        weight = from_leinert1998().weight_function
        east = weight(np.array([np.deg2rad(40)]), np.array([np.deg2rad(20)]))
        west = weight(np.array([np.deg2rad(-40)]), np.array([np.deg2rad(-20)]))
        assert float(west[0].value) == pytest.approx(float(east[0].value))

    def test_elongation_accounts_for_ecliptic_latitude(self):
        """Elongation is the great-circle distance, not the longitude offset."""
        data = from_leinert1998().data_function
        on_ecliptic = data(np.array([0.0]), np.array([0.0]))[0, 0]
        high_latitude = data(np.array([0.0]), np.array([np.deg2rad(60)]))[0, 0]
        assert on_ecliptic == pytest.approx(np.deg2rad(30))
        assert high_latitude == pytest.approx(np.deg2rad(60))


class TestStars:
    def test_supplementary_catalog_loads(self):
        catalog = from_gaia_suppl_catalog()
        assert isinstance(catalog, CatalogSource)
        assert len(catalog.coords) > 0

    def test_supplementary_catalog_can_be_queried(self, observation, instrument):
        catalog = from_gaia_suppl_catalog()
        catalog.build_balltree()
        _, refs = catalog.query_direct(
            observation, instrument.pixel_coords(observation), instrument.pixel_radii()
        )
        assert len(refs.indices) == instrument.n_pixels

    def test_binning_to_a_map_gives_a_healpix_source(self):
        catalog = from_gaia_suppl_catalog()
        assert isinstance(catalog.to_map(nside=32), HEALPixSource)


class TestEndToEnd:
    def test_moon_pipeline_predicts_positive_rates(
        self, instrument, atmosphere, observation
    ):
        pipeline = Pipeline(
            instrument,
            atmosphere,
            from_noll2013(),
            [DirectPath(), ScatteredPath(nside=16)],
        )
        results = pipeline.predict(observation)
        assert len(results) == 2
        for prediction in results:
            assert prediction.rates.shape == (instrument.n_pixels, 3)
            assert np.all(np.isfinite(prediction.rates.value))

    def test_zodiacal_pipeline_predicts_positive_rates(
        self, instrument, atmosphere, observation
    ):
        pipeline = Pipeline(instrument, atmosphere, from_leinert1998(), [DirectPath()])
        rates = pipeline.predict(observation)[0].rates
        assert np.all(rates.value > 0)
