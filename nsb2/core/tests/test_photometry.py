import astropy.units as u
import numpy as np
import pytest

from nsb2.conftest import calibrated_bandpass
from nsb2.core.photometry import (
    PicklesTRDSAtlas1998,
    SolarSpectrumRieke2008,
    create_color_grid,
    synthetic_magnitude,
)
from nsb2.core.spectral import SpectralGrid


def _library(n_wvl=40, n_templates=2):
    """A flat stand-in for the Pickles template atlas."""
    return SpectralGrid(
        [],
        np.linspace(3000, 10000, n_wvl) * u.angstrom,
        np.ones((n_wvl, n_templates)) * u.erg / u.angstrom / u.s / u.cm**2,
    )


@pytest.fixture
def bands():
    """Blue/red passband."""
    return calibrated_bandpass(400, 500), calibrated_bandpass(600, 700)


class TestSyntheticMagnitude:
    def test_is_zero_for_the_reference_spectrum(self):
        bandpass = calibrated_bandpass(400, 600)
        wvl = np.linspace(400, 600, 100) * u.nm
        flx = np.ones(100) * u.erg / u.s / u.cm**2 / u.nm

        assert synthetic_magnitude(wvl, flx, bandpass) == pytest.approx(0.0, abs=1e-6)
        assert synthetic_magnitude(wvl, 0.01 * flx, bandpass) == pytest.approx(
            5.0, abs=1e-6
        )
        # Leading axes are reduced, leaving one magnitude per spectrum.
        stacked = synthetic_magnitude(wvl, np.ones((3, 100)) * flx.unit, bandpass)
        assert stacked.shape == (3,)

        # The wavelength unit must not matter.
        in_angstrom = synthetic_magnitude(
            np.linspace(4000, 6000, 100) * u.angstrom,
            flx.to(u.erg / u.s / u.cm**2 / u.angstrom),
            bandpass,
        )
        assert in_angstrom == pytest.approx(synthetic_magnitude(wvl, flx, bandpass))


class TestPicklesTRDSAtlas1998:
    def test_loads_the_bundled_template_atlas(self):
        atlas = PicklesTRDSAtlas1998()
        assert isinstance(atlas, SpectralGrid)
        assert atlas.points == []
        assert atlas.flx.shape[-1] > 100
        assert np.all(np.diff(atlas.wvl.value) > 0)
        assert atlas.wvl.min() < 400 * u.nm
        assert atlas.wvl.max() > 900 * u.nm
        assert np.all(np.isfinite(atlas.flx.value))
        assert np.mean(atlas.flx.value >= 0) > 0.99


class TestCreateColorGrid:
    def test_builds_a_grid_indexed_by_colour(self, bands):
        blue, red = bands
        for ebvs in (np.linspace(0, 2, 5), None):
            grid = create_color_grid(
                blue, [blue, red], [-1.0, 1.0], _library(60, 4), EBVs=ebvs
            )
            assert isinstance(grid, SpectralGrid)
            assert len(grid.points) == 1
            assert grid.flx.shape[0] == len(grid.points[0])

    def test_accepts_an_alternative_extinction_model(self, bands):
        """The extmod argument is the documented injection point."""
        blue, red = bands

        class GreyExtinction:
            def extinguish(self, wvl, Ebv):
                return 10 ** (-0.4 * np.asarray(Ebv)) * np.ones(len(wvl))

        grid = create_color_grid(
            blue,
            [blue, red],
            [-1.0, 1.0],
            _library(),
            EBVs=np.linspace(0, 2, 5),
            extmod=GreyExtinction(),
        )
        assert isinstance(grid, SpectralGrid)


@pytest.mark.remote_data
class TestSolarSpectrumRieke2008:
    def test_returns_a_solar_spectrum_peaking_in_the_visible(self):
        wvl, flx = SolarSpectrumRieke2008()
        assert wvl.unit.is_equivalent(u.nm)
        assert flx.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.nm)
        assert wvl.shape == flx.shape

        optical = (wvl > 300 * u.nm) & (wvl < 1000 * u.nm)
        peak = wvl[optical][np.argmax(flx[optical])]
        assert 400 * u.nm < peak < 600 * u.nm
