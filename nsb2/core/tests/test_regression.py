"""Numerical drift detectors for the offline computation chain.

These compare against reference arrays generated from a known-good revision
and stored under ``resources/``.  They pin *current* behaviour rather than
validating it against a published table, so a failure means the numbers
moved -- which may be a fix or a regression.  Investigate before regenerating.

To regenerate after an intentional change::

    python -m nsb2.core.tests.test_regression

Anything whose value depends on an astropy coordinate transform is compared
loosely, because the IERS tables update over time.
"""

import astropy.units as u
import numpy as np
import pytest
from scipy.integrate import simpson

from nsb2.atmosphere import SingleScatteringAtmosphere
from nsb2.conftest import (
    make_bandpass,
    make_observation,
    make_response,
    make_spectral_grid,
)
from nsb2.core.instrument import EffectiveApertureInstrument
from nsb2.core.lightpath import DirectPath, ScatteredPath
from nsb2.core.pipeline import Pipeline
from nsb2.core.solver import LUTDirectSolver
from nsb2.core.sources import LonLatSource
from nsb2.core.spectral import SpectralGrid
from nsb2.emitter.airglow import van_rhijn
from nsb2.emitter.moon import load_rolo_table, rolo_albedo
from nsb2.emitter.zodiacal import color_correction
from nsb2.instrument.CTAO import LST1North, MSTNorth
from nsb2.instrument.HESS import CT1

REFERENCE_FILE = "offline_reference.npz"

#: Pure-numpy results are deterministic to the last bit.
EXACT = dict(rtol=1e-10, atol=0)

#: Results that pass through scipy quadrature or spline fitting, or through
#: an astropy frame transform, move in their last digits with the versions of
#: those libraries and with the IERS tables.  Measured drift between scipy
#: 1.11 and 1.18 is a few times 1e-6, so this leaves an order of magnitude.
TOLERANT = dict(rtol=1e-5, atol=0)

#: Quantities computed with numpy alone, and so reproducible exactly.
#: Anything not listed here is compared with :data:`TOLERANT`, which is the
#: safe default for a new entry.
EXACTLY_REPRODUCIBLE = frozenset(
    {"extinction", "scattering", "van_rhijn", "color_correction"}
)


def _atmosphere():
    return SingleScatteringAtmosphere(
        airmass_func=lambda z: 1 / np.cos(np.clip(z, 0, np.deg2rad(85))),
        tau_rayleigh=lambda wvl: 0.1 * (400 * u.nm / wvl) ** 4,
        tau_mie=lambda wvl: 0.05 * np.ones_like(wvl.value),
        tau_absorption=lambda wvl: 0.01 * np.ones_like(wvl.value),
        g=0.65,
    )


def _diffuse_source():
    wvl = np.linspace(300, 700, 20) * u.nm
    flx = np.ones((20, 1)) * 1e-12 * u.erg / u.s / u.cm**2 / u.nm
    return LonLatSource(
        None,
        lambda lon, lat: np.ones(len(lon)) * u.dimensionless_unscaled,
        lambda lon, lat: np.empty((len(lon), 0)),
        SpectralGrid([], wvl, flx),
    )


def compute_reference() -> dict[str, np.ndarray]:
    """Compute every pinned quantity.

    Kept in one place so the test and the regeneration entry point cannot
    drift apart.

    Returns
    -------
    dict of str to numpy.ndarray
        Quantity name to value, with units stripped to their canonical form.
    """
    atmosphere = _atmosphere()
    wvl = np.linspace(300, 700, 9) * u.nm
    alt = np.linspace(0.1, np.pi / 2, 5)

    values = {
        "extinction": np.asarray(atmosphere.extinction(alt, np.zeros(5), wvl)),
        "scattering": np.asarray(
            atmosphere.scattering(
                alt[:, None, None],
                np.zeros(5)[:, None, None],
                alt[None, :, None],
                np.full(5, 0.3)[None, :, None],
                wvl,
            ).to_value(1 / u.sr)
        ),
        "van_rhijn": van_rhijn(90, np.linspace(0, np.deg2rad(85), 12)),
        "color_correction": np.asarray(
            color_correction(np.linspace(300, 900, 13) * u.nm)
        ),
        "rolo_albedo": rolo_albedo(
            load_rolo_table(), np.linspace(400, 1000, 13) * u.nm, 0.4, 0.0
        ),
    }

    # Bundled instrument summaries: the response files are large, so pin
    # reductions of them rather than the arrays themselves.
    for name, factory in (("lst", LST1North), ("mst", MSTNorth), ("ct1", CT1)):
        instrument = factory()
        values[f"{name}_pix_area_sr"] = np.array(
            [
                instrument.n_pixels,
                instrument._pix_area_sr.sum(),
                instrument._pix_area_sr.mean(),
                instrument._pix_area_sr.std(),
            ]
        )
        values[f"{name}_bandpass"] = np.array(
            [
                instrument.bandpass.min.to_value(u.nm),
                instrument.bandpass.max.to_value(u.nm),
                simpson(
                    instrument.bandpass(instrument.bandpass.lam),
                    x=instrument.bandpass.lam.to_value(u.nm),
                ),
            ]
        )

    # End-to-end pipeline rates, both solvers, for the synthetic instrument.
    instrument = EffectiveApertureInstrument(make_response(), make_bandpass())
    observation = make_observation()
    source = _diffuse_source()

    explicit = Pipeline(instrument, atmosphere, source, [DirectPath()])
    values["pipeline_direct"] = explicit.predict(observation)[0].rates.value

    scattered = Pipeline(instrument, atmosphere, source, [ScatteredPath(nside=8)])
    values["pipeline_scattered"] = scattered.predict(observation)[0].rates.value

    lut = Pipeline(instrument, atmosphere, source, [DirectPath(LUTDirectSolver())])
    lut.compile(extinction_z_bins=30)
    values["pipeline_direct_lut"] = lut.predict(observation)[0].rates.value

    # Spectral machinery, independent of any coordinate frame.
    grid = make_spectral_grid(n_wvl=15, n_comp=2)
    values["spectral_integrate"] = grid.integrate().rate.value

    return values


@pytest.fixture(scope="module")
def reference():
    from importlib.resources import files

    path = files("nsb2.core.tests").joinpath("resources", REFERENCE_FILE)
    with path.open("rb") as stream:
        with np.load(stream) as archive:
            return {key: archive[key] for key in archive.files}


@pytest.fixture(scope="module")
def computed():
    return compute_reference()


def test_reference_file_covers_every_quantity(reference, computed):
    """A new pinned quantity must be added to the stored reference."""
    assert set(reference) == set(computed)


@pytest.mark.parametrize("name", sorted(compute_reference()))
def test_matches_reference(name, reference, computed):
    tolerance = EXACT if name in EXACTLY_REPRODUCIBLE else TOLERANT
    np.testing.assert_allclose(
        computed[name],
        reference[name],
        err_msg=(
            f"{name} drifted from its reference. If the change is intended, "
            f"regenerate with: python -m nsb2.core.tests.test_regression"
        ),
        **tolerance,
    )


def _regenerate() -> None:
    """Write the reference archive from the current implementation."""
    from pathlib import Path

    target = Path(__file__).parent / "resources" / REFERENCE_FILE
    target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target, **compute_reference())
    print(f"wrote {target}")


if __name__ == "__main__":
    _regenerate()
