import astropy.units as u
import numpy as np

from nsb2.atmosphere import SingleScatteringAtmosphere
from nsb2.core.dtypes import Prediction
from nsb2.core.instrument import EffectiveApertureInstrument
from nsb2.core.lightpath import DirectPath
from nsb2.core.pipeline import CompositePipeline, Pipeline
from nsb2.core.solver import LUTDirectSolver
from nsb2.core.sources import LonLatSource
from nsb2.core.spectral import SpectralGrid
from tests.conftest import make_bandpass, make_observation


def _make_atmosphere():
    def X(z):
        return 1 / np.cos(np.clip(z, 0, np.deg2rad(85)))
    def tau_r(wvl):
        return 0.1 * (400 * u.nm / wvl) ** 4
    def tau_a(wvl):
        return 0.01 * np.ones_like(wvl.value)
    return SingleScatteringAtmosphere(
        airmass_func=X,
        tau_rayleigh=tau_r,
        tau_mie=lambda wvl: 0.05 * np.ones_like(wvl.value),
        tau_absorption=tau_a,
        g=0.65,
    )


def _make_instrument(n_pix=4, grid_size=5):
    x_arr, y_arr, v_arr = [], [], []
    for i in range(n_pix):
        cx = np.deg2rad(i * 0.5 - 0.75)
        cy = 0.0
        x = np.linspace(cx - 0.005, cx + 0.005, grid_size)
        y = np.linspace(cy - 0.005, cy + 0.005, grid_size)
        v = np.ones((grid_size, grid_size)) * 10.0
        x_arr.append(x)
        y_arr.append(y)
        v_arr.append(v)
    resp = {'x': np.array(x_arr), 'y': np.array(y_arr), 'values': np.array(v_arr)}
    bp = make_bandpass()
    return EffectiveApertureInstrument(resp, bp)


def _make_diffuse_source():
    """Create a trivial uniform diffuse source."""
    wvl = np.linspace(300, 700, 20) * u.nm
    flx = np.ones((20, 1)) * 1e-12 * u.erg / u.s / u.cm**2 / u.nm
    sg = SpectralGrid([], wvl, flx)

    def weight_fn(lon, lat):
        return np.ones(len(lon)) * u.dimensionless_unscaled

    def data_fn(lon, lat):
        return np.empty((len(lon), 0))

    return LonLatSource(None, weight_fn, data_fn, sg)


class TestPrediction:
    def test_fields(self):
        p = Prediction(rates=np.zeros((4, 3)), indirect=False)
        assert p.indirect is False
        assert p.rates.shape == (4, 3)


class TestPipeline:
    def test_init_wraps_single_source_in_list(self):
        src = _make_diffuse_source()
        pipe = Pipeline(_make_instrument(), _make_atmosphere(), src,
                        [DirectPath()])
        assert isinstance(pipe.sources, list)
        assert len(pipe.sources) == 1

    def test_compile_is_noop_for_explicit_solver(self):
        src = _make_diffuse_source()
        pipe = Pipeline(_make_instrument(), _make_atmosphere(), src,
                        [DirectPath()])
        assert pipe.compile() == 0

    def test_predict_direct_path(self):
        """Pipeline with DirectPath should produce direct predictions."""
        src = _make_diffuse_source()
        pipe = Pipeline(_make_instrument(), _make_atmosphere(), src,
                        [DirectPath()])
        obs = make_observation()
        results = pipe.predict(obs)
        assert len(results) == 1
        assert isinstance(results[0], Prediction)
        assert results[0].indirect is False

    def test_add_creates_composite(self):
        src1 = _make_diffuse_source()
        src2 = _make_diffuse_source()
        atm = _make_atmosphere()
        inst = _make_instrument()
        p1 = Pipeline(inst, atm, src1, [DirectPath()])
        p2 = Pipeline(inst, atm, src2, [DirectPath()])
        combined = p1 + p2
        assert isinstance(combined, CompositePipeline)


class TestCompositePipeline:
    def test_predict_combines_results(self):
        src1 = _make_diffuse_source()
        src2 = _make_diffuse_source()
        atm = _make_atmosphere()
        inst = _make_instrument()
        p1 = Pipeline(inst, atm, src1, [DirectPath()])
        p2 = Pipeline(inst, atm, src2, [DirectPath()])
        combined = p1 + p2
        obs = make_observation()
        results = combined.predict(obs)
        assert len(results) == 2


class TestLUTPipeline:
    def test_compile_and_predict_via_lut(self):
        src = _make_diffuse_source()
        lut_solver = LUTDirectSolver()
        pipe = Pipeline(_make_instrument(), _make_atmosphere(), src,
                        [DirectPath(solver=lut_solver)])
        pipe.compile(extinction_z_bins=10)
        obs = make_observation()
        results = pipe.predict(obs)
        assert len(results) == 1
        assert results[0].indirect is False
