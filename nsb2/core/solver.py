"""Strategies for turning source spectra into band-integrated rates.

Two families of solver implement the same physics with different trade-offs.
The *explicit* solvers integrate the full spectrum of every source, which is
exact but scales with the number of sources.  The *LUT* solvers pre-integrate
the spectra onto a lookup table during a compilation step, exploiting the
azimuthal symmetry of the atmosphere, and then only interpolate at predict
time.  For catalogues with many sources the lookup tables are orders of
magnitude faster at a small cost in accuracy.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, cast

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from nsb2.core.interpolation import UnitRegularGridInterpolator
from nsb2.core.spectral import integrate_wavelength

__all__ = [
    "DirectSolver",
    "ExplicitDirectSolver",
    "ExplicitScatteredSolver",
    "LUTDirectSolver",
    "LUTScatteredSolver",
    "ScatteredSolver",
    "Solver",
]


if TYPE_CHECKING:
    from nsb2.core.atmosphere import Atmosphere
    from nsb2.core.dtypes import SourceField
    from nsb2.core.instrument import Instrument
    from nsb2.core.sources import Source
    from nsb2.core.spectral import Bandpass

logger = logging.getLogger(__name__)


def _trapz_einsum(a, b, wvl, eins_str):
    """Contract two arrays and integrate over wavelength in one pass.

    Equivalent to an :func:`numpy.einsum` contraction of ``a`` and ``b``
    followed by trapezoidal integration over wavelength, but without
    materialising the full outer product, which would not fit in memory for
    realistic lookup table sizes.

    Parameters
    ----------
    a, b : array_like
        Operands, with the wavelength axis as indicated by ``eins_str``.
    wvl : astropy.units.Quantity
        Wavelength grid.
    eins_str : str
        Subscript string for :func:`numpy.einsum`, taking ``a``, ``b`` and
        the wavelength spacings as its three operands.

    Returns
    -------
    array_like
        The contracted and integrated result.
    """
    delta_x = np.diff(wvl) / 2
    return np.einsum(eins_str, a[..., :-1], b[..., :-1, :], delta_x) + np.einsum(
        eins_str, a[..., 1:], b[..., 1:, :], delta_x
    )


class Solver:
    """Base class for rate computation strategies.

    Subclasses implement ``compute_rates``; see :class:`DirectSolver` and
    :class:`ScatteredSolver` for the two signatures in use.
    """

    def compile(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        **kwargs,
    ) -> float:
        """Pre-compute whatever this solver needs before predicting.

        The default implementation does nothing, which is correct for solvers
        that integrate explicitly at predict time.

        Parameters
        ----------
        source : nsb2.core.sources.Source
            Source the pre-computation applies to.
        instrument : nsb2.core.instrument.Instrument
            Instrument whose passband the spectra are restricted to.
        atmosphere : nsb2.core.atmosphere.Atmosphere
            Atmosphere model to tabulate.
        **kwargs
            Solver-specific compilation options.

        Returns
        -------
        float
            Compilation cost metric; zero when nothing was compiled.
        """
        return 0


class DirectSolver(Solver):
    """Base class for solvers of the direct, extinction-only light path."""

    @abstractmethod
    def compute_rates(
        self,
        source: Source,
        field: SourceField,
        atmosphere: Atmosphere,
        bandpass: Bandpass,
    ) -> u.Quantity:
        """Compute per-source rates along the direct path.

        Parameters
        ----------
        source : nsb2.core.sources.Source
            The source the field was queried from.
        field : nsb2.core.dtypes.SourceField
            Sources to compute rates for.
        atmosphere : nsb2.core.atmosphere.Atmosphere
            Atmosphere providing the extinction.
        bandpass : nsb2.core.spectral.Bandpass
            Instrument passband.

        Returns
        -------
        astropy.units.Quantity
            Rates, shape ``(N_sources, C_comp)``, with ``field.weights``
            already applied.
        """
        ...


class ScatteredSolver(Solver):
    """Base class for solvers of the scattered, indirect light path."""

    @abstractmethod
    def compute_rates(
        self,
        source: Source,
        field: SourceField,
        atmosphere: Atmosphere,
        bandpass: Bandpass,
        eval_coords: SkyCoord,
    ) -> u.Quantity:
        """Compute in-scattered rates on an evaluation grid.

        Parameters
        ----------
        source : nsb2.core.sources.Source
            The source the field was queried from.
        field : nsb2.core.dtypes.SourceField
            Sources illuminating the atmosphere.
        atmosphere : nsb2.core.atmosphere.Atmosphere
            Atmosphere providing the scattering kernel.
        bandpass : nsb2.core.spectral.Bandpass
            Instrument passband.
        eval_coords : astropy.coordinates.SkyCoord
            Grid of directions to evaluate at, shape ``(M_lat, M_lon)``.

        Returns
        -------
        astropy.units.Quantity
            Rates, shape ``(M_lat, M_lon, N_sources, C_comp)``, with
            ``field.weights`` already applied.
        """
        ...


class ExplicitDirectSolver(DirectSolver):
    """Direct extinction evaluated by full spectral integration.

    Exact, and the right choice for a handful of sources or when maximum
    accuracy is wanted.  For large catalogues prefer :class:`LUTDirectSolver`.
    """

    def compute_rates(
        self,
        source: Source,
        field: SourceField,
        atmosphere: Atmosphere,
        bandpass: Bandpass,
    ) -> u.Quantity:
        """Compute per-source rates by integrating each extinguished spectrum.

        See :meth:`DirectSolver.compute_rates` for the parameters.
        """
        resolved = field.resolve_spectra(bandpass)
        coords = cast(SkyCoord, np.atleast_1d(resolved.coords))
        ext_weights = atmosphere.extinction(coords.alt.rad, coords.az.rad, resolved.wvl)
        return resolved.integrate(extra_weights=ext_weights[..., None])


class ExplicitScatteredSolver(ScatteredSolver):
    """Scattered light evaluated by full spectral integration.

    Exact, and the right choice for a handful of illuminating sources such as
    the Moon.  For hemispheres discretised into many HEALPix cells prefer
    :class:`LUTScatteredSolver`.
    """

    def compute_rates(
        self,
        source: Source,
        field: SourceField,
        atmosphere: Atmosphere,
        bandpass: Bandpass,
        eval_coords: SkyCoord,
    ) -> u.Quantity:
        """Compute in-scattered rates by integrating each scattered spectrum.

        See :meth:`ScatteredSolver.compute_rates` for the parameters.
        """
        resolved = field.resolve_spectra(bandpass)
        scat_weights = atmosphere.scattering(
            eval_coords.alt.rad[..., None, None],
            eval_coords.az.rad[..., None, None],
            resolved.coords.alt.rad[None, ..., None],
            resolved.coords.az.rad[None, ..., None],
            resolved.wvl,
        )
        integrand = (scat_weights * resolved.weights)[..., None] * resolved.flx
        return integrate_wavelength(integrand, resolved.wvl, axis=-2)


class LUTDirectSolver(DirectSolver):
    """Direct extinction interpolated from a pre-compiled lookup table.

    The table is tabulated over zenith angle and the parameter axes of the
    source's spectral grid, which is exact for atmospheres without azimuthal
    structure.  :meth:`compile` must be called before :meth:`compute_rates`.
    """

    def __init__(self) -> None:
        self._luts: dict[Source, UnitRegularGridInterpolator] = {}

    def compile(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        *,
        extinction_z_bins: int = 90,
        **kwargs,
    ) -> float:
        """Tabulate band-integrated, extinguished rates over zenith angle.

        Parameters
        ----------
        source : nsb2.core.sources.Source
            Source whose spectral grid is tabulated.
        instrument : nsb2.core.instrument.Instrument
            Instrument whose passband the spectra are restricted to.
        atmosphere : nsb2.core.atmosphere.Atmosphere
            Atmosphere providing the extinction.
        extinction_z_bins : int, optional
            Number of zenith angle samples between zenith and the horizon.
            Default is 90, i.e. one per degree.
        **kwargs
            Ignored; accepted so that a single ``compile`` call can carry the
            options of every solver in a pipeline.

        Returns
        -------
        float
            Always zero; the table is stored on the solver.
        """
        spectral_grid = source.spectral_grid.apply_bandpass(instrument.bandpass)
        z_range = np.linspace(0, np.pi / 2, extinction_z_bins)
        logger.debug(
            "compiling direct extinction table for %s with %d zenith bins",
            source.name,
            extinction_z_bins,
        )
        res = _trapz_einsum(
            atmosphere.extinction(z_range, 0, spectral_grid.wvl),
            spectral_grid.flx,
            spectral_grid.wvl,
            "zN,...Nc,N->z...c",
        )
        self._luts[source] = UnitRegularGridInterpolator(
            (z_range, *spectral_grid.points),
            res,
            method="linear",
            bounds_error=False,
        )
        return 0

    def compute_rates(
        self,
        source: Source,
        field: SourceField,
        atmosphere: Atmosphere,
        bandpass: Bandpass,
    ) -> u.Quantity:
        """Interpolate per-source rates from the compiled table.

        See :meth:`DirectSolver.compute_rates` for the parameters.

        Raises
        ------
        RuntimeError
            If :meth:`compile` has not been called for ``source``.
        """
        try:
            lut = self._luts[source]
        except KeyError:
            raise RuntimeError(
                f"No compiled LUT for {type(source).__name__}. "
                f"Call compile() before predict()."
            ) from None
        coords = cast(SkyCoord, np.atleast_1d(field.coords))
        raw_rates = lut(np.column_stack([coords.alt.rad, field.spectral_data]))
        return field.weights * raw_rates


class LUTScatteredSolver(ScatteredSolver):
    """Scattered light interpolated from a pre-compiled lookup table.

    The table is tabulated over the zenith angles of the evaluation point and
    of the source, their relative azimuth, and the parameter axes of the
    source's spectral grid.  :meth:`compile` must be called before
    :meth:`compute_rates`.
    """

    def __init__(self) -> None:
        self._luts: dict[Source, UnitRegularGridInterpolator] = {}

    def compile(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        *,
        scattering_z_bins: int = 10,
        scattering_theta_bins: int = 10,
        **kwargs,
    ) -> float:
        """Tabulate band-integrated, in-scattered rates over the sky geometry.

        Parameters
        ----------
        source : nsb2.core.sources.Source
            Source whose spectral grid is tabulated.
        instrument : nsb2.core.instrument.Instrument
            Instrument whose passband the spectra are restricted to.
        atmosphere : nsb2.core.atmosphere.Atmosphere
            Atmosphere providing the scattering kernel.
        scattering_z_bins : int, optional
            Number of zenith angle samples, used for both the evaluation
            point and the source.  Default is 10.
        scattering_theta_bins : int, optional
            Number of relative azimuth samples.  Default is 10.  The samples
            are spaced quadratically, which concentrates them at small
            separations where the scattering kernel varies fastest.
        **kwargs
            Ignored; accepted so that a single ``compile`` call can carry the
            options of every solver in a pipeline.

        Returns
        -------
        float
            Always zero; the table is stored on the solver.
        """
        spectral_grid = source.spectral_grid.apply_bandpass(instrument.bandpass)
        z_range = np.linspace(0, np.pi / 2, scattering_z_bins)
        theta_range = np.linspace(0, np.sqrt(np.pi), scattering_theta_bins) ** 2
        logger.debug(
            "compiling scattering table for %s with %d x %d x %d bins",
            source.name,
            scattering_z_bins,
            scattering_z_bins,
            scattering_theta_bins,
        )
        res = _trapz_einsum(
            atmosphere.scattering(
                z_range[:, None, None, None],
                0,
                z_range[None, :, None, None],
                theta_range[None, None, :, None],
                spectral_grid.wvl,
            ),
            spectral_grid.flx,
            spectral_grid.wvl,
            "abcN,...Nd,N->abc...d",
        )
        self._luts[source] = UnitRegularGridInterpolator(
            (z_range, z_range, theta_range, *spectral_grid.points),
            res,
            method="linear",
            bounds_error=False,
        )
        return 0

    def compute_rates(
        self,
        source: Source,
        field: SourceField,
        atmosphere: Atmosphere,
        bandpass: Bandpass,
        eval_coords: SkyCoord,
    ) -> u.Quantity:
        """Interpolate in-scattered rates from the compiled table.

        See :meth:`ScatteredSolver.compute_rates` for the parameters.

        Raises
        ------
        RuntimeError
            If :meth:`compile` has not been called for ``source``.
        """
        try:
            lut = self._luts[source]
        except KeyError:
            raise RuntimeError(
                f"No compiled LUT for {type(source).__name__}. "
                f"Call compile() before predict()."
            ) from None
        coord_array = np.broadcast_arrays(
            eval_coords.alt.rad[..., np.newaxis],
            field.coords.alt.rad[np.newaxis, np.newaxis, :],
            np.pi
            - np.abs(
                np.abs(eval_coords.az.rad[..., np.newaxis] - field.coords.az.rad)
                - np.pi
            ),
        )
        lut_input = np.concatenate(
            [
                np.stack(coord_array, axis=-1),
                np.broadcast_to(
                    field.spectral_data,
                    eval_coords.shape + field.spectral_data.shape,
                ),
            ],
            axis=-1,
        )
        raw_rates = lut(lut_input)
        return raw_rates * field.weights
