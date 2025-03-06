import numpy as np
from astropy.coordinates import SkyCoord


class Ray:
    def __init__(
        self, coords, weight=None, pixels=None, parent=None, source=None, direction=None
    ):
        """
        Effectively a wrapper around astropy SkyCoord, adding some additional logic,
        especially regarding weights and inheriting a parentage.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            Astropy Coordinates in AltAz
        weight : numpy.array, optional
            Weight of each ray, if None all are weighted equally, by default None
        pixels : np.array, optional
            Array of integers representing the camera pixel the ray is associated to. If None, they are assigned as -1, by default None
        parent : np.array, optional
            Array of integers representing the parentage of each ray as an indice. If None, they are assigned as -1, by default None
        source : nsb2.emitter, optional
            The emitting source associated to this ray, by default None
        direction : str, optional
            Marking if the ray goes in a forward or backward direction, by default None
        """
        self.coords = coords
        self.weight = weight
        self.pixels = pixels
        self.parent = parent
        self.source = source
        self.direction = direction

        if not isinstance(self.weight, np.ndarray):
            self.weight = np.ones(self.coords.shape)
        if not isinstance(self.pixels, np.ndarray):
            self.pixels = -1 * np.ones(self.coords.shape)
        if not isinstance(self.parent, np.ndarray):
            self.parent = -1 * np.ones(self.coords.shape)
        if not isinstance(self.source, np.ndarray):
            self.source = np.repeat([self.source], self.coords.shape)

    def separation(self, ray):
        """
        Calculates the separation between this ray and another one

        Parameters
        ----------
        ray : Ray
            The other ray object for which to calculate the separation to.

        Returns
        -------
        astropy.coordinates.Angle
            Separation Angle between rays.
        """
        return self.coords.separation(ray.coords)

    def position_angle(self, ray):
        """
        Calculates the position angle to another ray

        Parameters
        ----------
        ray : Ray
            The other ray object for which to calculate the position angle

        Returns
        -------
        astropy.coordinates.Angle
            Position angle to ray
        """
        return self.coords.position_angle(ray.coords)

    def directional_offset_by(self, pos, rho):
        """
        Offset each element of the ray by rho at a position angle pos

        Parameters
        ----------
        pos : astropy.coordinates.Angle
            Position angle for directional offset
        rho : astropy.coordinates.Angle
            Offset angle for directional offset

        Returns
        -------
        Ray
            Each element of the ray with a directional offset according to pos, rho
        """
        N = pos.shape[-1]
        new_coords = (
            self.coords[..., np.newaxis].directional_offset_by(pos, rho).flatten()
        )
        parents = np.arange(self.N)
        return Ray(
            new_coords,
            self.weight.repeat(N, axis=0) / N,
            self.pixels.repeat(N),
            parents.repeat(N),
            self.source.repeat(N),
            direction=self.direction,
        )

    def transform_to(self, frame):
        """
        Transform ray to given astropy BaseCoordinateFrame

        Parameters
        ----------
        frame : astropy.coordinates.BaseCoordinateFrame
            The frame into which to transform the ray

        Returns
        -------
        Ray
            The transformed Ray
        """
        return Ray(
            self.coords.transform_to(frame),
            self.weight,
            self.pixels,
            self.parent,
            self.source,
            self.direction,
        )

    def repeat(self, repeats):
        """
        Repeats each component of the frame according to the repeats array

        Parameters
        ----------
        repeats : int or array of ints
            Number of repetitions for each element of the ray

        Returns
        -------
        Ray
            Ray with elements repeated
        """
        alt = np.repeat(self.coords.alt, repeats)
        az = np.repeat(self.coords.az, repeats)
        new_coord = SkyCoord(az, alt, frame=self.coords.frame)
        return Ray(
            new_coord,
            np.repeat(self.weight, repeats, axis=0),
            np.repeat(self.pixels, repeats),
            np.repeat(self.parent, repeats),
            np.repeat(self.source, repeats),
            self.direction,
        )

    @property
    def N(self):
        """
        Returns amount of elements in ray

        Returns
        -------
        int
            Amount of elements in ray
        """
        return self.coords.shape[0]

    def __getitem__(self, item):
        """
        Get items from ray via classical numpy indexing

        Parameters
        ----------
        item : int or array-like of ints
            Mask to get array

        Returns
        -------
        Ray
            Ray[item] returns
        """
        return Ray(
            self.coords[item],
            self.weight[item],
            self.pixels[item],
            self.parent[item],
            self.source[item],
            direction=self.direction,
        )

    def __add__(self, ray2):
        """
        Append a new Ray to this ray

        Parameters
        ----------
        ray2 : Ray
            Rays to append to this ray

        Returns
        -------
        Ray
            Combined Ray
        """
        alt_new = np.append(self.coords.alt.deg, ray2.coords.alt.deg)
        az_new = np.append(self.coords.az.deg, ray2.coords.az.deg)
        sc_new = SkyCoord(az_new, alt_new, unit="deg", frame=self.coords.frame)

        # Combine into one ray:
        return Ray(
            sc_new,
            np.append(self.weight, ray2.weight, axis=0),
            np.append(self.pixels, ray2.pixels),
            np.append(self.parent, ray2.parent),
            np.append(self.source, ray2.source),
            self.direction,
        )

    def __mul__(self, value):
        """
        Multiplying ray weights

        Parameters
        ----------
        value : float or array of floats of length self.N
            Value to multiply array

        Returns
        -------
        Ray
            Ray multiplied with value
        """
        return Ray(
            self.coords,
            self.weight * value,
            self.pixels,
            self.parent,
            self.source,
            self.direction,
        )

    def __rmul__(self, value):
        return self.__mul__(value)
