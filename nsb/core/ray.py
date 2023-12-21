import numpy as np
from astropy.coordinates import SkyCoord

class Ray:
    def __init__(self, coords, weight=None, pixels=None, parent=None, source=None, direction=None):
        """
        'Ray' base class.
        
        Effectively a wrap around astropy SkyCoord, enabling some additional logic,
        especially regarding weights.
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
            self.pixels = -1*np.ones(self.coords.shape)
        if not isinstance(self.parent, np.ndarray):
            self.parent = -1*np.ones(self.coords.shape)
        if not isinstance(self.source, np.ndarray):
            self.source = np.repeat([self.source], self.coords.shape)

    def separation(self, ray):
        return self.coords.separation(ray.coords)

    def position_angle(self, ray):
        return self.coords.position_angle(ray.coords)

    def directional_offset_by(self, pos, rho):
        N = pos.shape[-1]
        new_coords = self.coords[...,np.newaxis].directional_offset_by(pos, rho).flatten()
        parents = np.arange(self.N)
        return Ray(new_coords, 
                   self.weight.repeat(N, axis=0)/N,
                   self.pixels.repeat(N), 
                   parents.repeat(N), 
                   self.source.repeat(N),
                   direction=self.direction)

    def transform_to(self, frame):
        return Ray(self.coords.transform_to(frame),
                   self.weight,
                   self.pixels,
                   self.parent,
                   self.source,
                   self.direction)

    def repeat(self, repeats):
        alt = np.repeat(self.coords.alt, repeats)
        az  = np.repeat(self.coords.az, repeats)
        new_coord = SkyCoord(az, alt, frame=self.coords.frame)
        return Ray(new_coord,
                   np.repeat(self.weight, repeats, axis=0),
                   np.repeat(self.pixels, repeats),
                   np.repeat(self.parent, repeats),
                   np.repeat(self.source, repeats),
                   self.direction)

    @property
    def N(self):
        return self.coords.shape[0]

    def __getitem__(self, item):
        return Ray(self.coords[item], 
                   self.weight[item],
                   self.pixels[item], 
                   self.parent[item], 
                   self.source[item],
                   direction=self.direction)

    def __add__(self, ray2):
        # First we need to create a combined SkyCoord
        alt_new = np.append(self.coords.alt.deg, ray2.coords.alt.deg)
        az_new  = np.append(self.coords.az.deg, ray2.coords.az.deg)
        sc_new = SkyCoord(az_new, alt_new, unit='deg', frame=self.coords.frame)
        
        # Combine into one ray:
        return Ray(sc_new,
                   np.append(self.weight, ray2.weight, axis=0),
                   np.append(self.pixels, ray2.pixels),
                   np.append(self.parent, ray2.parent),
                   np.append(self.source, ray2.source),
                   self.direction)
    
    def __mul__(self, value):
        # Multiplying with an array
        return Ray(self.coords, 
                   self.weight*value, 
                   self.pixels, 
                   self.parent, 
                   self.source,
                   self.direction)
    
    def __rmul__(self, value):
        return self.__mul__(value)