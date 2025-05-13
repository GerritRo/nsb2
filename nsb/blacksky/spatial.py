import astropy.units as u
from astropy.coordinates import SkyCoord
from sklearn.neighbors import BallTree

def method_from_string(string):
    if string == 'balltree':
        return CatalogBallTree

class CatalogBallTree():
    def __init__(self, coords):
        self.coords = coords
        self.build_balltree()

    def skycoord2bcoord(skycoords):
        return np.vstack([skycoords.dec.rad, skycoords.ra.rad]).T
    
    def build_balltree(self):
        self.balltree = BallTree(skycoord2bcoord(self.coords), metric='haversine')

    def query(self, coords, radii):
        inds = self.balltree.query_radius(skycoord2bcoord(coords), radii)
        return np.unique(np.concatenate(inds))

    def apply_space_motion(self, time):
        self.coords = self.coords.apply_space_motion(new_obstime = time)
        self.build_balltree()

    def __add__(self, spatial):
        return CatalogBallTree(self.coords.concatenate(spatial.coords))

    def __getitem__(self, item):
        return CatalogBallTree(self.coords[item])

