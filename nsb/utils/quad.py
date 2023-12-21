import numpy as np
import matplotlib.pyplot as plt
import quadpy

class QuadScheme():
    def __init__(self, order, cam):
        self.order = order
        self.width = cam.pixel_width
        self.points, self.weights = self.get_points(cam.pix_type.value)
        
    def get_points(self, pixel_shape):
        if pixel_shape == 'square':
            scheme = quadpy.c2.get_good_scheme(self.order)
            return scheme.points, scheme.weights
        elif pixel_shape == 'circle':
            scheme = quadpy.s2.get_good_scheme(self.order)
            return scheme.points, scheme.weights
        elif pixel_shape == 'hexagon':
            # Hacky solution using 6 triangles:
            scheme = quadpy.t2.get_good_scheme(np.ceil(self.order/3))
            tri_points = scheme.points
            # Convert from trilateral to cartesian:
            x = 1/np.sqrt(3) * (tri_points[0] - tri_points[1])
            y = tri_points[0] + tri_points[1]
            # Rotate 6 times to get hexagon (off angle determines hexago n rotation):
            ang_off = np.pi/6
            x_p, y_p = np.asarray([]), np.asarray([])
            for i in range(6):
                x_p = np.append(x_p, x*np.cos(i*np.pi/3+ang_off) - y*np.sin(i*np.pi/3+ang_off))
                y_p = np.append(y_p, y*np.cos(i*np.pi/3+ang_off) + x*np.sin(i*np.pi/3+ang_off))
            
            return np.vstack([x_p, y_p]), np.resize(scheme.weights/6, len(scheme.weights)*6)
    
    def oversample(self, x, y):
        return np.asarray([x, y]).T[:,:,np.newaxis] + self.width.value[:, np.newaxis, np.newaxis]*self.points/2