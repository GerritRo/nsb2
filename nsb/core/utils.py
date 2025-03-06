import numpy as np
import functools

import nsb as nsb
from collections import defaultdict


def reduce_rays(func):
    def _decorator(self, frame, rays):
        if rays != None:
            rays = functools.reduce(lambda a, b: a + b, rays)
        return func(self, frame, rays)

    return _decorator


def multi_rays(func):
    def _decorator(self, frame, rays):
        if rays != None:
            return [func(self, frame, r) for r in rays]
        return [func(self, frame, rays)]

    return _decorator


def hist_sample(hist, args, N):
    if type(args) == int:
        pos = np.random.random((args, N)) * 2 * np.pi
        rho = hist.sample((args, N))
        return pos, rho
    else:
        d = np.digitize(args, hist.bins[0]) - 1
        pos, rho = np.ones((len(args), N)), np.ones((len(args), N))
        vals, counts = np.unique(d, return_counts=True)
        for i in range(len(vals)):
            p, r = hist.sample(N * counts[i], hist.centers[0][vals[i]])
            pos[d == vals[i]] = p.reshape((counts[i], N))
            rho[d == vals[i]] = r.reshape((counts[i], N))
        return pos, rho


def sq_solid_angle(A, f):
    """
    Calculates the solid angle of a square on the sky depending on focal ratio

    Parameters
    ----------
    A : float or floatlike
        edge size of square
    f : float or floatlike
        focal ratio of telescope

    Returns
    -------
    float or floatlike
        Solid angle of square projected onto the sky
    """
    return 4 * np.arcsin(A / (A + 4 * f**2))


def haversine(delta_lon, lat1, lat2):
    """
    The haversine function.
    """
    delta_lat = lat1 - lat2
    sin_delta_lat = np.sin(delta_lat / 2) ** 2
    sin_sum_lat = np.sin((lat1 + lat2) / 2) ** 2
    sin_delta_lon = np.sin(delta_lon / 2) ** 2

    return 2 * np.arcsin(
        np.sqrt(sin_delta_lat + (1 - sin_delta_lat - sin_sum_lat) * sin_delta_lon)
    )


def create_physical_graph(layer):
    """
    Creates the physics graph.

    Graph describing the actual path of light rays from source to instrument.

    Parameters
    ----------
    layer : Layer
        Linked layers for which to calculate computational graph
    """

    def add_entries(graph, layer):
        for parent in layer.parents:
            if type(parent) == nsb.core.PhotonMap:
                parent = parent.layer
            graph[layer].append(parent)
            add_entries(graph, parent)

    graph = defaultdict(list)
    add_entries(graph, layer)

    return graph


def create_computational_graph(layer):
    """
    Creates the computational graph.

    Graph describing the computational order to make backwards/forwards rays possible

    Parameters
    ----------
    layer : Layer
        Linked layers for which to calculate computational graph
    """

    def add_entries(graph, layer):
        for parent in layer.parents:
            if parent.mode == "backward" or parent.mode == "bidirectional":
                graph[layer.call_backward].append(parent.call_backward)
            if parent.mode == "forward" or parent.mode == "bidirectional":
                graph[parent.call_forward].append(layer.call_forward)
            add_entries(graph, parent)

    graph = defaultdict(list)
    add_entries(graph, layer)

    return graph


def reverse_graph(graph):
    """
    Reverse order of directional graph

    Parameters
    ----------
    graph : dictionary
        Graph that should be reversed in dictionary representation

    Returns
    -------
    dictionary
        Graph with all directions reversed
    """
    r_graph = defaultdict(list)
    for x, y in graph.items():
        for j in y:
            r_graph[j].append(x)

    return r_graph
