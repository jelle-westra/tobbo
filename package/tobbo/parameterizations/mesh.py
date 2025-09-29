import numpy as np
from scipy import spatial
from shapely.geometry import MultiPolygon, LineString, box
from shapely import unary_union

from dataclasses import dataclass
from itertools import combinations
from typing import List

from tobbo.core import Parameterization, Topology
from .mmc import MMC, MMCCenterpointsConfig, StraightBeam

def get_voronoi_line_segments(mesh: spatial.Voronoi) -> List[np.ndarray]:
    assert ((0 < mesh.points) & (mesh.points < 1)).all(), 'assuming points are normalized; i.e. all points in unit square [0,1]^2'
    domain = box(0,0,1,1)

    # borrowed some lines form `scipy.spatial.voronoi_plot_2d`
    center = mesh.points.mean(axis=0)
    segments = []
    for pointidx, simplex in zip(mesh.ridge_points, mesh.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            line = LineString(mesh.vertices[simplex])
        else:
            v = mesh.vertices[simplex[simplex >= 0][0]] # finite end Voronoi vertex
            # we're only interested in the vertices within the domain 
            if not((0 <= v) & (v <= 1)).all() : continue

            t = mesh.points[pointidx[1]] - mesh.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = mesh.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n

            line = LineString([v, v + np.sqrt(2)*direction])
        # clipping the segment to the domain, only add if remaining geo is line
        if (endpoints := np.c_[line.intersection(domain).xy]).size: segments.append(endpoints)
    return segments

@dataclass
class Voronoi(Parameterization):
    topology: Topology
    symmetry_x: bool
    symmetry_y: bool
    n_points: int
    mmc_cls : MMC

    def __post_init__(self):
        self.mmc = self.mmc_cls(self.topology, False, False, 1, MMCCenterpointsConfig, StraightBeam, 100)
        self._dimension = 2*self.n_points

    @property
    def dimension(self) -> int : return self._dimension

    def compute_geometry(self, x: np.ndarray) -> MultiPolygon :
        x_pts = x.reshape(self.n_points, 2)
        mesh = spatial.Voronoi(x_pts)
        segments = get_voronoi_line_segments(mesh)
        return unary_union([self.mmc.compute_geometry(np.r_[segment.flatten(), 1.]) for segment in segments])

@dataclass
class Delaunay(Parameterization):
    topology: Topology
    symmetry_x: bool
    symmetry_y: bool
    n_points: int
    mmc_cls : MMC

    def __post_init__(self) :
        self.mmc = self.mmc_cls(self.topology, False, False, 1, MMCCenterpointsConfig, StraightBeam, 100)
        self.max_n_edges = 3*self.n_points - 6
        self._dimension = 2*self.n_points + self.max_n_edges

    @property
    def dimension(self) -> int : return self._dimension
    
    def compute_geometry(self, x: np.ndarray) -> MultiPolygon :
        x_pts = x[:2*self.n_points].reshape(-1, 2)
        x_rad = x[2*self.n_points:]
        mesh = spatial.Delaunay(x_pts)
        edges = {(u, v) if (v > u) else (v, u) for tri in mesh.simplices for (u, v) in combinations(tri, r=2)}
        # return unary_union([self.mmc.compute_geometry(np.r_[x_pts[u], x_pts[v], 0.08 if x_rad[i] else 0.0]) for (i, (u, v)) in enumerate(edges)])
        return unary_union([self.mmc.compute_geometry(np.r_[x_pts[u], x_pts[v], x_rad[i]]) for (i, (u, v)) in enumerate(edges)])