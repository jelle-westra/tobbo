import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from dataclasses import dataclass, replace

from TO import Parameterization, Topology

@dataclass 
class CapsuleConfig:
    x1: float
    y1: float
    x2: float
    y2: float
    r: float

# regular 2D rotation matrix
R2D = lambda phi : np.array([
    [np.cos(phi), -np.sin(phi)],
    [np.sin(phi),  np.cos(phi)]
])

def compute_curve(cap: CapsuleConfig, t: np.ndarray) -> np.ndarray :
    (p1, p2) = np.array([cap.x1, cap.y1]), np.array([cap.x2, cap.y2])

    phi = np.arctan2(*(p2 - p1)[::-1])
    d = np.linalg.norm(p2 - p1)

    xy = cap.r * np.c_[np.cos(2*np.pi*t - np.pi/2), np.sin(2*np.pi*t - np.pi/2)]
    xy[:t.size//2] += (d/2, 0)
    xy[t.size//2:] -= (d/2, 0)

    return (R2D(phi) @ xy.T).T + (p1 + p2)/2


def compute_polygon(cap: CapsuleConfig, n_samples: int) -> Polygon : 
    return Polygon(compute_curve(cap, np.linspace(0, 1, n_samples)))

@dataclass
class Capsules(Parameterization):
    topology: Topology 
    n_components_x: int
    n_components_y: int
    symmetry_y: bool
    n_samples: int
    # TODO : symmetry_x
    symmetry_x: bool=False

    def __post_init__(self):
        self.dimension = 5 * self.n_components_x * self.n_components_y
        self.normalization_factors = np.array([
            self.topology.domain_size_x, self.topology.domain_size_y,
            self.topology.domain_size_x, self.topology.domain_size_y,
            float('inf')
        ])
        if (self.symmetry_x) :
            assert not(self.n_components_x % 2), 'for using symmetry the number of components in x-direction needs to be even'
            # self.normalization_factors[[0,2]] /= 2.
            self.dimension = int(self.dimension/2)

        if (self.symmetry_y) : 
            assert not(self.n_components_y % 2), 'for using symmetry the number of components in y-direction needs to be even'
            # self.normalization_factors[[1,3]] /= 2.
            self.dimension = int(self.dimension/2)
        # taking the smallest as normalization for the width of the beam    
        self.normalization_factors[-1] = min(self.normalization_factors)


    def scale(self, x_configs: np.ndarray) -> np.ndarray :
        return x_configs*self.normalization_factors
    
    def compute_geometry(self, x: np.ndarray) -> np.ndarray :
        x_configs = x.reshape(-1, len(self.normalization_factors))
        caps = [CapsuleConfig(*config) for config in self.scale(x_configs)]

        if (self.symmetry_x) : # for all current configs mirror x-components
            caps += [replace(cap, x1=self.topology.domain_size_x-cap.x1, x2=self.topology.domain_size_x-cap.x2) for cap in caps]
        if (self.symmetry_y) : # for all currecnt configs mirror y-components, inlcuding ones just mirrored
            caps += [replace(cap, y1=self.topology.domain_size_y-cap.y1, y2=self.topology.domain_size_y-cap.y2) for cap in caps]

        geo = MultiPolygon()
        for cap in caps : # drawing and merging the polygons
            geo = geo.union(compute_polygon(cap, self.n_samples))
        return geo