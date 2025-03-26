import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from dataclasses import dataclass, replace

from problem import Parameterization

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

        
class Capsules(Parameterization):
    def __init__(self, symmetry: bool) -> None :
        self.symmetry = symmetry

        self.dimension = 15
        self.domain = box(0, 0, 100, 50)

        self.n_samples = 1_000

        self.normalization_factors = (
            100, 
            50/2 if symmetry else 50, 
            100, 
            50/2 if symmetry else 50, 
            4*(5)
        )

    def scale(self, x_configs: np.ndarray) -> np.ndarray :
        return x_configs*self.normalization_factors
    
    def compute_geometry(self, x: np.ndarray) -> np.ndarray :
        x_configs = x.reshape(-1, 5)
        caps = [CapsuleConfig(*config) for config in self.scale(x_configs)]
        if (self.symmetry) :
            # TODO : use the domain to do this, something with the normalization factors
            caps += [replace(cap, y1=50-cap.y1, y2=50-cap.y2) for cap in caps]

        t = np.linspace(0, 1, self.n_samples)
        geo = MultiPolygon()
        for cap in caps :
            geo = geo.union(Polygon(compute_curve(cap, t)))
        return geo