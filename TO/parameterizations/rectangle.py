import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from dataclasses import dataclass, replace

from TO import Parameterization

@dataclass 
class RectangleConfig:
    x1: float
    y1: float
    x2: float
    y2: float
    t: float

def compute_curve(rect: RectangleConfig) -> np.ndarray :
    n = (-(rect.y2 - rect.y1), (rect.x2 - rect.x1))
    n = n/np.linalg.norm(n)
    (p1, p2) = (rect.x1, rect.y1), (rect.x2, rect.y2)
    return Polygon([p1 - (rect.t/2)*n, p1 + (rect.t/2)*n, p2 + (rect.t/2)*n, p2 - (rect.t/2)*n])
        
class Rectangles(Parameterization):
    def __init__(self, symmetry: bool) -> None :
        self.symmetry = symmetry

        self.dimension = 15
        self.domain = box(0, 0, 100, 50)

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
        rects = [RectangleConfig(*config) for config in self.scale(x_configs)]
        if (self.symmetry) :
            # TODO : use the domain to do this, something with the normalization factors
            rects += [replace(rect, y1=50-rect.y1, y2=50-rect.y2) for rect in rects]

        geo = MultiPolygon()
        for rect in rects :
            geo = geo.union(Polygon(compute_curve(rect)))
        return geo