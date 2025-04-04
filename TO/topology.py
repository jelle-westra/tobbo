import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry import Polygon, MultiPolygon
from dataclasses import dataclass

@dataclass
class Topology:
    continuous: bool
    domain: Polygon
    density: float # [elements/squared domain unit]

    def __post_init__(self) :
        assert (self.domain.bounds[:2] == (0,0)), 'lower-left corner should be (0,0)'
        (_, _, *self.domain_size) = (_, _, self.domain_size_x, self.domain_size_y) = self.domain.bounds
        
        geometry: MultiPolygon = None
        mask: np.ndarray = None

    def plot(self, ax: Axes) :
        ax.plot(*self.domain.exterior.xy, 'k', lw=.8)
        for geo in self.geometry.geoms : 
            ax.plot(*geo.exterior.xy, lw=2)
            ax.fill(*geo.exterior.xy, alpha=.2)
        ax.axis('equal')