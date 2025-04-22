import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry import Polygon, MultiPolygon
from shapely.plotting import plot_polygon

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

    def plot(self, ax: Axes=None) :
        if (ax is None) : ax = plt.gca()
        plot_polygon(self.geometry, ax, add_points=False, lw=2)
        ax.plot(*self.domain.exterior.xy, 'k')
        ax.grid(False)
        ax.axis('equal')