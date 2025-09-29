import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.plotting import plot_polygon

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Topology:
    continuous: bool
    domain_size: Tuple[float, float]
    density: float # [elements/squared domain unit]

    def __post_init__(self) :
        (self.domain_size_x, self.domain_size_y) = self.domain_size
        self.domain = box(0, 0, self.domain_size_x, self.domain_size_y)
        
        geometry: MultiPolygon = None
        mask: np.ndarray = None

    def plot(self, ax: Axes=None) :
        if (ax is None) : ax = plt.gca()
        plot_polygon(self.geometry, ax, add_points=False, lw=2)
        ax.plot(*self.domain.exterior.xy, 'k')
        ax.grid(False)
        ax.axis('equal')