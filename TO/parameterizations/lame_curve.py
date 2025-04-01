import numpy as np
from shapely.geometry import Polygon
from dataclasses import dataclass

from .rectangle import Rectangles

@dataclass
class LameCurves(Rectangles):
    m: int

    def compute_base_polygon(self) -> Polygon :
        t = np.linspace(0, 2*np.pi, self.n_samples)
        return Polygon(np.c_[
            np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/self.m), 
            np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/self.m)
        ])