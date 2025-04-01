import numpy as np
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