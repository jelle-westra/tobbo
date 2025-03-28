import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from dataclasses import dataclass

@dataclass
class Topology:
    continuous: bool
    domain: Polygon
    geometry: MultiPolygon = None
    mask: np.ndarray = None