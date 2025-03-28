import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from rasterio.features import rasterize

from dataclasses import dataclass
from abc import ABC, abstractmethod

from .topology import Topology

class Parameterization(ABC) :
    domain: Polygon # bounding box
    dimension: int

    @abstractmethod
    def compute_geometry(self, x: np.ndarray) -> MultiPolygon : ...

    @staticmethod
    def compute_geometry_from_image(mask: np.ndarray) -> MultiPolygon :
        # TODO : here
        raise NotImplemented('TODO')
    
    @staticmethod
    def rasterize_geometry(geo: MultiPolygon) -> np.ndarray:
        # TODO : use domain to do this
        return rasterize(geo.geoms, (50, 100))

    def update_topology(self, topology: Topology, x: np.ndarray) -> None :
        geo = self.compute_geometry(x).intersection(topology.domain)
        topology.geometry = geo if isinstance(geo, MultiPolygon) else MultiPolygon([geo])
        topology.mask = Parameterization.rasterize_geometry(topology.geometry).astype(bool)

        if not(topology.continuous) : 
            topology.geometry = Parameterization.compute_geometry_from_image(topology.mask)