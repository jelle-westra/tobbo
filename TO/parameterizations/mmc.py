import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from typing import List, Union

from TO import Parameterization, Topology

class MMCConfig(ABC):
    @staticmethod
    @abstractmethod
    def get_normalization_factors(topology: Topology, symmetry_x, symmetry_y) -> np.ndarray : ...

    @abstractmethod
    def to_angular(self) -> 'MMCAngularConfig' : ...

@dataclass
class MMCAngularConfig(MMCConfig):
    x: float
    y: float
    rx: float
    ry: float
    theta: float

    @staticmethod
    def get_normalization_factors(topology: Topology, symmetry_x, symmetry_y) -> np.ndarray :
        normalization_factors = np.array([
            topology.domain_size_x, topology.domain_size_y, # (x,y)
            float('nan'), float('nan'), np.pi # (rx, ry, theta)
        ])
        if (symmetry_x) : normalization_factors[0] /= 2.
        if (symmetry_y) : normalization_factors[1] /= 2.
        normalization_factors[[2,3]] = np.hypot(topology.domain_size_x, topology.domain_size_y)/2
        return normalization_factors
    
    def to_angular(self) -> 'MMCAngularConfig' : return self 
    
@dataclass 
class MMCEndpointsConfig(MMCConfig):
    x1: float
    y1: float
    x2: float
    y2: float
    r: float

    @staticmethod
    def get_normalization_factors(topology: Topology, symmetry_x, symmetry_y) -> np.ndarray :
        normalization_factors = np.array([
            topology.domain_size_x, topology.domain_size_y, # (x1, y1)
            topology.domain_size_x, topology.domain_size_y, # (x2, y2)
            float('nan') # (r)
            ])
        normalization_factors[-1] = np.hypot(topology.domain_size_x, topology.domain_size_y)/2
        return normalization_factors
    
    def to_angular(self) -> MMCAngularConfig :
        p1 = np.array([self.x1, self.y1])
        p2 = np.array([self.x2, self.y2])

        theta = np.arctan2(self.y1 - self.y2, self.x1 - self.x2)
        if (theta < 0) : theta += np.pi

        return MMCAngularConfig(*(p1 + p2)/2, *(np.linalg.norm(p1 - p2)/2, self.r), theta)
    

class MMCCenterpointsConfig(MMCEndpointsConfig):
    def to_angular(self) -> MMCAngularConfig :
        config: MMCAngularConfig = super().to_angular()
        return replace(config, rx=config.rx + self.r)

@dataclass
class MMC(Parameterization, ABC) :
    topology: Topology 
    n_components_x: int
    n_components_y: int
    symmetry_y: bool
    symmetry_x: bool
    representation: MMCConfig
    n_samples: int

    @abstractmethod
    def compute_base_polygon() -> Polygon : ...

    def __post_init__(self):
        self.dimension = 5 * self.n_components_x * self.n_components_y
        if (self.symmetry_x) :
            assert not(self.n_components_x % 2), 'for using symmetry the number of components in x-direction needs to be even'
            self.dimension = int(self.dimension/2)

        if (self.symmetry_y) : 
            assert not(self.n_components_y % 2), 'for using symmetry the number of components in y-direction needs to be even'
            self.dimension = int(self.dimension/2)

        self.normalization_factors = self.representation.get_normalization_factors(
            self.topology, self.symmetry_x, self.symmetry_y
        )
        self.base_polygon: Polygon = self.compute_base_polygon()

    def scale(self, x_configs: np.ndarray) -> np.ndarray :
        return x_configs*self.normalization_factors
    
            
    def compute_geometry(self, x: np.ndarray) -> np.ndarray :
        x_configs = x.reshape(-1, len(self.normalization_factors))

        mmcs: List[MMCAngularConfig] = [self.representation(*config).to_angular() for config in self.scale(x_configs)]

        if (self.symmetry_x) : # for all current configs mirror x-components
            mmcs += [replace(mmc, x=self.topology.domain_size_x-mmc.x, theta=np.pi-mmc.theta) for mmc in mmcs]
        if (self.symmetry_y) : # for all currecnt configs mirror y-components, inlcuding ones just mirrored
            mmcs += [replace(mmc, y=self.topology.domain_size_y-mmc.y, theta=np.pi-mmc.theta) for mmc in mmcs]

        geo = MultiPolygon()
        for mmc in mmcs : # drawing and merging the polygons
            geo = geo.union(self.compute_polygon(mmc))
        return geo