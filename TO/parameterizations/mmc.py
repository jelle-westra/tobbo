import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.affinity import scale, rotate, translate

from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from typing import List, Union

from TO import Parameterization, Topology

__all__ = [
    'MMCConfig', 'MMCAngularConfig', 'MMCEndpointsConfig', 'MMCCenterpointsConfig', 'MMCAxiSymmetricConfig', 'MMC'
    'Rectangles', 'Capsules', 'LameCurves', 'Ellipses', 'Circles'
]

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
class MMCAxiSymmetricConfig(MMCConfig):
    # essentially a shape that has a single radius component, i.e. rx=ry
    # and is axi-symmetric, theta = 0
    x: float
    y: float
    r: float

    @staticmethod
    def get_normalization_factors(topology: Topology, symmetry_x: bool, symmetry_y: bool) -> np.ndarray :
        normalization_factors = np.array([
            topology.domain_size_x, topology.domain_size_y, # (x,y, r)
            np.hypot(topology.domain_size_x, topology.domain_size_y)/2
        ])
        if (symmetry_x) : normalization_factors[0] /= 2
        if (symmetry_y) : normalization_factors[1] /= 2
        return normalization_factors
    
    def to_angular(self) -> MMCAngularConfig:
        return MMCAngularConfig(self.x, self.y, self.r, self.r, theta=0)

@dataclass
class MMC(Parameterization, ABC) :
    topology: Topology 
    n_components: int
    symmetry_x: bool
    symmetry_y: bool
    infill_parameter: bool
    representation: MMCConfig
    n_samples: int

    @abstractmethod
    def compute_base_polygon() -> Polygon : ...

    def __post_init__(self):
        self.normalization_factors = self.representation.get_normalization_factors(
            self.topology, self.symmetry_x, self.symmetry_y
        )
        self.dimnesion_per_mmc = len(self.normalization_factors) + self.infill_parameter
        self.dimension = self.dimnesion_per_mmc * self.n_components
        self.base_polygon: Polygon = self.compute_base_polygon()

    def scale(self, x_configs: np.ndarray) -> np.ndarray :
        return x_configs*self.normalization_factors
    
    def compute_geometry(self, x: np.ndarray) -> np.ndarray :
        x_configs = x.reshape(-1, self.dimnesion_per_mmc)
        if (self.infill_parameter) : 
            x_infill = x_configs[:,-1]
            x_configs = x_configs[:,:-1]
        else:
            x_infill = np.zeros(len(x_configs))

        mmcs: List[MMCAngularConfig] = [self.representation(*config).to_angular() for config in self.scale(x_configs)]

        geo = MultiPolygon() # drawing and merging the polygons
        for (mmc, ins) in zip(mmcs, x_infill) : 
            poly = self.compute_polygon(mmc)
            if (self.infill_parameter) :
                poly = poly.difference(poly.buffer(-min(mmc.rx, mmc.ry)*ins)).buffer(0.1)
            geo = geo.union(poly)
        
        if (self.symmetry_x) : geo = geo.union(scale(geo, -1, 1, origin=(self.topology.domain_size_x/2,0)))
        if (self.symmetry_y) : geo = geo.union(scale(geo, 1, -1, origin=(0,self.topology.domain_size_y/2)))
        return geo
    
# PARAMETERIZATIONS

class Rectangles(MMC):
    def compute_base_polygon(self) -> Polygon :
        assert (self.n_samples >= 4), ''
        return box(-1, -1, 1, 1) # independent on the number of samples
    
    def compute_polygon(self, config: MMCAngularConfig) -> Polygon :
        return translate(rotate(scale(
                self.base_polygon, config.rx, config.ry # scale
            ), config.theta, use_radians=True # rotate
        ), config.x, config.y) # translate

class Capsules(MMC) :
    def compute_base_polygon(self) -> Polygon :
        # NOTE: the base shape of the capsule is a semi-circle
        # on `compute_polygon` we then first move this semicircle to the endpoint mirror and connect it to the other side
        return Polygon(np.c_[np.cos(t := np.linspace(-np.pi/2, np.pi/2, self.n_samples//2)), np.sin(t)])

    def compute_polygon(self, config: MMCAngularConfig) -> Polygon :
        r = min(config.rx, config.ry)
        # move the semi-circle to the capsules endpoint
        poly: Polygon = translate(scale(self.base_polygon, r, r, origin=(0,0)), config.rx-r, 0).union(
            box(0,-r, config.rx-r, r) # adding the straight middle portion of the capsule
        ).buffer(1e-3)
        # mirroring the half capsule on the y-axis to create the full geo
        poly = poly.union(scale(poly, -1, 1, origin=(0,0))).buffer(1e-3)

        return translate(rotate(
            poly, config.theta, use_radians=True # rotate
        ), config.x, config.y) # translate
    
@dataclass
class LameCurves(Rectangles):
    m: int

    def compute_base_polygon(self) -> Polygon :
        t = np.linspace(0, 2*np.pi, self.n_samples)
        return Polygon(np.c_[
            np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/self.m), 
            np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/self.m)
        ])

class Ellipses(LameCurves):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, m=2)

class Circles(Ellipses):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, representation=MMCAxiSymmetricConfig)

