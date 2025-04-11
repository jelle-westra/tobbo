import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import scale, rotate, translate
from shapely import unary_union

from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from typing import List, ClassVar

from TO import Parameterization, Topology

__all__ = [
    'MMCConfig', 'MMCAngularConfig', 'MMCEndpointsConfig', 'MMCCenterpointsConfig', 'MMCAxiSymmetricConfig', 'MMC'
    'Rectangles', 'Capsules', 'LameCurves', 'Ellipses', 'Circles'
]

def sample_equidistant_pts(pts: np.ndarray, n_samples: int) -> np.ndarray :
    # inspiration from : https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values
    d = np.zeros(len(pts)) # cumulative distance over the points
    d[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1).cumsum()

    d_equidistant = np.linspace(0,d.max(),n_samples) # sampling equidistant points
    return np.c_[[np.interp(d_equidistant, d, xi) for xi in pts.T]].T

class MMCConfig(ABC):
    dimension: int=5
    @classmethod
    def get_normalization_scale(cls, topology: Topology, symmetry_x, symmetry_y) -> np.ndarray :
        return np.ones(cls.dimension)
    
    @classmethod
    def get_normalization_shift(cls, topology: Topology, symmetry_x, symmetry_y) -> np.ndarray :
        return np.zeros(cls.dimension)

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
    def get_normalization_scale(topology: Topology, symmetry_x, symmetry_y) -> np.ndarray :
        normalization_scale = np.array([
            topology.domain_size_x, topology.domain_size_y, # (x,y)
            float('nan'), float('nan'), np.pi # (rx, ry, theta)
        ])
        if (symmetry_x) : normalization_scale[0] /= 2.
        if (symmetry_y) : normalization_scale[1] /= 2.
        normalization_scale[[2,3]] = np.hypot(topology.domain_size_x, topology.domain_size_y)/2
        return normalization_scale
    
    def to_angular(self) -> 'MMCAngularConfig' : return self 
    
@dataclass 
class MMCEndpointsConfig(MMCConfig):
    x1: float
    y1: float
    x2: float
    y2: float
    r: float

    @staticmethod
    def get_normalization_scale(topology: Topology, symmetry_x, symmetry_y) -> np.ndarray :
        normalization_scale = np.array([
            topology.domain_size_x, topology.domain_size_y, # (x1, y1)
            topology.domain_size_x, topology.domain_size_y, # (x2, y2)
            np.hypot(topology.domain_size_x, topology.domain_size_y)/2 # (r)
            ])
        return normalization_scale
    
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
    dimension: ClassVar[int]= 3

    @staticmethod
    def get_normalization_scale(topology: Topology, symmetry_x: bool, symmetry_y: bool) -> np.ndarray :
        normalization_scale = np.array([
            topology.domain_size_x, topology.domain_size_y, # (x,y, r)
            np.hypot(topology.domain_size_x, topology.domain_size_y)/2
        ])
        if (symmetry_x) : normalization_scale[0] /= 2
        if (symmetry_y) : normalization_scale[1] /= 2
        return normalization_scale
    
    def to_angular(self) -> MMCAngularConfig:
        return MMCAngularConfig(self.x, self.y, self.r, self.r, theta=0)

class MMCDeformer:
    dimension: int=1

    @classmethod
    def get_normalization_scale(cls, topology: Topology, symmetry_x: bool, symmetry_y: bool) -> np.ndarray :
        return np.ones(cls.dimension)
    
    @classmethod
    def get_normalization_shift(cls, topology: Topology, symmetry_x: bool, symmetry_y: bool) -> np.ndarray :
        return np.zeros(cls.dimension)

    @staticmethod
    def deform_pre_scale_x(geo: Polygon, config: MMCAngularConfig, x: np.ndarray) -> Polygon :
        return geo
    
    @staticmethod
    def deform_pre_scale_y(geo: Polygon, config: MMCAngularConfig, x: np.ndarray) -> Polygon :
        return geo

class StraightBeam(MMCDeformer):
    dimension: int = 0

@dataclass
class GuoBeam(StraightBeam):
    n_samples: int
    dimension: ClassVar[int] = 5

    def get_normalization_scale(self, topology: Topology, symmetry_x: bool, symmetry_y: bool) :
        self.rnorm = np.hypot(topology.domain_size_x, topology.domain_size_y)/2
        # we normalize r1 and r2 as it is -> 0-1 instead of rnorm
        # and b -> [-2, 2]
        return np.array([1, 1, self.rnorm, 4, np.pi])

    def deform_pre_scale_y(self, geo: Polygon, config: MMCAngularConfig, x_scaled: np.ndarray):
        (r_left, r_right, a, b, phi) = x_scaled.flatten()
        (a, b, phi) = (a-self.rnorm/2, b-2, phi-np.pi/2)

        # let's linearly map 0-1 to 0.25-4 * config.ry
        r_left = ((4 - 0.25)*r_left + 0.25) * config.ry
        r_right = ((4 - 0.25)*r_right + 0.25) * config.ry

        (x, y) = sample_equidistant_pts(np.c_[geo.exterior.xy], self.n_samples).T

        ry = (r_left + r_right - 2*config.ry) / 2 * (x/config.rx)**2 + (r_right - r_left)/2 * (x/config.rx) + config.ry
        f = a*np.sin(b*(x/config.rx + phi))
        y = (f + ry*y)/config.ry
        
        return Polygon(np.c_[x, y]).buffer(1e-2)

@dataclass
class HarmonicDeformer(MMCDeformer):
    order: int
    n_samples: int
    dimension: ClassVar[int]=1

    def get_normalization_scale(self, topology: Topology, symmetry_x: bool, symmetry_y: bool) :
        self.rnorm = np.hypot(topology.domain_size_x, topology.domain_size_y)/4
        return np.array([1.,])

    def deform_pre_scale_y(self, geo: Polygon, config: MMCAngularConfig, x_scaled: np.ndarray):
        (x, y) = sample_equidistant_pts(np.c_[geo.exterior.xy], self.n_samples).T

        y += self.rnorm/config.ry*(2*x_scaled-1) * np.cos(np.pi/2*self.order*(x/config.rx) + (self.order-1)*np.pi/2)
        
        return Polygon(np.c_[x, y]).buffer(1e-2)
    
@dataclass
class CosineDeformer(MMCDeformer):
    n_samples: int
    dimension: ClassVar[int] = 1

    def get_normalization_scale(self, topology: Topology, symmetry_x: bool, symmetry_y: bool) :
        self.rnorm = np.hypot(topology.domain_size_x, topology.domain_size_y)/4
        return np.array([1.,])

    def deform_pre_scale_y(self, geo: Polygon, config: MMCAngularConfig, x_scaled: np.ndarray):
        (x, y) = sample_equidistant_pts(np.c_[geo.exterior.xy], self.n_samples).T

        y += self.rnorm/config.ry*(2*x_scaled-1) * (np.cos(np.pi*(x/config.rx)) + 1)/2
        
        return Polygon(np.c_[x, y]).buffer(1e-2)
    
@dataclass
class QuadraticBezierDeformer(MMCDeformer):
    n_samples: int
    dimension: ClassVar[int] = 2

    def get_normalization_scale(self, topology: Topology, symmetry_x: bool, symmetry_y: bool) -> np.ndarray:
        self.rnorm = np.hypot(topology.domain_size_x, topology.domain_size_y)/4
        return np.array([1., 1.])
    
    def deform_pre_scale_y(self, geo: Polygon, config: MMCAngularConfig, x_scaled: np.ndarray) -> Polygon:
        (x, y) = sample_equidistant_pts(np.c_[geo.exterior.xy], self.n_samples).T

        # keeping the bezier curve completely in normalized space
        P1 = (2*x_scaled - 1)
        P0 = (-1, 0)
        P2 = ( 1, 0)

        t = np.linspace(0, 1, 100).reshape(-1, 1)
        Q0 = (1-t)*P0 + t*P1
        Q1 = (1-t)*P1 + t*P2

        (x_bezier, y_bezier) = ((1-t)*Q0 + t*Q1).T

        y += self.rnorm/config.ry * np.interp(x/config.rx, x_bezier, y_bezier)

        return Polygon(np.c_[x, y]).buffer(1e-2)
    
class Infill(MMCDeformer):
    dimension: ClassVar[int] = 1
    
    def deform_pre_scale_y(self, geo: Polygon, config: MMCAngularConfig, x_scaled: np.ndarray) -> Polygon:
        geo = scale(geo, 1, config.ry)
        geo = geo.difference(geo.buffer(-min(config.rx, config.ry)*x_scaled[0])).buffer(1e-2)
        return scale(geo, 1, 1/config.ry)


class MMCDeformerPipeline(MMCDeformer):
    def __init__(self, deformers: List[MMCDeformer]) :
        self.deformers = deformers
        self.dimension = sum(d.dimension for d in self.deformers)

    def get_normalization_scale(self, topology: Topology, symmetry_x: bool, symmetry_y: bool) -> np.ndarray:
        return np.r_[*[d.get_normalization_scale(topology, symmetry_x, symmetry_y) for d in self.deformers]]
    
    def get_normalization_shift(self, topology: Topology, symmetry_x: bool, symmetry_y: bool) -> np.ndarray:
        return np.r_[*[d.get_normalization_shift(topology, symmetry_x, symmetry_y) for d in self.deformers]]
    
    def deform_pre_scale_x(self, geo: Polygon, config: MMCAngularConfig, x_scaled: np.ndarray) -> Polygon :
        i = 0
        for d in self.deformers : 
            geo = d.deform_pre_scale_x(geo, config, x_scaled[i:i+d.dimension])
            i += d.dimension
        return geo
    
    def deform_pre_scale_y(self, geo: Polygon, config: MMCAngularConfig, x_scaled: np.ndarray) -> Polygon :
        i = 0
        for d in self.deformers :
            geo = d.deform_pre_scale_y(geo, config, x_scaled[i:i+d.dimension])
            i += d.dimension
        return geo

@dataclass
class MMC(Parameterization, ABC) :
    topology: Topology 
    n_components: int
    symmetry_x: bool
    symmetry_y: bool
    representation: MMCConfig
    deformer: MMCDeformer
    n_samples: int

    @abstractmethod
    def compute_base_polygon() -> Polygon : ...

    def __post_init__(self):
        self.normalization_scale = np.r_[
            self.representation.get_normalization_scale(self.topology, self.symmetry_x, self.symmetry_y),
            self.deformer.get_normalization_scale(self.topology, self.symmetry_x, self.symmetry_y),
        ]
        self.normalization_shift = np.r_[
            self.representation.get_normalization_shift(self.topology, self.symmetry_x, self.symmetry_y),
            self.deformer.get_normalization_shift(self.topology, self.symmetry_x, self.symmetry_y),
        ]
        self.dimension_per_mmc = self.representation.dimension
        self.dimension = self.dimension_per_mmc * self.n_components
        self.base_polygon: Polygon = self.compute_base_polygon()

    def scale(self, x_configs: np.ndarray) -> np.ndarray :
        return x_configs*self.normalization_scale + self.normalization_shift
    
    def compute_geometry(self, x: np.ndarray) -> np.ndarray :
        x_configs = self.scale(x.reshape(-1, self.dimension_per_mmc))
        if (self.deformer.dimension > 0) :
            (x_mmc, x_deformer) = (x_configs[:,:-self.deformer.dimension], x_configs[:,-self.deformer.dimension:])
        else :
            (x_mmc, x_deformer) = (x_configs, np.array([[]]*len(x_configs)))
  
        mmcs: List[MMCAngularConfig] = [self.representation(*config).to_angular() for config in x_mmc]
        return unary_union([self.compute_polygon(config, x_tr) for (config, x_tr) in zip(mmcs, x_deformer)])
    
    def compute_polygon(self, config: MMCAngularConfig, x_deformer: np.ndarray) -> Polygon :
        return translate(rotate(
            self.scale_y(
                self.deformer.deform_pre_scale_y(
                    self.scale_x(
                        self.deformer.deform_pre_scale_x( 
                            self.base_polygon, config, x_deformer # pre-scale x
                        ), config # scale x
                    ), config, x_deformer # pre-scale y
                ), config # scale y
            ), # scaling
            config.theta, origin=(0,0), use_radians=True # rotate
        ), config.x, config.y) # translate
    
    def scale_x(self, geo: Polygon, config: MMCAngularConfig) -> Polygon : 
        return scale(geo, config.rx, 1, origin=(0,0))

    def scale_y(self, geo: Polygon, config: MMCAngularConfig) -> Polygon : 
        return scale(geo, 1, config.ry, origin=(0,0))
    
# PARAMETERIZATIONS

class Rectangles(MMC):
    def compute_base_polygon(self) -> Polygon :
        assert (self.n_samples >= 4), ''
        return box(-1, -1, 1, 1) # independent on the number of samples

class Capsules(MMC) :
    def compute_base_polygon(self) -> Polygon :
        # NOTE: the base shape of the capsule is a semi-circle
        # on `compute_polygon` we then first move this semicircle to the endpoint mirror and connect it to the other side
        return Polygon(np.c_[np.cos(t := np.linspace(-np.pi/2, np.pi/2, self.n_samples//2)), np.sin(t)])

    def scale_x(self, geo: Polygon, config: MMCAngularConfig) -> Polygon :
        r = min(config.rx, config.ry)
        # move the semi-circle to the capsules endpoint
        geo: Polygon = translate(scale(geo, r, 1, origin=(0,0)), config.rx-r, 0).union(
            box(0,-1, config.rx-r, 1) # adding the straight middle portion of the capsule
        ).buffer(1e-3)
        # mirroring the half capsule on the y-axis to create the full geo
        return geo.union(scale(geo, -1, 1, origin=(0,0))).buffer(1e-3)
    
@dataclass
class LameCurves(MMC):
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