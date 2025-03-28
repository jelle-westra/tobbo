import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from dataclasses import dataclass, replace
from typing import Tuple, List

from TO import Parameterization


@dataclass
class LameCurveConfig:
    x: float
    y: float
    theta: float
    a: float
    b: float
    m: int = 6
    
    def __post_init__(self) : assert not(self.m%2), '`m` parameter of level set should be even.'


# regular 2D rotation matrix
R2D = lambda phi : np.array([
    [np.cos(phi), -np.sin(phi)],
    [np.sin(phi),  np.cos(phi)]
])


def compute_curve(
    lc: LameCurveConfig,
    t: float, 
) -> Tuple[float, float] :
    X = np.c_[
        (lc.a/2) * np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/lc.m), 
        (lc.b/2) * np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/lc.m)
    ].T
    return (R2D(lc.theta) @ X).T + (lc.x, lc.y)


class LameCurves(Parameterization):
    lcs: List[LameCurveConfig]

    def __init__(self, symmetry: bool, m: int):
        self.m = m
        self.symmetry = symmetry

        self.dimension = 15
        self.n_samples = 1_000

        self.domain = box(0, 0, 100, 50)

        # TODO : calculate from domain
        self.normalization_factors = (
            100, 
            50/2 if symmetry else 50, 
            np.pi,
            np.hypot(100, 50/2) if symmetry else np.hypot(100, 50), 
            4*(5), 
        )

    def scale(self, x_configs: np.ndarray) -> np.ndarray :
        return x_configs*self.normalization_factors

    def compute_geometry(self, x: np.ndarray) -> MultiPolygon :
        x_configs = x.reshape(-1, 5)
        # TODO : remove class lcs as class attribute
        self.lcs = [LameCurveConfig(*config, self.m) for config in self.scale(x_configs)]
        if (self.symmetry) : 
            # TODO : use the domain to do this, something with the normalization factors
            self.lcs += [replace(lc, y=50-lc.y, theta=np.pi-lc.theta) for lc in self.lcs]

        t = np.linspace(0, 2*np.pi, self.n_samples)
        geo: MultiPolygon = MultiPolygon()
        for lc in self.lcs : 
            geo = geo.union(Polygon(compute_curve(lc, t)))
        return geo