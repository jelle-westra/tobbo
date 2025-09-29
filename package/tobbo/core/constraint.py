import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.geometry import MultiPolygon
from shapely.affinity import scale
from scipy.sparse.csgraph import minimum_spanning_tree

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .topology import Topology

@dataclass
class Constraint(ABC):
    weight: float

    @abstractmethod
    def compute(self, topology: Topology) -> float : ...

@dataclass
class VolumeConstraint(Constraint):
    max_relative_volume: float

    def compute(self, topology: Topology) -> float :
        A = (topology.geometry.area//1) / (topology.domain_size_x*topology.domain_size_y)
        return max(A - self.max_relative_volume, 0)
    
@dataclass
class DisconnectionConstraint(Constraint):
    topology: Topology
    boundaries: List[BaseGeometry]

    # this will be overridden in the __post_init__
    def compute(self, topology: Topology) -> float : return 1.

    def __post_init__(self):
        self.boundaries = [
            scale(b, xfact=1/self.topology.domain_size_x, yfact=1/self.topology.domain_size_y, origin=(0,0)) 
            for b in self.boundaries
        ]
        self.d_threshold = (1/2)*min(
             1/(self.topology.density*self.topology.domain_size_x), 1/(self.topology.density*self.topology.domain_size_y)
        )
        self.compute = self.compute_continuous if (self.topology.continuous) else self.compute_discrete

    def compute_discrete(self, topology: Topology) -> float :
        # operating in normalized space [0,1]^2
        geo = scale(topology.geometry, xfact=1/topology.domain_size_x, yfact=1/topology.domain_size_y, origin=(0,0))
        # checking shapely's dynamic typing
        components = geo.geoms if isinstance(geo, MultiPolygon) else [geo]

        n = len(components)
        if (n == 0) : return 1.
        D = np.zeros((n,n))
        if (n > 1):
            for i in range(n):
                for j in range(i) :
                        d_inter_component = components[i].distance(components[j])
                        # prevent checkerboard and staircase patterns in discrete topology
                        d_inter_component = max(d_inter_component, self.d_threshold)
                        D[i,j] = D[j,i] = d_inter_component
            D = minimum_spanning_tree(D).toarray()
        d = D.sum()

        for boundary in self.boundaries :
            d += geo.distance(boundary)
        # normalization, the largest distance in [0,1]^2 is sqrt(2)
        return d/np.sqrt(2)

    def compute_continuous(self, topology: Topology) -> float :
        # operating in normalized space [0,1]^2
        geo = scale(topology.geometry, xfact=1/topology.domain_size_x, yfact=1/topology.domain_size_y, origin=(0,0))
        # checking shapely's dynamic typing
        components = geo.geoms if isinstance(geo, MultiPolygon) else [geo]

        n = len(components)
        if (n == 0) : return 1.
        D = np.zeros((n,n))
        if (n > 1):
            for i in range(n):
                for j in range(i) :
                        d = components[i].distance(components[j])
                        # making sure the MST understands there is an edge, for 0 it thinks it's not connected,
                        # also if the values get too low (<1e-8) this can happen
                        D[i,j] = D[j,i] = d if (d > 1e-6) else -1

            D = minimum_spanning_tree(D).toarray()
            # filtering out the negative values and less than pixel distances for very close components
            D[D < self.d_threshold] = 0
        d = D.sum()

        for boundary in self.boundaries :
            d_to_boundary = geo.distance(boundary)
            if (d_to_boundary > self.d_threshold):
                d += d_to_boundary
        # normalization, the largest distance in [0,1]^2 is sqrt(2)
        return d/np.sqrt(2)
    
@dataclass
class ConstraintMix(Constraint):
    constraints: List[Constraint]
    offset: float

    def compute(self, topology: Topology): 
        g = sum(c.weight*c.compute(topology) for c in self.constraints)
        return self.offset + g if (g > 0) else 0.