import numpy as np
from shapely.geometry.base import BaseGeometry
from scipy.sparse.csgraph import minimum_spanning_tree

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ioh import ConstraintEnforcement

from .topology import Topology

@dataclass
class Constraint(ABC):
    weight: float
    enforcement: ConstraintEnforcement

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
    boundaries: List[BaseGeometry]

    def compute(self, topology: Topology) -> float :
        components = list(topology.geometry.geoms)

        n = len(components)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i) :
                    d = components[i].distance(components[j])
                    # making sure the MST understands there is an edge, for 0 it thinks it's not connected,
                    # also if the values get too low (<1e-8) this can happen
                    D[i,j] = D[j,i] = d if (d > 1e-6) else -1

        D = minimum_spanning_tree(D).toarray()
        # TODO : reimplement desnity-based thresholding
        D[D < 1/2] = 0
        d = D.sum()

        for boundary in self.boundaries :
            if (distance_to_boundary := topology.geometry.distance(boundary)) > 1/2 : 
                d += distance_to_boundary
        return d