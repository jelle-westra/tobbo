import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry.base import BaseGeometry

from dataclasses import dataclass
from time import time
from typing import List, Set
import os
from functools import partial
from abc import ABC, abstractmethod

import ioh
from ioh.iohcpp import RealConstraint
from ioh import ConstraintEnforcement

from .models import BinaryElasticMembraneModel
from .parameterization import Parameterization
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

@dataclass
class ProblemInstance(ioh.problem.RealSingleObjective):
    topology: Topology
    parameterization: Parameterization
    model: BinaryElasticMembraneModel
    topology_constraints: List[Constraint]
    budget: int
    # objective: Objective

    def __post_init__(self) -> None :
        self.x = float('nan')*np.ones(self.parameterization.dimension)
        self.score = float('nan')
        # JELLE DEBUG
        self.count: int = 0
        self.start_time = time()

        bounds = ioh.iohcpp.RealBounds(self.parameterization.dimension, 0.0, 1.0)
        optimum = ioh.iohcpp.RealSolution([0]*self.parameterization.dimension, 0.0)

        super().__init__(
            name='TopologyOptimization',
            n_variables=self.parameterization.dimension,
            instance=0,
            is_minimization=True,
            bounds= bounds,
            optimum=optimum
        )
        for constraint in self.topology_constraints:
            super().add_constraint(RealConstraint(
                partial(self.compute_constraint, constraint),
                name=str(constraint),
                enforced=constraint.enforcement,
                weight = constraint.weight, 
                exponent=1.0
            ))

    def update(self, x:np.ndarray) -> None :
        # updating the topology geomtery and material mask
        if (self.x != x).any() : 
            self.parameterization.update_topology(self.topology, x)
            self.score = float('nan')
            self.x = x

    def compute_constraint(self, constraint: Constraint, x: np.ndarray) :
        self.update(x) # updating the topology first if x has changed
        constraint.response = constraint.compute(self.topology)
        return constraint.response

    def evaluate(self, x):
        self.update(x)
        # JELLE DEBUG
        self.count += 1
        if self.count > self.budget : raise KeyboardInterrupt()

        # we pass the new topology corresponding to`x` to the simulation
        self.model.update(self.topology)
        self.score = self.model.compute_element_compliance().sum()

        with open(os.path.join(self.logger_output_directory, 'evals.dat'), 'a') as handle :
            # TODO : reintroduce the constraint values in the evals.txt just to be sure, just use constraint.response
            handle.write(f'{self.count} {self.score} ') #{response_vol_original:.6f}\n')
            handle.write(' '.join(map(str, x)) + '\n')
        # TODO : let's first check if the mesh is connected before evluating, sometimes it can generates negative values for the HORIZONTAL loading problem
        return abs(self.score)