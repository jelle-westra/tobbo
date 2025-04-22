import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry.base import BaseGeometry

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import os
from time import time
from typing import List

import ioh
from ioh.iohcpp import RealConstraint

from .constraint import Constraint
from .parameterization import Parameterization
from .topology import Topology
    
class Model(ABC):
    @abstractmethod
    def update(self, topology: Topology) : ...

@dataclass
class ProblemInstance(ioh.problem.RealSingleObjective):
    topology: Topology
    parameterization: Parameterization
    model: Model # TODO : implement some TO.core.model base class, and objective
    topology_constraints: List[Constraint]
    # objective: callable[]

    def __post_init__(self) -> None :
        self.x = float('nan')*np.ones(self.parameterization.dimension)
        self.x_best = self.x.copy()
        self.score = self.score_best = float('inf')
        # JELLE DEBUG
        self.count: int = 0
        self.start_time = time()
        self.budget = float('inf')

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

    def set_budget(self, budget) : 
        self.budget = budget
        self.configs = np.nan*np.ones((self.budget, self.parameterization.dimension))
        self.scores = np.nan*np.ones(self.budget)

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
        
        if (self.score < self.score_best) : (self.x_best, self.score_best) = (self.x.copy(), self.score)
        (self.configs[self.count-1], self.scores[self.count-1]) = (x, self.score)
        with open(os.path.join(self.logger_output_directory, 'evals.dat'), 'a') as handle :
            # TODO : reintroduce the constraint values in the evals.txt just to be sure, just use constraint.response
            handle.write(f'{self.count} {self.score} ') #{response_vol_original:.6f}\n')
            handle.write(' '.join(map(str, x)) + '\n')
        # TODO : let's first check if the mesh is connected before evluating, sometimes it can generates negative values for the HORIZONTAL loading problem
        return abs(self.score)
    
    def plot_best(self, ax: Axes=None) :
        if (ax is None) : ax = plt.gca()
        self.parameterization.update_topology(self.topology, self.x_best)
        self.topology.plot(ax)
        for c in self.topology_constraints:
            if hasattr(c, 'boundaries'):
                for b in c.boundaries : ax.plot(*b.xy, 'ko--', lw=2)