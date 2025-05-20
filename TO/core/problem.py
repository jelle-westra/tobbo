import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry.base import BaseGeometry

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import os
from datetime import datetime
from typing import List

from .constraint import Constraint
from .parameterization import Parameterization
from .topology import Topology
from .model import Model

@dataclass
class ProblemInstance:
    topology: Topology
    parameterization: Parameterization
    model: Model
    topology_constraints: List[Constraint]
    objective: Callable[[Model], float]

    def __post_init__(self) -> None :
        self.x = float('nan')*np.ones(self.parameterization.dimension)
        self.x_best = self.x.copy()
        self.score = self.score_best = float('inf')
        # JELLE DEBUG
        self.count: int = 0
        self.budget = float('inf')

    def set_budget(self, budget) : 
        self.budget = budget

    def update(self, x:np.ndarray) -> None :
        # updating the topology geomtery and material mask
        if (self.x != x).any() : 
            self.parameterization.update_topology(self.topology, x)
            self.score = float('nan')
            self.x = x

    def __call__(self, x: np.ndarray):
        self.update(x)
        for constraint in self.topology_constraints:
            constraint.response = constraint.compute(self.topology)
            if constraint.response : return constraint.response
        return self.evaluate(x)

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
        self.score = self.objective(self.model)
        
        if (self.score < self.score_best) or not(self.count%100): 
            with open(os.path.join(self.logger_output_directory, 'evals.dat'), 'a') as handle :
                if not(self.count%100) :
                    handle.write(f'# [{self.count}/{self.budget}] ' + str(datetime.now()))
                # TODO : reintroduce the constraint values in the evals.txt just to be sure, just use constraint.response
                handle.write(f'{self.count} {self.score} ')
                handle.write(' '.join(map(str, x)) + '\n')
            if (self.score < self.score_best) : 
                (self.x_best, self.score_best) = (self.x.copy(), self.score)
        return abs(self.score)
    
    def plot_best(self, ax: Axes=None) :
        if (ax is None) : ax = plt.gca()
        self.parameterization.update_topology(self.topology, self.x_best)
        self.topology.plot(ax)
        for c in self.topology_constraints:
            if hasattr(c, 'boundaries'):
                for b in c.boundaries : ax.plot(*b.xy, 'ko--', lw=2)