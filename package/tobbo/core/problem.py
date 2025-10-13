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
        self.evaluations: int = 0
        self.simulation_calls: int = 0
        self.budget = float('inf')
        self.log = {'total_evaluations': [], 'simulations': [], 'fitness': [], 'configuration': []}

    def set_budget(self, budget) : 
        self.budget = budget
        with open(os.path.join(self.logger_output_directory, 'evals.dat'), 'a') as handle :
            handle.write(f'# total_evaluations, simulation_calls, fitness, configuration\n')
            handle.write(f'# [{self.simulation_calls}/{self.budget}] ' + str(datetime.now()) + '\n')

    def update(self, x:np.ndarray) -> None :
        # updating the topology geomtery and material mask
        if (self.x != x).any() : 
            self.parameterization.update_topology(self.topology, x)
            self.score = float('nan')
            self.x = x

    def log_call(self) -> None :
        improved_log = (self.score < self.score_best)
        intermediate_log = not(self.simulation_calls%100) and self.simulation_calls > 0


        if improved_log or intermediate_log: 
            with open(os.path.join(self.logger_output_directory, 'evals.dat'), 'a') as handle :
                if (intermediate_log) :
                    handle.write(f'# [{self.simulation_calls}/{self.budget}] ' + str(datetime.now()) + '\n')

                handle.write(f'{self.evaluations} {self.simulation_calls} {self.score} ')
                handle.write(' '.join(map(str, self.x)) + '\n')

                for (k, v) in zip(self.log.keys(), (self.evaluations, self.simulation_calls, self.score, self.x)):
                    self.log[k].append(v)

    def __call__(self, x: np.ndarray) -> float:
        self.evaluations += 1
        self.update(x)

        feasible = True
        for constraint in self.topology_constraints:
            constraint.response = constraint.compute(self.topology)
            if constraint.response : 
                self.score = constraint.response
                feasible = False; break
        # checking if objective still is 0., i.e. constraints are 0.
        if (feasible):
            self.score = self.evaluate(x)

        self.log_call()

        if (self.score < self.score_best) : 
            (self.x_best, self.score_best) = (self.x.copy(), self.score)

        return self.score

    def compute_constraint(self, constraint: Constraint, x: np.ndarray) :
        self.update(x) # updating the topology first if x has changed
        constraint.response = constraint.compute(self.topology)
        return constraint.response

    def evaluate(self, x):
        self.update(x)
        self.simulation_calls += 1
        if self.simulation_calls > self.budget : raise KeyboardInterrupt()

        # we pass the new topology corresponding to`x` to the simulation
        self.model.update(self.topology)
        self.score = self.objective(self.model)
        
        
        return self.score
    
    def plot_best(self, ax: Axes=None) :
        if (ax is None) : ax = plt.gca()
        self.parameterization.update_topology(self.topology, self.x_best)
        self.topology.plot(ax)
        for c in self.topology_constraints:
            if hasattr(c, 'boundaries'):
                for b in c.boundaries : ax.plot(*b.xy, 'ko--', lw=2)

    def plot_log(self, ax: Axes=None, inset_bounds=[0.4,0.4,0.5,0.5]):
        if (ax is None) : ax = plt.gca()
        simulations = np.arange(1,self.budget+1)
        best_fitness = np.inf*np.ones(self.budget, dtype=float)

        best_fitness[[i-1 for i in self.log['simulations']]] = self.log['fitness']
        best_fitness = np.minimum.accumulate(best_fitness)

        if (inset_bounds) : self.plot_best(ax.inset_axes(inset_bounds))

        ax.semilogy(simulations, best_fitness)
        ax.set_xlabel('Simulation Calls'); ax.set_ylabel('Compliance')