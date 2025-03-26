import numpy as np

from dataclasses import dataclass
from time import time
import os

import simulation.FEA as FEA
from parameterization import Parameterization, Topology
import ioh
from ioh.iohcpp import RealConstraint

from shapely import Point


@dataclass
class ProblemTO(ioh.problem.RealSingleObjective):
    topology: Topology
    parameterization: Parameterization
    model: FEA.BinaryElasticMembraneModel
    # objective: Objective

    def __post_init__(self) -> None :
        self.x = float('nan')*np.ones(self.parameterization.dimension)
        self.score = float('nan')

        # TODO : make the constraints modular
        self.max_relative_volume = 0.5

        # JELLE DEBUG
        self.count: int = 0
        self.budget: int = 50
        self.response_vol: float = 0
        self.response_yaxis: float = 0
        self.response_pt: float = 0
        self.response_disc: float = 0
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
        super().add_constraint(RealConstraint(
            self.volume,
            name='Volume condition',
            enforced=ioh.ConstraintEnforcement.HARD,
            weight = 1e9, 
            exponent=1.0
        ))
        super().add_constraint(RealConstraint(
            self.disconnection,
            name='Disconnection condition',
            enforced=ioh.ConstraintEnforcement.HARD,
            weight = 1e3, 
            exponent=1.0
        ))

    def update(self, x:np.ndarray) -> None :
        # updating the topology geomtery and material mask
        if (self.x != x).any() : 
            self.parameterization.update_topology(self.topology, x)
            self.score = float('nan')
            self.x = x

    def evaluate(self, x):
        self.count += 1
        if self.count > self.budget : exit()

        self.update(x)
        # we pass the new topology corresponding to`x` to the simulation
        self.model.update(self.topology)
        self.score = self.model.compute_element_compliance().sum()

        volume_exceeded = (self.topology.mask.sum()/self.topology.mask.size > self.max_relative_volume)
        with open(os.path.join(self.logger_output_directory, 'evals.dat'), 'a') as handle :
            handle.write(f'{self.count} {self.score} {self.response_vol:.6f} {volume_exceeded:d}\n') #{response_vol_original:.6f}\n')

        return self.score
    
    def volume(self, x: np.ndarray) -> float :
        self.update(x)

        # floor it to one pixel, and normalize
        # TODO : use the domain here
        A: float = (self.topology.geometry.area//1) / (100 * 50)

        self.response_vol = max(A - self.max_relative_volume, 0)
        return self.response_vol
    
    
    def disconnection(self, x: np.ndarray) -> float :
        self.update(x)

        # JELLE DEBUG :
        if (self.count < 10) and (time() - self.start_time) > 60 : exit()

        from shapely.geometry import Point, LineString
        from scipy.sparse.csgraph import minimum_spanning_tree

        components = list(self.topology.geometry.geoms)

        n = len(components)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i) :
                    d = components[i].distance(components[j])
                    # making sure the MST understands there is an edge, for 0 it thinks it's not connected,
                    # also if the values get too low (<1e-8) this can happen
                    D[i,j] = D[j,i] = d if (d > 1e-6) else -1

         # this does not perform MST since it does not overwrite D (first its converted to sparse, it overwrites that instance)
        # TODO : make a boolean for this to activate or not
        D = minimum_spanning_tree(D, overwrite=True).toarray()
        D[D < 1/2] = 0
        d_MST = D.sum()

        # TODO : use the domain here
        pt: Point = Point(100, 50/2)
        line: LineString = LineString([(0,0), (0,50)])

        if (d_pt := self.topology.geometry.distance(pt)) > 1/2 : d_MST += d_pt
        if (d_line := self.topology.geometry.distance(line)) > 1/2 : d_MST += d_line
        
        self.response_disc = d_MST
        return self.response_disc
