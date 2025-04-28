import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, VolumeConstraint, DisconnectionConstraint, ConstraintEnforcement, Constraint, run_experiment_parallel
from TO.models.membrane import BinaryElasticMembraneModel, RigidEdge, Load
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig

import numpy as np
from shapely.geometry import box, Point, LineString
from shapely.geometry.base import BaseGeometry
from scipy.sparse.csgraph import minimum_spanning_tree

from dataclasses import dataclass
from typing import List

@dataclass
class DisconnectionConstraintNoMST(Constraint):
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

        # ignoring the MST proceudre and consider all pairwise distances
        # D = minimum_spanning_tree(D).toarray()
        D[D < 1/2] = 0
        d = D.sum()

        for boundary in self.boundaries :
            if (distance_to_boundary := topology.geometry.distance(boundary)) > 1/2 : 
                d += distance_to_boundary
        return d

def construct_model(topology: Topology) -> BinaryElasticMembraneModel:
    model = BinaryElasticMembraneModel(
        topology, thickness=1, E11=25, E22=1, G12=0.5, nu12=0.25, Emin=1e-9
    )
    model.bcs.add(RigidEdge(
        nodes=model.mesh.nodes[:,0], state=model.state)
    )
    model.bcs.add(Load(
        nodes=[model.mesh.nodes[model.mesh.nely//2,-1]], loads=[(0, -0.1)])
    )
    return model

def construct_problem(weight_volume:float, weight_disconnection:float, use_MST:bool) -> ProblemInstance :
    topology = Topology(True, box(0, 0, 100, 50), 1.0)
    parameterization = Capsules(topology, False, True, 1, MMCCenterpointsConfig, None, 1_000)
    model = construct_model(topology)
    topology_constraints = [
        DisconnectionConstraint(
            weight=weight_disconnection, enforcement=ConstraintEnforcement.HARD, boundaries=[
                Point(topology.domain_size_x, topology.domain_size_y/2), # the loading point
                LineString([(0,0), (0,topology.domain_size_y)]) # the wall
            ]
        ) if (use_MST) else DisconnectionConstraintNoMST(
            weight=weight_disconnection, enforcement=ConstraintEnforcement.HARD, boundaries=[
                Point(topology.domain_size_x, topology.domain_size_y/2), # the loading point
                LineString([(0,0), (0,topology.domain_size_y)]) # the wall
            ]
        ),
        VolumeConstraint(
            weight=weight_volume, enforcement=ConstraintEnforcement.HARD, max_relative_volume=0.5
        ),
    ]
    objective = lambda model : model.compute_element_compliance().sum()
    return ProblemInstance(topology, parameterization, model, topology_constraints, objective)


def constructor_1e3_1e3_MST() : return construct_problem(1e3, 1e3, True)
def constructor_1e6_1e3_MST() : return construct_problem(1e6, 1e3, True)
def constructor_1e3_1e6_MST() : return construct_problem(1e3, 1e6, True)

def constructor_1e3_1e3_NoMST() : return construct_problem(1e3, 1e3, False)
def constructor_1e6_1e3_NoMST() : return construct_problem(1e6, 1e3, False)
def constructor_1e3_1e6_NoMST() : return construct_problem(1e3, 1e6, False)

experiments = {
    'reversed_volume=1e3_disconnection=1e3_MST': constructor_1e3_1e3_MST,
    'reversed_volume=1e6_disconnection=1e3_MST': constructor_1e6_1e3_MST,
    'reversed_volume=1e3_disconnection=1e6_MST': constructor_1e3_1e6_MST,
    
    'reversed_volume=1e3_disconnection=1e3_NoMST': constructor_1e3_1e3_NoMST,
    'reversed_volume=1e6_disconnection=1e3_NoMST': constructor_1e6_1e3_NoMST,
    'reversed_volume=1e3_disconnection=1e6_NoMST': constructor_1e3_1e6_NoMST,
}

if (__name__ == '__main__'):
    budget = lambda p : 500
    for (exp, constructor) in experiments.items():
        print('\n' + exp)
        run_experiment_parallel(constructor, budget, exp, experiments=30, threads=30)