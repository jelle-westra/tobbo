import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment_parallel
from TO.problems.cantilever import create_horizontal_cantilever_problem
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig, GuoBeam

import numpy as np
from shapely.geometry import box, Point, LineString
from shapely.geometry.base import BaseGeometry
from scipy.sparse.csgraph import minimum_spanning_tree

from dataclasses import dataclass
from typing import List

def construct_problem(n_components: int) -> ProblemInstance :
    topology = Topology(True, box(0,0,100,50), 1.0)
    parameterization = Capsules(topology, False, True, n_components, MMCCenterpointsConfig, GuoBeam(1_000), 1_000)
    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_1mmcs() : return construct_problem(1) # 10D
def constructor_2mmcs() : return construct_problem(2) # 20D
def constructor_5mmcs() : return construct_problem(5) # 50D
def constructor_10mmcs() : return construct_problem(10) # 100D
def constructor_20mmcs() : return construct_problem(20) # 200D

experiments = {
    '1mmc': constructor_1mmcs,
    '2mmc': constructor_2mmcs,
    '5mmc': constructor_5mmcs,
    '10mmc': constructor_10mmcs,
    '20mmc': constructor_20mmcs,
}

if (__name__ == '__main__'):
    budget = lambda parameterization : 100*(parameterization.dimension)
    for (exp, constructor) in experiments.items():
        print('\n' + exp)
        run_experiment_parallel(constructor, budget, exp, experiments=30, threads=30)