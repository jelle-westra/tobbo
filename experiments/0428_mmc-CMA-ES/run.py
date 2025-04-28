import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment_parallel
from TO.problems.cantilever import create_horizontal_cantilever_problem
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig

import numpy as np
from shapely.geometry import box, Point, LineString
from shapely.geometry.base import BaseGeometry
from scipy.sparse.csgraph import minimum_spanning_tree

from dataclasses import dataclass
from typing import List

def construct_problem(n_components: int) -> ProblemInstance :
    topology = Topology(True, box(0,0,100,50), 1.0)
    parameterization = Capsules(topology, False, True, n_components, MMCCenterpointsConfig, None, 1_000)
    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_2mmcs() : return construct_problem(2) # 10D
def constructor_4mmcs() : return construct_problem(4) # 20D
def constructor_10mmcs() : return construct_problem(10) # 50D
def constructor_20mmcs() : return construct_problem(20) # 100D
def constructor_40mmcs() : return construct_problem(40) # 200D

experiments = {
    '2mmc': constructor_2mmcs,
    '4mmc': constructor_4mmcs,
    '10mmc': constructor_10mmcs,
    '20mmc': constructor_20mmcs,
    '40mmc': constructor_40mmcs,
}

if (__name__ == '__main__'):
    budget = lambda parameterization : 100*(parameterization.dimension)
    for (exp, constructor) in experiments.items():
        print('\n' + exp)
        run_experiment_parallel(constructor, budget, exp, experiments=30, threads=30)