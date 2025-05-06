import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment_parallel
from TO.problems.cantilever import create_horizontal_cantilever_problem
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig

import numpy as np

def construct_problem(n_components: int, continuous: bool) -> ProblemInstance :
    topology = Topology(continuous, (100,50), 1.0)
    parameterization = Capsules(topology, False, True, n_components, MMCCenterpointsConfig, None, 1_000)
    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_5D_discrete () : return construct_problem(n_components=1, continuous=False)
def constructor_10D_discrete() : return construct_problem(n_components=2, continuous=False)
def constructor_15D_discrete() : return construct_problem(n_components=3, continuous=False)

def constructor_5D_continuous () : return construct_problem(n_components=1, continuous=False)
def constructor_10D_continuous() : return construct_problem(n_components=2, continuous=False)
def constructor_15D_continuous() : return construct_problem(n_components=3, continuous=False)

experiments = {
    '5D-discrete': constructor_5D_discrete,
    '10D-discrete': constructor_10D_discrete,
    '15D-discrete': constructor_15D_discrete,

    '5D-continuous': constructor_5D_continuous,
    '10D-continuous': constructor_10D_continuous,
    '15D-continuous': constructor_15D_continuous,
}

if (__name__ == '__main__'):
    budget = lambda parameterization : 100*(parameterization.dimension)
    print('checking the dimension of the problem constructors')
    for (exp, constructor) in experiments.items():
        dimension = int(exp.split('-')[0][:-1])
        assert (dimension == constructor().parameterization.dimension), ''
    print('[passed]')

    print('starting experiments')
    for (exp, constructor) in experiments.items():
        print('\n' + exp)
        run_experiment_parallel(constructor, budget, exp, experiments=30, threads=30)
