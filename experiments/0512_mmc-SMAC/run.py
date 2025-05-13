import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment_parallel, OptimizationMethod
from TO.problems.cantilever import create_horizontal_cantilever_problem
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig

def construct_problem(n_components: int) -> ProblemInstance :
    topology = Topology(False, (100, 50), 1.0)
    parameterization = Capsules(topology, False, True, n_components, MMCCenterpointsConfig, None, 1_000)
    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_10D() : return construct_problem(2) # 10D
def constructor_20D() : return construct_problem(4) # 20D
def constructor_50D() : return construct_problem(10) # 50D
def constructor_100D() : return construct_problem(20) # 100D
def constructor_200D() : return construct_problem(40) # 200D

experiments = {
    '10D': constructor_10D,
    '20D': constructor_20D,
    '50D': constructor_50D,
    '100D': constructor_100D,
    '200D': constructor_200D,
}

if (__name__ == '__main__'):
    budget = lambda parameterization : 100*(parameterization.dimension)
    for (exp, constructor) in experiments.items():
        print('\n' + exp)
        run_experiment_parallel(constructor, budget, exp, experiments=30, method=OptimizationMethod.SMAC, threads=30)