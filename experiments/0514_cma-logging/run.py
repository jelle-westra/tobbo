import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment, OptimizationMethod
from TO.problems.cantilever import create_horizontal_cantilever_problem
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig

if __name__ == '__main__':
    topology = Topology(False, (100, 50), 1.0)
    parameterization = Capsules(topology, False, True, 1, MMCCenterpointsConfig, None, 1000)
    problem = create_horizontal_cantilever_problem(topology, parameterization)
    run_experiment(problem, budget=30, seed=1, method=OptimizationMethod.CMAES)