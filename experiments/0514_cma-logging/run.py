import sys
sys.path.append('../..')
from TO.utils import check_package_status

# check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment, OptimizationMethod
from TO.problems.cantilever import create_horizontal_cantilever_problem
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig

from contextlib import redirect_stdout

if __name__ == '__main__':
    topology = Topology(False, (100, 50), 1.0)
    parameterization = Capsules(topology, False, True, 1, MMCCenterpointsConfig, None, 1000)
    problem = create_horizontal_cantilever_problem(topology, parameterization)

    with open('cma.004.log', 'w') as fnull, redirect_stdout(fnull):
        run_experiment(problem, budget=500, seed=1, name='logging', method=OptimizationMethod.CMAES)