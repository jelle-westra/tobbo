import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment_parallel, OptimizationMethod
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig
from TO.problems.cantilever import create_horizontal_cantilever_problem

def constructor_normalization_scale(n_components: int) -> ProblemInstance:
    topology = Topology(False, (100, 50), 1.0)
    parameterization = Capsules(topology, False, True, n_components, MMCCenterpointsConfig, None, 1000)

    rxmin = topology.domain_size_x/parameterization.n_components/(2 if parameterization.symmetry_x else 1)/2
    rymin = topology.domain_size_y/parameterization.n_components/(2 if parameterization.symmetry_y else 1)/2
    parameterization.normalization_scale[-1] = min(rxmin, rymin)

    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_10components() : return constructor_normalization_scale(10)
def constructor_20components() : return constructor_normalization_scale(20)
def constructor_40components() : return constructor_normalization_scale(40)

experiments = {
    '50D_normalization-scale': constructor_10components,
    '100D_normalization-scale': constructor_20components,
    '200D_normalization-scale': constructor_40components,
}

if (__name__ == '__main__'):
    budget = lambda parameterization : 100*(parameterization.dimension)
    for (exp, constructor) in experiments.items():
        print('\n' + exp)
        # print('SMAC')
        # run_experiment_parallel(constructor, budget, exp + '_SMAC', experiments=30, method=OptimizationMethod.SMAC, threads=6)
        print('CMA-ES')
        run_experiment_parallel(constructor, budget, exp + '_CMAES', experiments=15, method=OptimizationMethod.CMAES, threads=15)