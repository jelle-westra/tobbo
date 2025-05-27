import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment_parallel, OptimizationMethod
from TO.parameterizations.mmc import Capsules
from TO.parameterizations.mesh import Voronoi
from TO.problems.cantilever import create_horizontal_cantilever_problem

def constructor_delaunay(n_points: int) -> ProblemInstance:
    topology = Topology(False, (100,50), 1.0)
    parameterization = Voronoi(topology, False, True, n_points, Capsules)

    # n_components is can be different for different configs, but this is a approximation.
    n_components = n_points

    rxmin = topology.domain_size_x/n_components/(2 if parameterization.mmc.symmetry_x else 1)/2
    rymin = topology.domain_size_y/n_components/(2 if parameterization.mmc.symmetry_y else 1)/2
    # need to cap the rmin bc otherwise we have so many components they become to thin
    parameterization.mmc.normalization_scale[-1] = max(min(rxmin, rymin), 0.5)

    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_10D() : return constructor_delaunay(5)
def constructor_20D() : return constructor_delaunay(10)
def constructor_50D() : return constructor_delaunay(25)
def constructor_100D() : return constructor_delaunay(50)
def constructor_200D() : return constructor_delaunay(100)

experiments = {
    '10D_normalization-scale': constructor_10D,
    '20D_normalization-scale': constructor_20D,
    '50D_normalization-scale': constructor_50D,
    '100D_normalization-scale': constructor_100D,
    '200D_normalization-scale': constructor_200D,
}

if (__name__ == '__main__'):
    budget = lambda parameterization : 100*(parameterization.dimension)
    for (exp, constructor) in experiments.items():
        print('\n' + exp)
        # print('SMAC')
        # run_experiment_parallel(constructor, budget, exp + '_SMAC', experiments=30, method=OptimizationMethod.SMAC, threads=6)
        print('CMA-ES')
        run_experiment_parallel(constructor, budget, exp + '_CMAES', experiments=15, method=OptimizationMethod.CMAES, threads=15)