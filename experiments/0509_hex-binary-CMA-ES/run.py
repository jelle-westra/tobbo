import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment_parallel
from TO.problems.cantilever import create_horizontal_cantilever_problem
from TO.parameterizations.tiling import BinaryCells, HexGrid, unit_hexagon

import numpy as np

def construct_problem(n_cells_y: int, cell_size_ratio_xy: float, dimension: int) -> ProblemInstance :
    topology = Topology(True, (100,50), 1.0)
    hex_cell_size_y = lambda n_cells_y : 2/np.sqrt(3)*(topology.domain_size_y/2)/(n_cells_y-1/2)
    parameterization = BinaryCells(
        topology,
        sampler=HexGrid(horizontal=False),
        unit_cell=unit_hexagon(),
        cell_size_x=cell_size_ratio_xy*hex_cell_size_y(n_cells_y),
        cell_size_y=hex_cell_size_y(n_cells_y),
        symmetry_x=False,
        symmetry_y=True
    )
    # dropping the cells on the bottom layer that stick out of the domain
    assert parameterization.dimension >= dimension, ''
    parameterization.cells = np.delete(parameterization.cells, [
        parameterization.dimension-2*(k+1)*(n_cells_y) for k in range(parameterization.dimension-dimension)
    ])
    parameterization._dimension = dimension
    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_10D() : return construct_problem(n_cells_y=2, cell_size_ratio_xy=1.5, dimension=10)
def constructor_20D() : return construct_problem(n_cells_y=3, cell_size_ratio_xy=1.8, dimension=20)
def constructor_50D() : return construct_problem(n_cells_y=4, cell_size_ratio_xy=1.3, dimension=50)
def constructor_100D() : return construct_problem(n_cells_y=5, cell_size_ratio_xy=1., dimension=100)
def constructor_200D() : return construct_problem(n_cells_y=7, cell_size_ratio_xy=1., dimension=200)

experiments = {
    '10D': constructor_10D,
    '20D': constructor_20D,
    '50D': constructor_50D,
    '100D': constructor_100D,
    '200D': constructor_200D,
}

if (__name__ == '__main__'):
    budget = lambda parameterization : 100*(parameterization.dimension)
    print('checking the dimension of the problem constructors')
    for (exp, constructor) in experiments.items():
        dimension = int(exp[:-1])
        assert (dimension == constructor().parameterization.dimension), ''
    print('[passed]')

    print('starting experiments')
    for (exp, constructor) in experiments.items():
        print('\n' + exp)
        run_experiment_parallel(constructor, budget, exp, experiments=30, threads=30)