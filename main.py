import numpy as np
import cma

import ioh
import os

from TO.core import ProblemInstance

def run_experiment(
    problem: ProblemInstance,
    budget: int,
    seed: int,
    name: str
) -> None :
    triggers = [
        ioh.logger.trigger.OnImprovement()
    ]

    logger = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name=f'./results/{name}/{seed}',       # in a folder named: './Figures_Python/Run_{run_e}'
        algorithm_name='CMA-ES',    # meta-data for the algorithm used to generate these results
        store_positions=True,               # store x-variables in the logged files
        triggers= triggers,

        additional_properties=[
            ioh.logger.property.CURRENTY,   # The constrained y-value, by default only the untransformed & unconstraint y
                                            # value is logged. 
            ioh.logger.property.RAWYBEST, # Store the raw-best
            ioh.logger.property.CURRENTBESTY, # Store the current best given the constraints
            ioh.logger.property.VIOLATION,  # The violation value
            ioh.logger.property.PENALTY,     # The applied penalty
        ]
    )

    def gen_x0() -> np.ndarray :
        nonlocal seed, problem
        np.random.seed(seed := seed + 1)
        print('generating new x0...')
        print(f'budget={problem.budget-problem.count}')
        return np.random.rand(problem.parameterization.dimension)

    assert (seed != 0), 'If the seed is 0, cma will generate a seed by itself which will make the experiment not reporducible.'
    opts:cma.CMAOptions = {'bounds':[0,1],'tolfun':1e-6,'seed':seed,'verb_filenameprefix':os.path.join(logger.output_directory,'outcmaes/')}

    problem.set_budget(budget)
    problem.attach_logger(logger)
    problem.logger_output_directory = logger.output_directory
    
    try:
        cma.fmin2(problem, gen_x0, sigma0=0.25, restarts=1, bipop=True, options=opts)
    except KeyboardInterrupt :
        pass

    problem.reset()
    logger.close()

from TO.parameterizations.tiling import BinaryCells, HexGrid, unit_hexagon
from TO.core import Topology
from TO.problems.cantilever import create_horizontal_cantilever_problem

from shapely.geometry import box

topology = Topology(True, box(0, 0, 100, 50), 1.0)

n_cells_y=3
parameterization = BinaryCells(
    topology,
    sampler=HexGrid(horizontal=False),
    unit_cell=unit_hexagon(),
    cell_size_x=2/np.sqrt(3)*(topology.domain_size_y/2)/(n_cells_y-1/2),
    cell_size_y=2/np.sqrt(3)*(topology.domain_size_y/2)/(n_cells_y-1/2),
    symmetry_x=False,
    symmetry_y=True
)

problem = create_horizontal_cantilever_problem(topology, parameterization)
budget = 100*problem.parameterization.dimension
run_experiment(problem, budget, 1, 'dummy')