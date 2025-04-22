from .topology import Topology
from .parameterization import Parameterization
from .models import BinaryElasticMembraneModel
from .problem import ProblemInstance

import numpy as np
import ioh
import cma

import os

def run_experiment(
    problem: ProblemInstance,
    budget: int,
    sigma0: float,
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

    np.random.seed(seed)
    x0 = np.random.rand(problem.parameterization.dimension)

    assert (seed != 0), 'If the seed is 0, cma will generate a seed by itself which will make the experiment not reporducible.'
    opts:cma.CMAOptions = {'bounds':[0,1],'tolfun':1e-6,'seed':seed,'verb_filenameprefix':os.path.join(logger.output_directory,'outcmaes/')}

    problem.set_budget(budget)
    problem.attach_logger(logger)
    problem.logger_output_directory = logger.output_directory

    try:
        cma.fmin2(problem, x0, sigma0, restarts=0, bipop=True, options=opts)
    except KeyboardInterrupt :
        pass

    problem.reset()
    logger.close()