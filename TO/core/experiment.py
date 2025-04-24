from .problem import ProblemInstance

import numpy as np
import ioh
import cma

import os

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
        np.random.seed(seed)
        # the cma will update it's random seed by one after a restart, to 
        # keep unique runs we increment the x0 seed by 1000, assuming less than 1000 experiments are performed
        seed += 1000 
        print('generating new x0...')
        print(f'budget={problem.budget-problem.count}')
        return np.random.rand(problem.parameterization.dimension)

    assert (seed != 0), 'If the seed is 0, cma will generate a seed by itself which will make the experiment not reporducible.'
    opts:cma.CMAOptions = {'bounds':[0,1],'tolfun':1e-6,'seed':seed,'verb_filenameprefix':os.path.join(logger.output_directory,'outcmaes/')}

    problem.set_budget(budget)
    problem.attach_logger(logger)
    problem.logger_output_directory = logger.output_directory

    try: # assuming we exhaust the budget before 100 restarts
        cma.fmin2(problem, gen_x0, 0.25, restarts=3, bipop=True, options=opts)
    except KeyboardInterrupt :
        pass

    problem.reset()
    logger.close()