import numpy as np
import ioh
import cma

from collections.abc import Callable
from contextlib import redirect_stdout
import multiprocessing
from threading import Thread
from tqdm import tqdm
from typing import List
from time import sleep
import os

from .problem import ProblemInstance
from .parameterization import Parameterization

BUDGET_PER_DIMENSION: int = 100

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
        cma.fmin2(problem, gen_x0, 0.25, restarts=100, bipop=True, options=opts)
    except KeyboardInterrupt :
        pass

    problem.reset()
    logger.close()

def _run_instance(args):
    (seed, problem_constructor, budget, name) = args
    problem: ProblemInstance = problem_constructor()
    with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
        run_experiment(problem, budget, seed, name)

def _run_progress_checker(seeds: List[int], budget: int, name: str):
    path = lambda seed : f'results/{name}/{seed}/evals.dat'
    n = len(seeds)
    relative_progress = [0.]*n
    print('\n'*n)
    while any(p < 1. for p in relative_progress):
        print('\033[F' * n, end='')  # moving cursor up
        for (i, seed) in enumerate(seeds):
            if not(relative_progress[i] == 1) and os.path.exists(path(seed)) :
                with open(path(seed), 'rb') as f:
                    relative_progress[i] = sum(1 for _ in f)/budget

            pbar_length: int = 50
            fill_length = int(relative_progress[i]*pbar_length)
            pbar = '='*fill_length + (pbar_length-fill_length)*' '
            print(f'[{i+1:02d}/{n:02d}] [' + pbar + f'] {relative_progress[i]*100:.1f}%')
        sleep(1)


def run_experiment_parallel(problem_constructor: Callable[[],ProblemInstance], budget_func: Callable[[Parameterization], int], name: str, experiments: int, threads: int):
    problem: ProblemInstance = problem_constructor()
    budget = budget_func(problem.parameterization)
    print(f'Problem dimension: {problem.parameterization.dimension}')
    print(f'Budget: {budget}')

    seeds = list(range(1, experiments+1))
    args = [(seed, problem_constructor, budget, name) for seed in seeds]
    del problem

    progress_thread = Thread(target=_run_progress_checker, args=(seeds, budget, name), daemon=True)
    progress_thread.start()
    
    with multiprocessing.Pool(processes=threads) as pool:
        for _ in pool.imap_unordered(_run_instance, args) : ...