import numpy as np
import cma

from collections.abc import Callable
from contextlib import redirect_stdout
import multiprocessing
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
    opts:cma.CMAOptions = {'bounds':[0,1], 'tolfun':1e-6, 'seed':seed, 'verb_log':0}

    problem.set_budget(budget)
    problem.logger_output_directory = f'./results/{name}/{seed}'
    if not(os.path.exists(problem.logger_output_directory)) : 
        os.makedirs(problem.logger_output_directory)

    try: # assuming we exhaust the budget before 100 restarts
        cma.fmin2(problem, gen_x0, 0.25, restarts=3, bipop=True, options=opts)
    except KeyboardInterrupt :
        pass

def _run_instance(args):
    (seed, problem_constructor, budget, name) = args
    try:
        problem: ProblemInstance = problem_constructor()

        problem.logger_output_directory = f'./results/{name}/{seed}'
        if not(os.path.exists(problem.logger_output_directory)) : 
            os.makedirs(problem.logger_output_directory)

        with open(os.path.join(problem.logger_output_directory, 'cma.log'), 'w') as fnull, redirect_stdout(fnull):
            return run_experiment(problem, budget, seed, name)
        
    except Exception as e:
        print(f'Exception in worker {seed}: {e}')
        return None


def run_experiment_parallel(problem_constructor: Callable[[],ProblemInstance], budget_func: Callable[[Parameterization], int], name: str, experiments: int, threads: int):
    problem: ProblemInstance = problem_constructor()
    budget = budget_func(problem.parameterization)
    print(f'Problem dimension: {problem.parameterization.dimension}')
    print(f'Budget: {budget}')

    seeds = list(range(1, experiments+1))
    args = [(seed, problem_constructor, budget, name) for seed in seeds]
    del problem
    
    with multiprocessing.Pool(processes=threads) as pool:
        for _ in pool.imap_unordered(_run_instance, args) : ...