import numpy as np
import cma

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import os

from .problem import ProblemInstance
from .parameterization import Parameterization

BUDGET_PER_DIMENSION: int = 100

def run_experiment(
    problem: ProblemInstance,
    budget: int,
    seed: int,
    name: str
) -> None:
    def gen_x0() -> np.ndarray:
        nonlocal seed, problem
        np.random.seed(seed)
        seed += 1000
        return np.random.rand(problem.parameterization.dimension)

    assert (seed != 0), 'If the seed is 0, cma will generate a seed by itself which will make the experiment not reproducible.'
    opts: cma.CMAOptions = {'bounds': [0, 1], 'tolfun': 1e-6, 'seed': seed, 'verb_log': 0}

    problem.set_budget(budget)
    problem.logger_output_directory = f'./results/{name}/{seed}'
    os.makedirs(problem.logger_output_directory, exist_ok=True)

    try:
        cma.fmin2(problem, gen_x0, 0.25, restarts=3, bipop=True, options=opts)
    except KeyboardInterrupt:
        pass
    print('[stop]')

def _run_instance(seed: int, problem_constructor: Callable[[], ProblemInstance], budget: int, name: str):
    try:
        problem: ProblemInstance = problem_constructor()
        log_dir = f'./results/{name}/{seed}'
        os.makedirs(log_dir, exist_ok=True)

        with open(os.path.join(log_dir, 'run.log'), 'w') as fnull, redirect_stdout(fnull):
            run_experiment(problem, budget, seed, name)
            return (seed, 'success')

    except Exception as e:
        return (seed, f'error: {str(e)}')

def run_experiment_parallel(problem_constructor: Callable[[], ProblemInstance], budget_func: Callable[[Parameterization], int], name: str, experiments: int, threads: int):
    problem: ProblemInstance = problem_constructor()
    budget = budget_func(problem.parameterization)
    print(f'Problem dimension: {problem.parameterization.dimension}')
    print(f'Budget: {budget}')
    del problem

    seeds = list(range(1, experiments + 1))
    args = [(seed, problem_constructor, budget, name) for seed in seeds]

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(_run_instance, *arg): arg[0] for arg in args}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                print(f'Seed {seed}: {result[1]}')
            except Exception as e:
                print(f'Seed {seed} crashed: {e}')