import numpy as np

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from enum import Enum
import os

from .problem import ProblemInstance
from .parameterization import Parameterization

BUDGET_PER_DIMENSION: int = 100

class OptimizationMethod(Enum):
    CMAES = 1
    HEBO = 2
    SMAC = 3

def run_experiment(
    problem: ProblemInstance,
    budget: int,
    seed: int,
    name: str,
    method: OptimizationMethod
) -> None:
    run_experiment_funcs = {
        OptimizationMethod.CMAES: run_experiment_CMAES,
        OptimizationMethod.HEBO: run_experiment_HEBO,
        OptimizationMethod.SMAC: run_experiment_SMAC,
    }
    problem.set_budget(budget)
    problem.logger_output_directory = f'./results/{name}/{seed}'
    os.makedirs(problem.logger_output_directory, exist_ok=True)

    try:
        run_experiment_funcs[method](problem, budget, seed, name)
    except KeyboardInterrupt:
        pass
    print('[stop]')
    
def run_experiment_CMAES(problem: ProblemInstance, budget: int, seed: int, name: str):
    import cma

    def gen_x0() -> np.ndarray:
        nonlocal seed, problem
        np.random.seed(seed)
        seed += 1000
        return problem.parameterization.gen_x0()

    assert (seed != 0), 'If the seed is 0, cma will generate a seed by itself which will make the experiment not reproducible.'
    opts: cma.CMAOptions = {'bounds': [0, 1], 'tolfun': 1e-6, 'seed': seed, 'verb_log': 0}
    cma.fmin2(problem, gen_x0, 0.25, restarts=3, bipop=True, options=opts)

def run_experiment_HEBO(problem: ProblemInstance, budget: int, seed: int, name: str) -> None :
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    from pymoo.config import Config
    Config.warnings['not_compiled'] = False

    dimension_specs = [{'name': str(i+1), 'type': 'num', 'lb' : 0, 'ub' : 1 } for i in range(problem.parameterization.dimension)]
    space = DesignSpace().parse(dimension_specs)
    opt = HEBO(space, scramble_seed=seed)

    it = 0
    y_best = float('inf')
    while (problem.budget-problem.count > 0):
        rec = opt.suggest(n_suggestions=1)
        opt.observe(rec, np.apply_along_axis(problem, axis=1, arr=rec.values))
        if (opt.y.min() < y_best) or not(it%100):
            print(f'{it:03d} [{problem.count}/{problem.budget}] {opt.y.min()}')
            if (opt.y.min() < y_best) : 
                y_best = opt.y.min()
        it += 1

def run_experiment_SMAC(problem: ProblemInstance, budget: int, seed: int, name: str) -> None :
    raise NotImplementedError()

def _run_instance(seed: int, problem_constructor: Callable[[], ProblemInstance], budget: int, name: str, method: OptimizationMethod):
    try:
        problem: ProblemInstance = problem_constructor()
        log_dir = f'./results/{name}/{seed}'
        os.makedirs(log_dir, exist_ok=True)

        with open(os.path.join(log_dir, 'run.log'), 'w') as fnull, redirect_stdout(fnull):
            run_experiment(problem, budget, seed, name, method)
            return (seed, 'success')

    except Exception as e:
        return (seed, f'error: {str(e)}')

def run_experiment_parallel(problem_constructor: Callable[[], ProblemInstance], budget_func: Callable[[Parameterization], int], name: str, method: OptimizationMethod, experiments: int, threads: int):
    problem: ProblemInstance = problem_constructor()
    budget = budget_func(problem.parameterization)
    print(f'Problem dimension: {problem.parameterization.dimension}')
    print(f'Budget: {budget}')
    del problem

    seeds = list(range(1, experiments + 1))
    args = [(seed, problem_constructor, budget, name, method) for seed in seeds]

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(_run_instance, *arg): arg[0] for arg in args}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                print(f'Seed {seed}: {result[1]}')
            except Exception as e:
                print(f'Seed {seed} crashed: {e}')