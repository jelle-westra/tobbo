import multiprocessing
from tqdm import tqdm
import os
from contextlib import redirect_stdout

import numpy as np
from shapely.geometry import box


from TO.core import Topology, ProblemInstance, run_experiment
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig
from TO.problems.cantilever import create_horizontal_cantilever_problem

NO_THREADS: int = 15
NO_EXPERIMENTS: int = 30
MIN_PER_EVAL: float = 2/500 # approximate [min/evaluations]

budget: int = lambda d : 100*d

def setup_problem() -> ProblemInstance:
    topology = Topology(True, box(0, 0, 100, 50), 1.0)
    parameterization = Capsules(
        topology, False, True, 2, MMCCenterpointsConfig, None, 1_000
    )
    return create_horizontal_cantilever_problem(topology, parameterization)

def run(seed: int):
    problem = setup_problem()
    with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
        run_experiment(problem, budget(problem.parameterization.dimension), seed, 'multi-test')

if __name__ == '__main__':
    problem = setup_problem()
    print(f'Problem dimension: {problem.parameterization.dimension}')
    print(f'Budget: {budget(problem.parameterization.dimension)}')
    eta = MIN_PER_EVAL * NO_EXPERIMENTS*budget(problem.parameterization.dimension) / NO_THREADS
    print(f'ETA : {eta:.1f}min')
    del problem
    seeds = list(range(1, NO_EXPERIMENTS+1))
    with multiprocessing.Pool(processes=NO_THREADS) as pool:
        with tqdm(total=len(seeds)) as pbar:
            for _ in pool.imap_unordered(run, seeds):
                pbar.update(1)