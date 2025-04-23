import argparse
from contextlib import redirect_stdout
import multiprocessing
from tqdm import tqdm
import os

import numpy as np
from shapely.geometry import box

from TO.core import Topology, ProblemInstance, run_experiment
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig
from TO.problems.cantilever import create_horizontal_cantilever_problem

MIN_PER_EVAL: float = 2/500 # approximate [min/evaluations]

budget: int = lambda d : 100*d

def setup_problem() -> ProblemInstance:
    topology = Topology(True, box(0, 0, 100, 50), 1.0)
    parameterization = Capsules(
        topology, False, True, 2, MMCCenterpointsConfig, None, 1_000
    )
    return create_horizontal_cantilever_problem(topology, parameterization)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int)
    parser.add_argument('--experiments', type=int, default=30)
    parser.add_argument('--components', type=int, default=2)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    def run(seed: int):
        problem = setup_problem(args.components)
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
            run_experiment(problem, budget(problem.parameterization.dimension), seed, 'multi-test')

    problem = setup_problem()
    print(f'Problem dimension: {problem.parameterization.dimension}')
    print(f'Budget: {budget(problem.parameterization.dimension)}')
    eta = MIN_PER_EVAL * args.experiments*budget(problem.parameterization.dimension) / args.threads
    print(f'ETA : {eta:.1f}min')
    del problem
    seeds = list(range(1, args.experiments+1))
    with multiprocessing.Pool(processes=args.threads) as pool:
        with tqdm(total=len(seeds)) as pbar:
            for _ in pool.imap_unordered(run, seeds):
                pbar.update(1)