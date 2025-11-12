from tobbo.core import ProblemInstance
import numpy as np
from tqdm import trange

from constructors import mmc_constructors, curved_mmc_constructors, honeycomb_constructors
import os
import argparse


def main() -> None:
    args = parse_args()

    problem: ProblemInstance = {
        'mmc': mmc_constructors,
        'curved-mmc': curved_mmc_constructors,
        'honeycomb': honeycomb_constructors,
    }[args.parameterization][args.dimension]()

    # everything below the offset is feasible
    constraint_offset: float = problem.topology_constraints[0].offset

    problem.logger_output_directory = f'./results/feasibility-ratio/{args.parameterization}/{args.dimension}'
    os.makedirs(problem.logger_output_directory, exist_ok=True)
    problem.set_budget(args.samples)

    # logging into an additional file to keep track of the feasibility ratio
    log_path = os.path.join(problem.logger_output_directory, 'feasibility-ratio.dat')
    with open(log_path, 'a') as handle : 
        handle.write(f'samples, feasibility_ratio\n')

    (n, cnt) = (0, 0)
    pbar = trange(args.samples//args.batch_size)
    for batch_idx in pbar:
        np.random.seed(batch_idx)
        X = np.random.rand(args.batch_size, args.dimension)
        y = np.apply_along_axis(problem, 1, X)

        cnt += (y < constraint_offset).sum()
        n += args.batch_size

        pbar.set_postfix({'feasibility_ratio': cnt/n, 'parameterization': args.parameterization, 'dimension': args.dimension})
        with open(log_path, 'a') as handle : 
            handle.write(f'{n}, {cnt/n:f}\n')


def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameterization', type=str)
    parser.add_argument('--dimension', type=int)
    parser.add_argument('--samples', type=int, default=1_000_000)
    parser.add_argument('--batch-size', type=int, default=10_000)
    return parser.parse_args()


if (__name__ == '__main__') : main()