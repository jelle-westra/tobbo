import argparse

from tobbo.core import run_experiment, OptimizationMethod

from constructors import (
    mmc_constructors, curved_mmc_constructors, honeycomb_constructors
)

def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameterization', type=str)
    parser.add_argument('--dimension', type=int)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--budget_multiplier', default=20, type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    method = {
        'CMAES': OptimizationMethod.CMAES, 
        'HEBO': OptimizationMethod.HEBO, 
        'DE' : OptimizationMethod.DE,
    }[args.optimizer]
    problem = {
        'mmc': mmc_constructors,
        'curved-mmc': curved_mmc_constructors,
        'honeycomb': honeycomb_constructors,
    }[args.parameterization][args.dimension]()

    budget = args.budget_multiplier*args.dimension
    name = f'{args.dimension}D_{args.parameterization}_{args.optimizer}'

    run_experiment(problem, budget, args.seed, name, method)

if (__name__ == '__main__') :
    main()