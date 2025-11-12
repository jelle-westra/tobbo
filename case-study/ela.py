import os
import pickle
import argparse
from tqdm import tqdm

import numpy as np

from pflacco.sampling import create_initial_sample
from pflacco import classical_ela_features

from tobbo.core import OptimizationMethod
from constructors import (
    mmc_constructors, curved_mmc_constructors, honeycomb_constructors
)

def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameterization', type=str)
    parser.add_argument('--dimension', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--sample_multiplier', default=250, type=int)
    return parser.parse_args()

def main():
    feature_sets = (
        classical_ela_features.calculate_ela_distribution,
        classical_ela_features.calculate_ela_level,
        classical_ela_features.calculate_ela_meta,
        classical_ela_features.calculate_dispersion,
        classical_ela_features.calculate_nbc,
        classical_ela_features.calculate_pca,
        classical_ela_features.calculate_information_content,
    )

    args = parse_args()

    problem = {
        'mmc': mmc_constructors,
        'curved-mmc': curved_mmc_constructors,
        'honeycomb': honeycomb_constructors,
    }[args.parameterization][args.dimension]()

    samples = args.dimension * args.sample_multiplier
    name = f'{args.dimension}D_{args.parameterization}'


    path = f'./ela/{args.parameterization}/{args.dimension}D/multiplier-{args.sample_multiplier}/seed-{args.seed}/'
    os.makedirs(path, exist_ok=True)

    problem.logger_output_directory = f'./results/_ELA_{name}/{args.seed}'
    os.makedirs(problem.logger_output_directory, exist_ok=True)
    problem.set_budget(samples)

    print(f'[seed={args.seed}; samples={samples}]')
    X = create_initial_sample(
        args.dimension, 
        sample_coefficient=args.sample_multiplier, 
        sample_type='lhs', 
        seed=args.seed, 
        lower_bound=0.,
        upper_bound=1.
    )

    print('computing problem responses')
    y = np.zeros(len(X))
    for (i, x) in enumerate(tqdm(X.values)) :
        y[i] = problem(x)
    
    print('computing ela features')
    # computing and writing the ELA features
    X = (10*X - 5) # transforming [0,1] -> [-5, 5] to align with the BBOB functions
    features = {}
    for fs in tqdm(feature_sets) :
        features.update(fs(X, y))
    with open(os.path.join(path, f'seed-{args.seed}'), 'wb') as handle :
        pickle.dump(features, handle)

if (__name__ == '__main__'):
    main()