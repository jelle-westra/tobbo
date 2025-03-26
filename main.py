from problem import ProblemTO, Topology
from parameterizations.lame_curve import LameCurves
from parameterizations.rectangle import Rectangles
from parameterizations.capsule import Capsules
from simulation.FEA import BinaryElasticMembraneModel

from shapely.geometry import box

import numpy as np

import ioh
import os

import cma

import argparse

def main() -> None :
    args = parse_args()

    topology = Topology(continuous=True, domain=box(0,0,100,50))
    
    # parameterization = LameCurves(symmetry=True, m=8)
    # parameterization = Rectangles(symmetry=True)
    parameterization = Capsules(symmetry=True)

    # add topology.domain and topology.element_size or density something.
    model = BinaryElasticMembraneModel((100, 50), (1, 1), 1, 25, 1, 0.5, 0.25, 1e-9)

    ioh_prob = ProblemTO(topology, parameterization, model)

    triggers = [
        ioh.logger.trigger.Each(1),
        ioh.logger.trigger.OnImprovement()
    ]

    logger = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name=f"./results/{args.run}",       # in a folder named: './Figures_Python/Run_{run_e}'
        algorithm_name="CMA-ES",    # meta-data for the algorithm used to generate these results
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

    x0 = np.random.rand(parameterization.dimension)

    RANDOM_SEED = 98894
    opts:cma.CMAOptions = {'bounds':[0,1],'tolfun':1e-6,'seed':RANDOM_SEED,'verb_filenameprefix':os.path.join(logger.output_directory,"outcmaes/")}

    ioh_prob.attach_logger(logger)
    ioh_prob.logger_output_directory = logger.output_directory

    cma.fmin2(ioh_prob, x0, args.sigma0, restarts=0, bipop=True, options=opts)

    ioh_prob.reset()
    logger.close()

def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=int)
    parser.add_argument('--sigma0', type=float, default=0.25)
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--discrete', action='store_true', default=False)
    return parser.parse_args()

if (__name__ == '__main__') : main()