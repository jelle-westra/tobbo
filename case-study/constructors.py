import numpy as np

from tobbo.core import Topology, ProblemInstance
from tobbo.problems import create_horizontal_cantilever_problem

# MMC problem constructors
from tobbo.parameterizations.mmc import Capsules, MMCCenterpointsConfig

def constructor_mmc(n_components: int) -> ProblemInstance:
    topology = Topology(False, (100, 50), 1.0)
    parameterization = Capsules(topology, False, True, n_components, MMCCenterpointsConfig, None, 1000)

    rxmin = topology.domain_size_x/parameterization.n_components/(2 if parameterization.symmetry_x else 1)/2
    rymin = topology.domain_size_y/parameterization.n_components/(2 if parameterization.symmetry_y else 1)/2
    parameterization.normalization_scale[-1] = min(rxmin, rymin)

    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_mmc_10D() : return constructor_mmc(2)
def constructor_mmc_20D() : return constructor_mmc(4) 
def constructor_mmc_50D() : return constructor_mmc(10)
def constructor_mmc_100D() : return constructor_mmc(20)
def constructor_mmc_200D() : return constructor_mmc(40)

mmc_constructors = {
    10 : constructor_mmc_10D,
    20 : constructor_mmc_20D,
    50 : constructor_mmc_50D,
    100 : constructor_mmc_100D,
    200 : constructor_mmc_200D,
}

# Curved MMC problem constructors
from tobbo.parameterizations.mmc import GuoBeam

def constructor_curved_mmc(n_components: int) -> ProblemInstance:
    topology = Topology(False, (100, 50), 1.0)
    parameterization = Capsules(topology, False, True, n_components, MMCCenterpointsConfig, GuoBeam(1000), 1000)

    rxmin = topology.domain_size_x/parameterization.n_components/(2 if parameterization.symmetry_x else 1)/2
    rymin = topology.domain_size_y/parameterization.n_components/(2 if parameterization.symmetry_y else 1)/2
    parameterization.normalization_scale[4] = min(rxmin, rymin)

    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_curved_mmc_10D() : return constructor_curved_mmc(1)
def constructor_curved_mmc_20D() : return constructor_curved_mmc(2) 
def constructor_curved_mmc_50D() : return constructor_curved_mmc(5)
def constructor_curved_mmc_100D() : return constructor_curved_mmc(10)
def constructor_curved_mmc_200D() : return constructor_curved_mmc(20)

curved_mmc_constructors = {
    10 : constructor_curved_mmc_10D,
    20 : constructor_curved_mmc_20D,
    50 : constructor_curved_mmc_50D,
    100 : constructor_curved_mmc_100D,
    200 : constructor_curved_mmc_200D,
}

# Honeycomb tiling
from tobbo.parameterizations.tiling import BinaryCells, HexGrid, unit_hexagon

def constructor_honeycomb(n_cells_y: int, cell_size_ratio_xy: float, dimension: int) -> ProblemInstance :
    topology = Topology(False, (100,50), 1.0)
    hex_cell_size_y = lambda n_cells_y : 2/np.sqrt(3)*(topology.domain_size_y/2)/(n_cells_y-1/2)
    parameterization = BinaryCells(
        topology,
        sampler=HexGrid(horizontal=False),
        unit_cell=unit_hexagon(),
        cell_size_x=cell_size_ratio_xy*hex_cell_size_y(n_cells_y),
        cell_size_y=hex_cell_size_y(n_cells_y),
        symmetry_x=False,
        symmetry_y=True
    )
    # dropping the cells on the bottom layer that stick out of the domain
    assert parameterization.dimension >= dimension, ''
    parameterization.cells = np.delete(parameterization.cells, [
        parameterization.dimension-2*(k+1)*(n_cells_y) for k in range(parameterization.dimension-dimension)
    ])
    parameterization._dimension = dimension
    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_honeycomb_10D() : return constructor_honeycomb(n_cells_y=2, cell_size_ratio_xy=1.5, dimension=10)
def constructor_honeycomb_20D() : return constructor_honeycomb(n_cells_y=3, cell_size_ratio_xy=1.8, dimension=20)
def constructor_honeycomb_50D() : return constructor_honeycomb(n_cells_y=4, cell_size_ratio_xy=1.3, dimension=50)
def constructor_honeycomb_100D() : return constructor_honeycomb(n_cells_y=5, cell_size_ratio_xy=1., dimension=100)
def constructor_honeycomb_200D() : return constructor_honeycomb(n_cells_y=7, cell_size_ratio_xy=1., dimension=200)

honeycomb_constructors = {
    10 : constructor_honeycomb_10D,
    20 : constructor_honeycomb_20D,
    50 : constructor_honeycomb_50D,
    100 : constructor_honeycomb_100D,
    200 : constructor_honeycomb_200D,
}