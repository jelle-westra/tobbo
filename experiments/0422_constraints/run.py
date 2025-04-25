import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, VolumeConstraint, DisconnectionConstraint, ConstraintEnforcement, run_experiment_parallel
from TO.models.membrane import BinaryElasticMembraneModel, RigidEdge, Load
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig
from TO.problems.cantilever import create_horizontal_cantilever_problem

from shapely.geometry import box, Point, LineString

def construct_problem() -> ProblemInstance :
    topology = Topology(True, box(0, 0, 100, 50), 1.0)
    parameterization = Capsules(topology, False, True, 1, MMCCenterpointsConfig, None, 1_000)

    model = BinaryElasticMembraneModel(
        topology, thickness=1, E11=25, E22=1, G12=0.5, nu12=0.25, Emin=1e-9
    )
    model.bcs.add(RigidEdge(
        nodes=model.mesh.nodes[:,0], state=model.state)
    )
    model.bcs.add(Load(
        nodes=[model.mesh.nodes[model.mesh.nely//2,-1]], loads=[(0, -0.1)])
    )
    topology_constraints = [
        VolumeConstraint(
            weight=1e9, enforcement=ConstraintEnforcement.HARD, max_relative_volume=0.5
        ),
        DisconnectionConstraint(
            weight=1e3, enforcement=ConstraintEnforcement.HARD, boundaries=[
                Point(topology.domain_size_x, topology.domain_size_y/2), # the loading point
                LineString([(0,0), (0,topology.domain_size_y)]) # the wall
            ]
        )
    ]
    objective = lambda model : model.compute_element_compliance().sum()
    return ProblemInstance(topology, parameterization, model, topology_constraints, objective)

if (__name__ == '__main__'):
    budget = lambda p : 100
    run_experiment_parallel(construct_problem, budget, 'dummy', experiments=30, threads=30)