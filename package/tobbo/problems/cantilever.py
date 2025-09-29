from tobbo.core import Parameterization, Topology, ProblemInstance, VolumeConstraint, DisconnectionConstraint, ConstraintMix
from tobbo.models.membrane import BinaryElasticMembraneModel, RigidEdge, Load

from shapely.geometry import Point, LineString

def create_horizontal_cantilever_problem(topology: Topology, parameterization: Parameterization) -> ProblemInstance :
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
            weight=1e3, 
            max_relative_volume=0.5
        ),
        DisconnectionConstraint(
            weight=1e3,
            topology=topology,
            boundaries=[
                Point(topology.domain_size_x, topology.domain_size_y/2), # the loading point
                LineString([(0,0), (0,topology.domain_size_y)]) # the wall
            ]
        )
    ]
    constraint_mixer = ConstraintMix(1, topology_constraints, offset=500)
    objective = lambda model : model.compute_element_compliance().sum()
    return ProblemInstance(topology, parameterization, model, [constraint_mixer], objective)