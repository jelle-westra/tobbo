import numpy as np

from dataclasses import dataclass
from typing import Tuple

import simulation.FEM as FEM
# from problem import Topology

from shapely.geometry import MultiPolygon, Polygon
@dataclass
class Topology:
    continuous: bool
    domain: Polygon
    geometry: MultiPolygon = None
    mask: np.ndarray = None


class RigidEdge(FEM.LinearSystemBoundaryCondition) :
    @staticmethod
    def apply(state: FEM.LinearSystem) -> None : 
        # selecting the first column of the grid
        nodes = state.mesh.nodes[:,0]
        # setting all node components to 0
        for i in range(FEM.QuadElement.NODE_DOF) : 
            state.C[FEM.QuadElement.NODE_DOF * nodes + i, :] = 0.0
            state.C[:, FEM.QuadElement.NODE_DOF * nodes + i] = 0.0
            state.C[FEM.QuadElement.NODE_DOF * nodes + i, FEM.QuadElement.NODE_DOF * nodes + i] = 1.0
            state.f[FEM.QuadElement.NODE_DOF * nodes + i, 0] = 0.0


class TipLoad(FEM.LinearSystemBoundaryCondition) :
    @staticmethod
    def apply(state: FEM.LinearSystem) -> None : 
        # selecting the middle node of last column in grid
        node = state.mesh.nodes[state.mesh.nodes.shape[0]//2,-1]
        # setting the y-component of `node`
        state.f[FEM.QuadElement.NODE_DOF * node + 1] = -0.1


class BinaryElasticMembraneModel():
    def __init__(self, 
        domain_size: Tuple[float, float], 
        element_size: Tuple[float, float],
        thickness: float,
        E11: float,
        E22: float,
        G12: float,
        nu12: float,
        Emin: float
    ) -> None :
        (x_domain, y_domain) = domain_size
        self.thickness = thickness

        self.mesh: FEM.StructuredQuadMesh = FEM.StructuredQuadMesh(domain_size, element_size)
        self.state: FEM.LinearSystem = FEM.LinearSystem(self.mesh)

        self.C_material = self.calculate_C(E11, E22, G12, nu12)
        self.C_void = Emin * (self.C_material != 0)

        self.Ke_material = self.thickness * self.mesh.integrate(self.C_material)
        self.Ke_void = self.thickness * self.mesh.integrate(self.C_void)

        self.topology = Topology(True, None, np.zeros(self.mesh.elements.shape, dtype=bool))

    @staticmethod
    def calculate_C(E11: float, E22: float, G12: float, nu12: float) -> np.ndarray:
        nu21 = (E22 * nu12) / E11
        denom = (1 - nu12 * nu21)

        (q11, q22, q12) = (E11/denom, E22/denom, nu12*E22/denom)
        q66 = G12

        u1 = (3*(q11 + q22) + 2*q12 + 4*q66)
        u4 = (q11 + q22 + 6*q12 - 4*q66)
        u5 = (q11 + q22 - 2*q12 + 4*q66)

        return (1/8)*np.array([[u1, u4, 0], [u4, u1, 0], [0, 0, u5]])
    
    @staticmethod
    def fill_C_matrix(
        C: np.ndarray,
        E: np.ndarray,
        mask: np.ndarray,
        Ke_material: np.ndarray,
        Ke_void: np.ndarray
    ):
        for elem_pos in range(len(mask)):
            for ii in range(FEM.QuadElement.ELEMENT_NO_NODES):
                # getting the vertex index in mesh (mesh vertex index)
                iN = int(E[elem_pos, ii])

                for jj in range(FEM.QuadElement.ELEMENT_NO_NODES):
                    # getting the vertex index in mesh (mesh vertex index)
                    jN = int(E[elem_pos, jj])
                    if (iN < jN - 1) : break

                    KeNNDOF = (Ke_material if mask[elem_pos] else Ke_void)[(ii) * FEM.QuadElement.NODE_DOF:(ii + 1) * FEM.QuadElement.NODE_DOF, (jj) * FEM.QuadElement.NODE_DOF:(jj + 1) * FEM.QuadElement.NODE_DOF]

                    C[(iN) * FEM.QuadElement.NODE_DOF + 0, (jN) * FEM.QuadElement.NODE_DOF + 0] += KeNNDOF[0, 0]
                    C[(iN) * FEM.QuadElement.NODE_DOF + 0, (jN) * FEM.QuadElement.NODE_DOF + 1] += KeNNDOF[0, 1]
                    C[(iN) * FEM.QuadElement.NODE_DOF + 1, (jN) * FEM.QuadElement.NODE_DOF + 0] += KeNNDOF[1, 0]
                    C[(iN) * FEM.QuadElement.NODE_DOF + 1, (jN) * FEM.QuadElement.NODE_DOF + 1] += KeNNDOF[1, 1]

    def update(self, topology: Topology) -> bool :
        self.reset_model()
        self.fill_C_matrix(self.state.C, self.mesh.E, topology.mask.flatten(), self.Ke_material, self.Ke_void)
        self.state.add_boundary_condition(RigidEdge)
        if (TipLoad not in self.state.bcs) : 
            self.state.add_boundary_condition(TipLoad)
        self.solve_displacement()
        self.topology = topology

    def reset_model(self) -> None :
        self.state.C[:] = self.state.u[:] = 0
        # since we clear the C matrix we have to impose the RigidEdge again after recalculating `C`
        if (RigidEdge in self.state.bcs) : self.state.bcs.remove(RigidEdge)
        
    def solve_displacement(self) -> None : 
        self.state.solve_primary_field()

    def compute_element_compliance(self) -> np.ndarray :
        DOF_arr_base = np.array(list(range(FEM.QuadElement.NODE_DOF))*FEM.QuadElement.ELEMENT_NO_NODES).reshape(
            FEM.QuadElement.ELEMENT_NO_NODES, FEM.QuadElement.NODE_DOF
        )

        Ke = np.tile(self.Ke_void, (self.mesh.elements.size, 1, 1))
        Ke[self.topology.mask.flatten()] = self.Ke_material

        DOF_arr = (2*self.mesh.E[:,:,np.newaxis] + DOF_arr_base).reshape(-1,DOF_arr_base.size)
        U = self.state.u[DOF_arr].reshape(-1, FEM.QuadElement.ELEMENT_DOF, 1)

        return (U.transpose(0,2,1) @ Ke @ U).reshape(-1,1)
    
    def compute_tip_displacement(self) -> float : ...