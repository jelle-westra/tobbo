import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from dataclasses import dataclass
from typing import Tuple
from itertools import product

from TO import FEM
from TO.topology import Topology

from TO.models._membrane_cython import fill_C_matrix as fill_C_matrix_c
# from TO.models._membrane_cython import fill_C_matrix_rigid_nodes as fill_C_matrix_rigid_nodes_c

class RigidEdge(FEM.LinearSystemBoundaryCondition) :
    def __init__(self, state: FEM.LinearSystem) :
        nodes_rigid = state.mesh.nodes[:,0]
        self.ii_nodes_rigid = (2*nodes_rigid + [[0], [1]]).flatten()

        s = set(self.ii_nodes_rigid)
        idx = [(i in s) or (j in s) for (i, j) in zip(state.ii_tril, state.jj_tril)]

        self.ii_tril = state.ii_tril[idx]
        self.jj_tril = state.jj_tril[idx]

    def apply(self, state: FEM.LinearSystem) -> None :
        state.C[self.ii_tril, self.jj_tril] = 0
        state.C[self.ii_nodes_rigid, self.ii_nodes_rigid] = 1.
        state.f[self.ii_nodes_rigid, 0]


class TipLoad(FEM.LinearSystemBoundaryCondition) :
    @staticmethod
    def apply(state: FEM.LinearSystem) -> None : 
        # selecting the middle node of last column in grid
        node = state.mesh.nodes[state.mesh.nodes.shape[0]//2,-1]
        # setting the y-component of `node`
        state.f[FEM.QuadElement.NODE_DOF * node + 1] = -0.1


class BinaryElasticMembraneModel():
    def __init__(self, 
        topology: Topology,
        thickness: float,
        E11: float,
        E22: float,
        G12: float,
        nu12: float,
        Emin: float
    ) -> None :
        self.mesh: FEM.StructuredQuadMesh = FEM.StructuredQuadMesh(topology.domain_size, topology.density)
        self.state: FEM.LinearSystem = FEM.LinearSystem(self.mesh)

        self.C_material = self.calculate_C(E11, E22, G12, nu12)
        self.C_void = Emin * (self.C_material != 0)

        self.Ke_material = thickness * self.mesh.integrate(self.C_material)
        self.Ke_void = thickness * self.mesh.integrate(self.C_void)

        self.topology = topology

        self.rigid_edge: FEM.LinearSystemBoundaryCondition = RigidEdge(self.state)

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

    def update(self, topology: Topology) :
        self.state.reset()
        fill_C_matrix_c(self.state.C, self.mesh.E, topology.mask.flatten(), self.Ke_material, self.Ke_void)
        self.rigid_edge.apply(self.state)
        TipLoad.apply(self.state)
        self.solve_displacement()
        self.topology = topology
        
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

        return (U.transpose(0,2,1) @ Ke @ U).reshape(self.mesh.elements.shape)
    
    def compute_tip_displacement(self) -> float : 
        raise NotImplemented('TODO')
    
    def plot(self, field: np.ndarray=None, fac: float=1, cmap: str='viridis', ax: Axes=None) -> Axes :
        if (field is None) : field = np.ones(self.mesh.elements.shape)
        assert (field.shape == self.mesh.elements.shape), ''
        if (ax is None) : ax = plt.gca()

        X_displaced = self.mesh.X + fac*self.state.u[::2].reshape(self.mesh.nodes.shape)
        Y_displaced = self.mesh.Y + fac*self.state.u[1::2].reshape(self.mesh.nodes.shape)

        rgba = plt.colormaps[cmap](field/field.max())
        rgba[...,-1] = self.topology.mask

        return ax.pcolormesh(X_displaced, Y_displaced, rgba, shading='flat')