import numpy as np
from scipy import sparse

from abc import ABC, abstractmethod
from itertools import product
from typing import List, Set, Tuple


class QuadElement:
    # a 2D quad element
    NODE_DOF: int = 2
    ELEMENT_NO_NODES: int = 4
    ELEMENT_DOF: int = ELEMENT_NO_NODES*NODE_DOF

    GQ_POINT_4 :np.ndarray = np.array([[-1,-1],[1,-1],[-1,1],[1,1]], dtype=float) / np.sqrt(3.0)
    GQ_WEIGHT_4:np.ndarray = np.ones(4, dtype=float)

    def __init__(self, verts: np.ndarray) -> None :
        assert verts.shape == (self.ELEMENT_NO_NODES, self.NODE_DOF)
        
        # local (u,v) 4-point Gaussian Quadrature
        self.S4:np.ndarray = QuadElement.shape(*QuadElement.GQ_POINT_4.T)
        self.S4_grad:np.ndarray = QuadElement.shape_grad(*QuadElement.GQ_POINT_4.T)

        self.S4_grad_xy:np.ndarray = np.zeros((1,2,4,4))
        # we'll need this during GQ integration.
        self.detJacob4:np.ndarray = np.zeros((1,4))

        for gqp in range(4):
            jacob:np.ndarray = self.S4_grad[:,:,gqp] @ verts
            invJacob:np.ndarray = np.linalg.inv(jacob)
            self.detJacob4[0,gqp] = np.linalg.det(jacob) 
            self.S4_grad_xy[0,:,:,gqp] = invJacob @ self.S4_grad[:,:,gqp]

        B_gqp = lambda gqp : np.array([
            [self.S4_grad_xy[0,0,0,gqp], 0, self.S4_grad_xy[0,0,1,gqp], 0, self.S4_grad_xy[0,0,2,gqp], 0, self.S4_grad_xy[0,0,3,gqp], 0 ],
            [0, self.S4_grad_xy[0,1,0,gqp], 0, self.S4_grad_xy[0,1,1,gqp], 0, self.S4_grad_xy[0,1,2,gqp], 0, self.S4_grad_xy[0,1,3,gqp]],
            [
                self.S4_grad_xy[0,1,0,gqp], self.S4_grad_xy[0,0,0,gqp], self.S4_grad_xy[0,1,1,gqp], self.S4_grad_xy[0,0,1,gqp], 
                self.S4_grad_xy[0,1,2,gqp], self.S4_grad_xy[0,0,2,gqp], self.S4_grad_xy[0,1,3,gqp], self.S4_grad_xy[0,0,3,gqp]
            ]
        ])

        self.B_gen = np.stack([B_gqp(gqp) for gqp in range(4)], axis=-1)


    @staticmethod
    def shape(xi: float, eta: float) -> Tuple[float, float, float, float] :
        return (1/4)*np.stack([(1.0-xi)*(1.0-eta), (1.0+xi)*(1.0-eta), (1.0+xi)*(1.0+eta), (1.0-xi)*(1.0+eta)])

    @staticmethod
    def shape_grad(xi: float, eta: float) -> Tuple :
        return (1/4) * np.stack((
            [-(1.0-eta),  (1.0-eta), (1.0+eta), -(1.0+eta)],
            [-(1.0-xi) , -(1.0+xi) , (1.0+xi) ,  (1.0-xi) ]
        ))

    def integrate(self, C_material: np.ndarray) -> np.ndarray :
        assert (C_material.shape == (3,3)), ''

        Ke = np.zeros((
            self.ELEMENT_NO_NODES*self.NODE_DOF,
            self.ELEMENT_NO_NODES*self.NODE_DOF
        ), dtype=np.float64)

        for gqp in range(self.ELEMENT_NO_NODES):
            # elemental membrane stiffness matrix
            B:np.ndarray = self.B_gen[:,:,gqp]
            Ke += self.detJacob4[0,gqp]*self.GQ_WEIGHT_4[gqp] * (B.transpose() @ C_material @ B)

        return Ke


class QuadMesh(ABC):
    E: List[Tuple[float, float, float, float]]
    quads: List[QuadElement]

    def generate_sparse_pattern(self) -> None :
        ij = set()
        for elem_pos in range(self.elements.size):
            for ii in range(QuadElement.ELEMENT_NO_NODES):
                # getting the vertex index in mesh (mesh vertex index)
                iN = int(self.E[elem_pos, ii])

                for jj in range(QuadElement.ELEMENT_NO_NODES):
                    # getting the vertex index in mesh (mesh vertex index)
                    jN = int(self.E[elem_pos, jj])
                    if (iN < jN - 1) : break
                    
                    for (i, j) in product(range(QuadElement.NODE_DOF), range(QuadElement.NODE_DOF)):
                        ij.add(((iN) * QuadElement.NODE_DOF + i, (jN) * QuadElement.NODE_DOF + j))

        (arr_ii, arr_jj) = map(np.array, zip(*sorted(ij)))

        idx_tril_diagonal = (arr_jj <= arr_ii)
        idx_tril = (arr_jj < arr_ii)

        ii_tril = np.r_[arr_ii[idx_tril_diagonal], arr_ii[idx_tril]]
        jj_tril = np.r_[arr_jj[idx_tril_diagonal], arr_jj[idx_tril]]
        ii_full = np.r_[arr_ii[idx_tril_diagonal], arr_jj[idx_tril]]
        jj_full = np.r_[arr_jj[idx_tril_diagonal], arr_ii[idx_tril]]

        return ii_tril, jj_tril, ii_full, jj_full

Mesh = QuadMesh # for now no triangle mesh, also only structured mesh for now

class StructuredQuadMesh(QuadMesh):
    def __init__(self, domain_size: Tuple[float, float], element_size: Tuple[float, float]) :
        (dom_x, dom_y) = domain_size
        (dx, dy) = element_size
        assert (dom_x > dx) and (dom_y > dy), ''

        # single element is shared, since all elements have the same geometry
        self.quads = [QuadElement(np.array([(0,0), (dx, 0), (dx, dy), (0, dy)]))]
        # number of elements
        (self.nelx, self.nely) = (int(dom_x/dx), int(dom_y/dy))
        # grids of nodes and elements
        self.nodes = np.arange((self.nely+1)*(self.nelx+1)).reshape(self.nely+1, self.nelx+1)
        self.elements = np.arange(self.nely*self.nelx).reshape(self.nely, self.nelx)

        # nodes that are part of corresponding element: shape (nelx*nely, 4)
        self.E = np.lib.stride_tricks.sliding_window_view(
            self.nodes, (2,2)
        ).reshape(self.elements.size, 4)[:,[0,1,3,2]] # node order is anti-clockwise, hence column indexing

    # forward the integration call to the single element
    def integrate(self, C_material: np.ndarray) -> np.ndarray : 
        return self.quads[0].integrate(C_material)


class LinearSystemBoundaryCondition(ABC):    
    @abstractmethod
    def apply(self, state: 'LinearSystem') -> None : ...

class LinearSystem:
    C: np.ndarray
    u: np.ndarray
    f: np.ndarray

    mesh: Mesh
    bcs: Set[LinearSystemBoundaryCondition]

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        self.C = np.zeros((mesh.nodes.size*QuadElement.NODE_DOF, mesh.nodes.size*QuadElement.NODE_DOF), dtype=float)
        self.f = np.zeros((mesh.nodes.size*QuadElement.NODE_DOF, 1), dtype=float)
        self.u = float('nan')*np.ones_like(self.f)

        self.bcs = set()

        (self.ii_tril, self.jj_tril, self.ii_full, self.jj_full) = self.mesh.generate_sparse_pattern()

    def add_boundary_condition(self, bc: LinearSystemBoundaryCondition) -> None :
        if (bc not in self.bcs) : bc.apply(self)

    def solve_primary_field(self) : 
        C_sparse = sparse.csr_matrix(
                (self.C[self.ii_tril, self.jj_tril], (self.ii_full, self.jj_full)), 
                shape=self.C.shape
            )
        self.u = sparse.linalg.spsolve(C_sparse, self.f)