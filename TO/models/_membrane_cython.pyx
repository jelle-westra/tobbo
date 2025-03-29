import numpy as np
cimport numpy as np
cimport cython

cdef int NUMBER_OF_NODES_X_ELEMENT = 4
cdef int NUMBER_OF_NODAL_DOF = 2

@cython.boundscheck(False)
@cython.wraparound(False) 
def fill_C_matrix(
    np.ndarray global_stiffness_mat,
    np.ndarray E,
    np.ndarray density_vector,
    np.ndarray Ke_material,
    np.ndarray Ke_void
):
    cdef int elem_pos, ii, jj, iN, jN
    cdef np.ndarray KeNNDOF

    for elem_pos in range(len(density_vector)):
        for ii in range(NUMBER_OF_NODES_X_ELEMENT):
            # getting the vertex index in mesh (mesh vertex index)
            iN = int(E[elem_pos, ii])

            for jj in range(NUMBER_OF_NODES_X_ELEMENT):
                # getting the vertex index in mesh (mesh vertex index)
                jN = int(E[elem_pos, jj])
                if (iN < jN - 1) : break

                KeNNDOF = (Ke_material if density_vector[elem_pos] else Ke_void)[(ii) * NUMBER_OF_NODAL_DOF:(ii + 1) * NUMBER_OF_NODAL_DOF, (jj) * NUMBER_OF_NODAL_DOF:(jj + 1) * NUMBER_OF_NODAL_DOF]

                global_stiffness_mat[(iN) * NUMBER_OF_NODAL_DOF + 0, (jN) * NUMBER_OF_NODAL_DOF + 0] += KeNNDOF[0, 0]
                global_stiffness_mat[(iN) * NUMBER_OF_NODAL_DOF + 0, (jN) * NUMBER_OF_NODAL_DOF + 1] += KeNNDOF[0, 1]
                global_stiffness_mat[(iN) * NUMBER_OF_NODAL_DOF + 1, (jN) * NUMBER_OF_NODAL_DOF + 0] += KeNNDOF[1, 0]
                global_stiffness_mat[(iN) * NUMBER_OF_NODAL_DOF + 1, (jN) * NUMBER_OF_NODAL_DOF + 1] += KeNNDOF[1, 1]