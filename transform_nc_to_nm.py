# SpectroscPy x.x.x
# SpectroscPy is a script package developed by and containing contributions from

    # Karen Oda Hjorth Minde Dundas
    # Magnus Ringholm
    # Yann Cornation
    # Benedicte Ofstad

# The package is released under a LGPL licence.
# For questions, please contact on karen.o.dundas@uit.no

# The functionality in transform_nc_to_nm.py and vib_analysis.py to various extents replicates or was 
# written on the basis of (or inspired by) corresponding functionality in the Dalton quantum chemistry
# program (Copyright (C) 2018 by the authors of Dalton) and unreleased code associated with Dalton/OpenRSP. # In the released Dalton program, the source code files containing this functionality are 
# DALTON/abacus/abavib.F and DALTON/abacus/hergp.F. An overview of the routines in the current module
# transform_nc_to_nm for which this applies and the routines in Dalton from which the functionality was
# replicated, based upon or inspired by, is as follows:

# Routine in transform_nc_to_nm.py              Corresponding routine(s)

# transform_cartesian_to_normal_one_rank        VIBV3 in DALTON/abacus/abavib.F (and unreleased OpenRSP/
#                                               Dalton functionality)
# transform_cartesian_to_normal                 VIBV3 in DALTON/abacus/abavib.F (and unreleased OpenRSP/
#                                               Dalton functionality)

# Dalton is released under a GNU Lesser General Public License v2.1, which can be found at
# https://gitlab.com/dalton/dalton/blob/master/LICENSE
# The full sourcecode of Dalton can be found at https://gitlab.com/dalton/dalton

import numpy as np
import copy

# Simple function to multiply elements in list
def list_product_int(l):

    v = 1

    for i in l:
        v *= i

    return v


# Recursive routine to transform one rank of geometric differentiation from Cartesian to normal mode
# basis. To be called from transform_cartesian_to_normal
def transform_cartesian_to_normal_one_rank(w, rank_geo, c, offset, offset_b, ind_w, \
                                        num_coordinates, min_element, max_element, \
                                        num_nongeo_components, A, B, T, act):

    if (c > 0):
        for i in range(num_coordinates):

            if ((c <= w[0])):
                if (i > max_element):
                    act = 0
                if (c == min(w)):
                    ind_w = i

            transform_cartesian_to_normal_one_rank(w, rank_geo, c-1, offset, offset_b, ind_w, \
                                                num_coordinates, min_element, max_element, \
                                                num_nongeo_components, A, B, T, act)

            if (c > min(w)):
                act = 1

            if not (c in w):
                offset[0] += int((num_nongeo_components) * num_coordinates**((c - 1)))

            offset_b[0] += int((num_nongeo_components) * num_coordinates**((c - 1)))

        if not(c in w):
            offset[0] -= int((num_coordinates * num_nongeo_components) * num_coordinates**((c - 1)))

        offset_b[0] -= int((num_coordinates * num_nongeo_components) * num_coordinates**((c - 1)))

    else:
        if (act and not (ind_w < min_element)):

            tmp = np.zeros((int(num_nongeo_components)))
            offset_n = offset[0]

            for j in range(num_coordinates):
                for i in range(int(num_nongeo_components)):
                    tmp[i] += A[offset_n + i] * T[j][ind_w - min_element]

                offset_n += int(num_nongeo_components * num_coordinates**((min(w) - 1)))

            B[offset_b[0]:offset_b[0] + int(num_nongeo_components)] = tmp[:]

    return


# Routine to transform a tensor A from Cartesian to normal mode basis
# Takes a tensor A with dimension dims where (the assumed to be first) rank_geo ranks involve
# differentiation w.r.t. Cartesian displacements, where there are num_coordinates such displacements;
# then transforms and returns another array A where all geometric ranks are in normal mode
# basis, using the transformation matrix T, where num_vib_coordinates is the number of normal modes
# Example arguments: A contains the second derivatives of the dipole moment for a system with 8 atoms
# dims should then be the tuple (24, 24, 3)
# rank_geo should be 2
# num_coordinates should be 24
# num_vib_coordinates should be 18 (if the system is not linear)
# T is a transformation matrix of size (num_coordinates, num_vib_coordinates) (it can also be
# (num_coordinates, num_coordinates) but only the first num_vib_coordinates components of the second rank
# should be used) expressing the normal modes in terms of Cartesian displacements
def transform_cartesian_to_normal(A, dims, rank_geo, num_coordinates, min_element, max_element, T):

    num_vib_coordinates = max_element - min_element + 1
    # Collapse array to be reshaped
    tmp = np.reshape(A, list_product_int(dims))
    A = tmp

    B = np.zeros(A.shape)

    num_nongeo_components = list_product_int(dims) / (num_coordinates**rank_geo)
    ind_w = 0

    # Go through ranks of geometric differentiation and transform
    for i in range(rank_geo):

        offset = [0]
        offset_b = [0]
        w = [i + 1]

        # Call recursive routine to transform one rank
        transform_cartesian_to_normal_one_rank(w, rank_geo, rank_geo, offset, offset_b, ind_w, \
                                            num_coordinates, min_element, max_element, \
                                            num_nongeo_components,  A, B, T, 1)

        A = copy.deepcopy(B)
        B = 0.0* B

    A = A.reshape(dims)

    return A
