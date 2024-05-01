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
# vib_analysis for which this applies and the routines in Dalton from which the functionality was
# replicated, based upon or inspired by, is as follows:

# Routine in vib_analysis.py            Corresponding routine(s)

# project_out_transl_and_rot            VIBHES in DALTON/abacus/abavib.F
# get_vib_harm_freqs_and_eigvecs        VIBNOR and VIBANA, possibly also VIBCTL, ISOMOL (all former four 
#                                       in DALTON/abacus/abavib.F), VIBMAS in DALTON/abacus/hergp.F

# Dalton is released under a GNU Lesser General Public License v2.1, which can be found at
# https://gitlab.com/dalton/dalton/blob/master/LICENSE
# The full sourcecode of Dalton can be found at https://gitlab.com/dalton/dalton

import numpy as np
import copy
import re
from math import acos, pi
from mock2D.fromspectroscpy.parameters import one_twelfth_carbon, electron_mass, hbar

# Take coordinates and determine if molecule is linear; return 1 if linear or 0 if not
# Coordinates given as nested list - first atoms and then Cartesian coordinates
def mol_is_linear(coords):

	# If any angle is not 180 degrees, return as not linear
	for i in range(len(coords)):

		for j in range(i + 1, len(coords)):

			for k in range(j + 1, len(coords)):

				a = [coords[j][m] - coords[i][m] for m in range(len(coords[j]))]
				b = [coords[k][m] - coords[j][m] for m in range(len(coords[j]))]

				alen = sum([m**2 for m in a])**0.5
				blen = sum([m**2 for m in b])**0.5

				theta = acos(sum([a[m]*b[m] for m in range(len(a))]) / (alen * blen))

				# Deemed not linear if some bond angle is close enough to pi (180 degrees)
				if not((pi - abs(theta)) < 0.00002):
					return 0

	# To get here, all angles must be 180 degrees and thus molecule is linear
	return 1


# Read a Dalton mol file and return the coordinates, charges and masses of each atom
def read_mol(fname):

	f = open(fname, 'r')
	
	start_of_atoms = 0

	coords = []
	charges = []
	masses = []

	while not(start_of_atoms):

		nline = f.readline()
		
		start_of_atoms = re.search('atomtypes', nline, flags=re.I)

		if (start_of_atoms):

			num_atomtypes = int(re.search(r'\d+', nline).group())

#NOTE: H, C, N, O, P masses from most abundant isotope - other elements may be abundance-weighted
# isotope mass average
# The values are verified with the NIST/CODATA, and are originally taken from elements.py 
# Copyright (c) 2005-2019, Christoph Gohlke

# These are in relative atomic masses, or m[kg]/(1/12)m_C[kg].  The mass relative to one 12th of a carbonatom.
	mass_dict = {
1:   1.007825,
2:   4.002602,
3:   6.941,
4:   9.012182,
5:   10.811,
6:   12.0,
7:   14.003074,
8:   15.994915,
9:   18.9984032,
10:  20.1797,
11:  22.98976928,
12:  24.3050,
13:  26.9815386,
14:  28.0855,
15:  30.973763,
16:  32.065,
17:  35.453,
19:  39.0983,
18:  39.948,
20:  40.078,
21:  44.955912,
22:  47.867,
23:  50.9415,
24:  51.9961,
25:  54.938045,
26:  55.845,
27:  58.933195,
28:  58.6934,
29:  63.546,
30:  65.409,
31:  69.723,
32:  72.64,
33:  74.92160,
34:  78.96,
35:  79.904,
36:  83.798,
37:  85.4678,
38:  87.62,
39:  88.90585,
40:  91.224,
41:  92.906,
42:  95.94,
43:  98.0,
44:  101.07,
45:  102.905,
46:  106.42,
47:  107.8682,
48:  112.411,
49:  114.818,
50:  118.710,
51:  121.760,
52:  127.60,
53:  126.904,
54:  131.293,
55:  132.9054519,
56:  137.327,
57:  138.90547,
58:  140.116,
59:  140.90765,
60:  144.242,
61:  145.0,
62:  150.36,
63:  151.964,
64:  157.25,
65:  158.92535,
66:  162.500,
67:  164.930,
68:  167.259,
69:  168.93421,
70:  173.04,
71:  174.967,
72:  178.49,
73:  180.94788,
74:  183.84,
75:  186.207,
76:  190.23,
77:  192.217,
78:  195.084,
79:  196.966569,
80:  200.59,
81:  204.3833,
82:  207.2,
83:  208.98040,
84:  210,
85:  210,
86:  220,
87:  223,
88:  226,
89:  227,
91:  231.03588,
90:  232.03806,
93:  237,
92:  238.02891}

	for i in range(num_atomtypes):

		nline = f.readline()

		re.search('atomtypes', nline, flags=re.I)
		curr_charge = int(re.sub(r'charge\s*\=\s*', '', re.search(r'charge\s*\=\s*\d+', nline, flags=re.I).group(), flags=re.I))
		curr_num_atoms = int(re.sub(r'atoms\s*\=\s*', '', re.search(r'atoms\s*\=\s*\d+', nline, flags=re.I).group(), flags=re.I))

		for j in range(curr_num_atoms):

			nline = f.readline().split()
			coords.append([float(k) for k in nline[1:]])
			charges.append(curr_charge)
			masses.append(mass_dict[curr_charge])

	return coords, charges, masses


# Project out translational and rotational degrees of freedom from Hessian
def project_out_transl_and_rot(num_vib_modes, hess, coords, is_linear):

    if (is_linear):
        t_r_mat = np.zeros((3 * len(coords), 5))
    else:
        t_r_mat = np.zeros((3 * len(coords), 6))

    k = 0

    for i in range(len(coords)):
        t_r_mat[k, 0] = 1.0
        t_r_mat[k + 1, 1] = 1.0
        t_r_mat[k + 2, 2] = 1.0
        t_r_mat[k, 3] = -1.0 * coords[i][1]
        t_r_mat[k + 1, 3] = coords[i][0]
        t_r_mat[k + 1, 4] = -1.0 * coords[i][2]
        t_r_mat[k + 2, 4] = coords[i][1]

        if (not is_linear):
            t_r_mat[k, 5] = coords[i][2]
            t_r_mat[k + 2, 5] = -1.0 * coords[i][0]

        k += 3


    # This would be the same as taking q, r = np.linalg.qr(t_r_mat) and using q
    t_r_ort =  np.linalg.qr(t_r_mat, mode='reduced')[0]

    # May be sign change here

    proj_t_r = (np.dot(t_r_ort, t_r_ort.T) - np.identity(3 * len(coords)))

    return np.dot(np.dot(proj_t_r, hess), proj_t_r)


# Take molecule filename and (NumPy) Hessian array and return number of Cartesian coordinates,
# whether the molecule is linear, vibrational frequencies and nm transformation matrix
def get_vib_harm_freqs_and_eigvecs(coords, charges, masses, hess, outproj, print_level, \
                                   harmonic_frequency_limits):

    amu_au_sqrt = (electron_mass/one_twelfth_carbon)**0.5
    is_linear = mol_is_linear(coords)
    num_vib_modes = 3 * len(coords) - 6 + is_linear
    mass_weighted_diag = np.zeros((3 * len(coords), 3 * len(coords) ))

    for i in range(len(masses)):

        for j in range(3):

            mass_weighted_diag[3 * i + j, 3 * i + j] = amu_au_sqrt/(masses[i]**0.5)

    if (outproj):
        vib_hess = project_out_transl_and_rot(num_vib_modes, hess, coords, is_linear)
    else:
        vib_hess = np.copy(hess)

    for i in range(3*len(coords)):

        for j in range(3*len(coords)):

            vib_hess[j,i] = mass_weighted_diag[j,j] * vib_hess[j,i] * mass_weighted_diag[i,i]

    w, T = np.linalg.eig(vib_hess)

    idx = w.argsort()[::-1]
    w = w[idx]
    T = T[:,idx]

    for i in range(3*len(coords)):

        for j in range(3*len(coords)):

            T[j,i] = mass_weighted_diag[j,j] * T[j,i]

    w = [float(i)**0.5 for i in w.real]

    vib_w, vib_T = get_vibrational_w_and_T(is_linear, 3*len(coords), w, T.real, outproj, print_level)

    if (harmonic_frequency_limits == 'Keep all'):
        cut_w = np.copy(vib_w)
        cut_T = np.copy(vib_T)
        min_element = 0
        max_element = len(vib_w) - 1
    else:
        cut_w, cut_T, min_element, max_element = \
            cut_at_harmonic_frequency_limits(vib_w, vib_T, 3*len(coords), harmonic_frequency_limits)

    return cut_w, cut_T, 3*len(coords), min_element, max_element


def cut_at_harmonic_frequency_limits(w, T, num_coordinates, harmonic_frequency_limits):

    num_valid_modes = 0
    for i in range(len(w)):
        if (w[i] > harmonic_frequency_limits[0] and w[i] < harmonic_frequency_limits[1]):
            num_valid_modes = num_valid_modes + 1

    cut_w = np.zeros(num_valid_modes)
    cut_T = np.zeros((num_coordinates, num_valid_modes))

    k = 0
    min_element = len(w)
    max_element = 0
    for i in range(len(w)):
        if (w[i] > harmonic_frequency_limits[0] and w[i] < harmonic_frequency_limits[1]):
            if (i < min_element):
                min_element = i
            if (i > max_element):
                max_element = i

            cut_w[k] = w[i]
            for j in range(num_coordinates):
                cut_T[j][k] = T[j][i]

            k = k + 1

    return cut_w, cut_T, min_element, max_element


def get_vibrational_w_and_T(is_linear, num_coordinates, w, T, outproj, print_level):

    if (is_linear):
        vib_degrees_of_freedom = num_coordinates - 5
    else:
        vib_degrees_of_freedom = num_coordinates - 6

    if (outproj):
        if (print_level > 0):
            print('Number of vibrational degrees of freedom: ', vib_degrees_of_freedom)
            print('Number of frequencies calculated: ', len(w))
            print('Will only include the vibrational frequencies, cutting the rotational' \
                    'and translational degrees of freedom')

        vib_w = np.zeros(vib_degrees_of_freedom)
        vib_T = np.zeros((num_coordinates, vib_degrees_of_freedom))

        for i in range(vib_degrees_of_freedom):
            vib_w[i] = w[i]
            for j in range(num_coordinates):
                vib_T[j][i] = T[j][i]
    else:
        vib_w = np.copy(w)
        vib_T = np.copy(T)

    # Look at this for the unproj case once actual environments can be used
    if (outproj):
        for i in range(len(vib_w)):
            if (vib_w[i].imag != 0):
                print('\n')
                print('Error in get_vibrational_frequencies_and_intensities')
                print('You have imaginary frequency values, are you certain your geometry is in its')
                print('minima?')
                exit()

    return vib_w, vib_T
