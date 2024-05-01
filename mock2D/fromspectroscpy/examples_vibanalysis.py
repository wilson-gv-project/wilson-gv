######################################################################
##                                                                  ##
##    EXAMPLES OF USING openrsp_tensor_reader and vib_analysis      ##
##                                                                  ##
######################################################################

import numpy as np

import openrsp_tensor_reader as orspReader
# import os
#
# path = "./2dcalc_1205_1"
# files = os.listdir(path)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, required=False)
args = parser.parse_args()
# print(args)

props_list, tens_list = orspReader.read_openrsp_tensor_file(args.f)
print(len(props_list))
# allprops = []

for i in range(len(props_list)):
    print(args.f)
    props_list[i].addTensor(tens_list[i])
    props_list[i].tellProp()

hessianProp = props_list[-1]
hessian_tensor = hessianProp.tensor

import vib_analysis as va
coordshere, chargeshere, masseshere = va.read_mol('MOLECULE.INP')
cut_w, cut_T, N_3, min_element, max_element = va.get_vib_harm_freqs_and_eigvecs(coordshere, chargeshere, masseshere, hessian_tensor,
                                                 outproj=True, print_level=1, harmonic_frequency_limits='Keep all')

# print(cut_T)
print(N_3, min_element, max_element)

# vib_w, vib_T = va.get_vibrational_w_and_T(is_linear=False, num_coordinates=N_3,
#                                           w=cut_w, T=cut_T, outproj=True, print_level=1)

# print(vib_w)

# cart to normal transformation
# print(props_list[1].tensor.shape)
# print(cut_T.shape)

tensor = props_list[1].tensor


# (9, 9, 3), (9, 3) -> (9, 3, 3) (9, 3) -> (3, 3, 3)
mu_Q1 = np.einsum('ijk,jq->iqk', tensor, cut_T)
mu_Q = np.einsum('ijk,iq->qjk', mu_Q1, cut_T)
print(mu_Q.shape, tensor.shape)
print(mu_Q)

ff = orspReader.cart2normal(props_list[1], 'MOLECULE.INP', args.f)
print(ff)

import transform_nc_to_nm as trnsfM1
old1 = trnsfM1.transform_cartesian_to_normal(tensor, (9, 9, 3), 2, 9, min_element, max_element, cut_T)
print(old1.shape)
print(old1)


