#!/usr/bin/env python
from mock2D.spectrum import c2DIRmain

import numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=10)
import os

print(f"""Generated with: 
'getcwd:        {os.getcwd()}
'__file__:      {__file__}\n\n""")

start1, stop1, step1 = 1580., 1650., 10.
start2, stop2, step2 = 2870., 3100., 10.

# ranges for 2 frequencies
omega1 = np.arange(start1, stop1, step1)
omega2 = np.arange(start2, stop2, step2)


################################################################

cfourdatafiles = {'out': '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/outfile0.out',
                  'cubic': '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/cubic',
                  'dipolexyz': '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/dipole',
                  'polar': '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/upd_polar/polar.pkl'
                  }

# spectrum is computing intensities on the grid of 2 frequencies
setup = c2DIRmain.SpectrumEVV(omega1, omega2, data={'source': 'cfour',
                                                'type': 'out',
                                                'files':cfourdatafiles})

################################################################

# g16files = {'log': '/home/vlew/scriptsHPC/data/TESTS_240607/g16_hfoptanhraman_STO_3G.out',
#             '3quanta': '/home/vlew/scriptsHPC/data/TESTS_240607/g16_hfoptanhraman_STO_3G.out',}
#
# # spectrum is computing intensities on the grid of 2 frequencies
# setup = c2DIRmain.SpectrumEVV(omega1, omega2, data={'source': 'gaussian',
#                                                 'type': 'log',
#                                                 'files':g16files})

################################################################

ders = setup.getDerivs()
print('\n----- CFOUR -----\n')
print(ders.keys(), '\n')

print(ders['alpha_Q'])
from scipy import constants
# print(constants.physical_constants['atomic unit of mass'][0])
# print(constants.physical_constants['atomic mass constant'][0])
amc_au = constants.physical_constants['atomic mass constant'][0]/constants.physical_constants['atomic unit of mass'][0]
print(amc_au)
print(np.sqrt(amc_au))
# quit()

w = setup.fundamentals_harmonic
w_array = np.array(list(w.values()))
w_au = c2DIRmain.rec_cm2hartree_amu_bohr_2(w_array)
print(w_au)

# print(ders['alpha_Q']/np.sqrt(amc_au)*np.sqrt(w_au))
result = np.einsum('ijk,i->ijk', ders['alpha_Q'], np.sqrt(amc_au)*np.sqrt(w_au))
print('---------------------')
print("np.einsum('ijk,i->ijk', ders['alpha_Q'], np.sqrt(amc_au)*np.sqrt(w_au))\n")
print(result)

# result1 = np.einsum('ijkl,i->ijkl', ders['alpha_QQ'], np.sqrt(amc_au)*np.sqrt(w_au))
# result2 = np.einsum('ijkl,j->ijkl', ders['alpha_QQ'], np.sqrt(amc_au)*np.sqrt(w_au))
# print('---------------------')
# print("np.einsum('ijk,i->ijk', ders['alpha_Q'], np.sqrt(amc_au)*np.sqrt(w_au))\n")
# print(result1)

# Reshape values to (6, 1, 1, 1) so it can be broadcasted to the shape of tensor
array_2d = np.outer(np.sqrt(amc_au)*np.sqrt(w_au), np.sqrt(amc_au)*np.sqrt(w_au))
print()

# Multiply tensor by values_reshaped
result = ders['alpha_QQ'] * array_2d.reshape(6, 6, 1, 1)
print('---------------------')
print("np.outer(np.sqrt(amc_au)*np.sqrt(w_au), np.sqrt(amc_au)*np.sqrt(w_au))\nders['alpha_QQ'] * array_2d.reshape(6, 6, 1, 1)\n")
print(result)


# print derivatives
# c2DIRmain.printed2DIRtensors(setup)
quit()
################################################################

g16files = {'log': '/home/vlew/scriptsHPC/data/TESTS_240607/g16_hfoptanhraman_STO_3G.out',
            '3quanta': '/home/vlew/scriptsHPC/data/TESTS_240607/g16_hfoptanhraman_STO_3G.out',}

# spectrum is computing intensities on the grid of 2 frequencies
setup2 = c2DIRmain.SpectrumEVV(omega1, omega2, data={'source': 'gaussian',
                                                'type': 'log',
                                                'files':g16files})

################################################################

ders = setup2.getDerivs()
print('\n----- Gaussian -----\n')
print(ders.keys(), '\n')

print(ders['alpha_Q'])

# print derivatives
# c2DIRmain.printed2DIRtensors(setup)
quit()

def print_nicely(data):
    # Filter and sort the dictionary where keys do not contain '_'
    sorted_single = {k: v for k, v in sorted(data.items()) if '_' not in k}

    # Filter and sort the dictionary where keys contain '_'
    sorted_pair = {k: v for k, v in sorted(data.items()) if '_' in k}

    # Print singles
    # print("\nSingles:")
    for k, v in sorted_single.items():
        print(f"\n{k}:")
        if type(v) == tuple:
            print(f"polarizability:\n{v[0]}")
            print(f"rotation:\n{v[1]}")
        else:
            print(v)

    # print("\nPairs:")
    for k, v in sorted_pair.items():
        print(f"\n{k}:")
        if type(v) == tuple:
            print(f"polarizability:\n{v[0]}")
            print(f"rotation:\n{v[1]}")
        else:
            print(v)
    # print("\n")

    # # Print triples
    # print("Triples:")
    # for k, v in triples.items():
    #     print(f"{k}: {v:.3f}")

#
# raw_polar_file = '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/upd_polar/polarData_raw.pkl'
# import pickle
#
# # load content of pickle file
# with open(raw_polar_file, 'rb') as f:
#     raw_polar = pickle.load(f)
#
# equil_alpha = raw_polar['equil'][0]
# equil_R = raw_polar['equil'][1]
#
# # print(f'equil_alpha: \n{equil_alpha}')
# # print(f'equil_R: \n{equil_R}')
#
# # temp = np.einsum('ij,jk->ik', equil_R.T, equil_alpha)
# # alpha_prime = np.einsum('ij,jk->ik', temp, equil_R)
#
# # print(f'alpha_prime: \n{alpha_prime}')
#
# # print(f'raw_polar: \n{raw_polar}')
# # print_nicely(raw_polar)
# # quit()
#
# polders_file = '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/upd_polar/polarData.pkl'
#
# # load content of pickle file
# with open(polders_file, 'rb') as f:
#     polders = pickle.load(f)
#
# # print('\n-----------------------------------\n')
# # print_nicely(polders)
# # print(f'polders: \n{polders}')
#
# # quit()
# print('\n-----------------------------------\n')
#
# print('\n', "polders['10p']-polders['10n']")
# print('\n', (polders['10p']-polders['10n']))
# print('\n', "(polders['10p']-polders['10n'])/0.02")
# print('\n', (polders['10p']-polders['10n'])/0.02)
#
# polar_file = '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/upd_polar/polar.pkl'
#
# # load content of pickle file
# with open(polar_file, 'rb') as f:
#     polar = pickle.load(f)
#
# print('\n-----------------------------------\n')
# # print_nicely(polar)
# # print(f'polar: \n{polar[0]}')
#
