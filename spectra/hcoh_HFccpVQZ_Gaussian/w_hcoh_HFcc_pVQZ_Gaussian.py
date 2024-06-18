#!/usr/bin/env python
import time
start_time_global = time.time()

from mock2D.spectrum import c2DIRmain
import faulthandler
faulthandler.enable()

import numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=17)
import os

print(f"""Generated with: 
'getcwd:        {os.getcwd()}
'__file__:      {__file__}\n\n""")

# start1, stop1, step1 = 1970., 2090., 5.
# start2, stop2, step2 = 3750., 3850., 5.

start1, stop1, step1 = 1080., 2450., 1.
start2, stop2, step2 = 2609., 6200., 1.


# ranges for 2 frequencies
omega1 = np.arange(start1, stop1, step1)
omega2 = np.arange(start2, stop2, step2)


################################################################

# cfourdatafiles = {'out': '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/outfile0.out',
#                   'cubic': '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/cubic',
#                   'dipolexyz': '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/dipole',
#                   'polar': '/home/vlew/scriptsHPC/data/TESTS_240607/c4_HF_STO_3G_allopt/upd_polar/polar.pkl'
#                   }
#
# # spectrum is computing intensities on the grid of 2 frequencies
# setup = c2DIRmain.SpectrumEVV(omega1, omega2, data={'source': 'cfour',
#                                                 'type': 'out',
#                                                 'files':cfourdatafiles})

################################################################

g16files = {'log': '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_coh2hfoptanhramanQZ.out',
            '3quanta': '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_coh2hfoptanhramanQZ_3q.out',}
gamma_rc=10.
gamma = c2DIRmain.rec_cm2rec_s(gamma_rc)

# spectrum is computing intensities on the grid of 2 frequencies
setup = c2DIRmain.SpectrumEVV(omega1, omega2, data={'source': 'gaussian',
                                                'type': 'log',
                                                'files':g16files})
# print(setup.all_states)
print(setup.fundamentals)

#quit()
################################################################

# ders = setup.getDerivs()

f1 = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))]
f2 = [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]

f3 = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc']
f5 = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc']
f6 = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']
f7 = [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc']

mechanical_avrg_r = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                     [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                     [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],
                     [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],
                     [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],
                     [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']]
# print(setup.gammaCompsAll)

setup.addTerms(None, None, None, None)

log10=True
w1mw2=False
gamma_rc=10.
el, mech = False, True

gamma = c2DIRmain.rec_cm2rec_s(gamma_rc)
gamma_str = f"{gamma_rc:.2f}".replace('.', 'p')
print('gamma_str: ', gamma_str, gamma)
name=f'./svgs/gammaopt1_w1mw2{str(w1mw2)[0]}_el{str(el)[0]}_mech{str(mech)[0]}_log10{str(log10)[0]}_gamma{gamma_str}_10_noninv.svg'
print('\n-----------------------------------\n')
print(name)
print('\n-----------------------------------\n')

start_time0 = time.time()
Z, savedict = setup.intensity(gamma, {}, el=el, mech=mech, printdata=False)
end_time0 = time.time()
execution_time0 = end_time0 - start_time0
print(f"\nExecution time - setup.intensity: {execution_time0} seconds")

print('\n-----------------------------------\n')
# print(savedict)
# print(Z)
print('\n-----------------------------------\n')

start_time = time.time()
setup.plot2Dmatplotlib(Z, w1mw2=w1mw2, name=name, Gamma=gamma, dpi=200, contour_levels=6, log10=log10, shift_scale=None)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time - setup.plot2Dmatplotlib: {execution_time} seconds")

# start_time = time.time()
# name=f'./svgs/skewed_w1mw2{str(w1mw2)[0]}_el{str(el)[0]}_mech{str(mech)[0]}_log10{str(log10)[0]}_gamma{gamma_str}.svg'
# setup.plt_matshow_Skewed(Z, w1mw2=w1mw2, figfilename=name, skew_factor=6)
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time - setup.plot2Dmatplotlib: {execution_time} seconds")


end_time_global = time.time()
execution_time_global = end_time_global - start_time_global

hours, rem = divmod(execution_time_global, 3600)
minutes, seconds = divmod(rem, 60)

print("\n{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print(f"Execution time - global: {execution_time_global} seconds")

quit()


print('\n----- CFOUR -----\n')
print(ders.keys(), '\n')

print(ders['alpha_Q'])
from scipy import constants
amc_au = constants.physical_constants['atomic mass constant'][0]/constants.physical_constants['atomic unit of mass'][0]
print("amc_au = constants.physical_constants['atomic mass constant'][0]/constants.physical_constants['atomic unit of mass'][0]")
print(amc_au, "m_e in 1 au")
# print(amc_au)
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

array_2d = np.outer(np.sqrt(amc_au)*np.sqrt(w_au), np.sqrt(amc_au)*np.sqrt(w_au))

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


