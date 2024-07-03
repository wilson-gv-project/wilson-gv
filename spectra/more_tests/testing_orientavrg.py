#!/usr/bin/env python
from wilson.spectrum import c2DIRmain
import numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=17)

electric_avrg_r = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
                   [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]
                   ]

mechanical_avrg_r = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],  # abc 1
                     [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],  # abc 1
                     [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],  # aba 3
                     [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],  # abb 4
                     [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],  # aab 5
                     [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']]  # abb 4

start1, stop1, step1 = 1190., 2280., 10.
start2, stop2, step2 = 2870., 3560., 10.

omega1 = np.arange(start1, stop1, step1)
omega2 = np.arange(start2, stop2, step2)

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
print('---\n')

derivatives_data = setup.deriv_data
inds = [(0, 1), (1, 3), (3, 0), (2, 2), (4, 4), (1, 1), (0, 0), (3, 1)]

# [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))]
tensor_el1 = c2DIRmain.avrg_abc_tensor_new(electric_avrg_r[0], derivatives_data, setup.gammaCompsAll)
print(electric_avrg_r[0])
tests = []
for i in inds:
    # print(tensor_el1[i])
    tot_hand = 0.
    for j in setup.gammaCompsAll:
        a, b, c, d = j
        tot_hand += derivatives_data['mu_Q'][i[0], b]*derivatives_data['alpha_Q'][i[1], a, d]*derivatives_data['mu_QQ'][i[0], i[1], c]
    tot_hand = tot_hand/15.
    # print(tot_hand)
    # print(tensor_el1[i]==tot_hand)
    tests.append(tensor_el1[i]==tot_hand)
print(f'tensor_el1 - {tests}')
print('---\n')

# [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]
tensor_el2 = c2DIRmain.avrg_abc_tensor_new(electric_avrg_r[1], derivatives_data, setup.gammaCompsAll)
print(electric_avrg_r[1])
tests = []
for i in inds:
    # print(tensor_el2[i])
    tot_hand = 0.
    for j in setup.gammaCompsAll:
        a, b, c, d = j
        tot_hand += derivatives_data['mu_Q'][i[0], b]*derivatives_data['alpha_QQ'][i[0], i[1], a, d]*derivatives_data['mu_Q'][i[1], c]
    tot_hand = tot_hand/15.
    # print(tot_hand)
    # print(tensor_el2[i]==tot_hand)
    tests.append(tensor_el2[i]==tot_hand)
print(f'tensor_el2 - {tests}')
print('---\n')

inds = [(0, 1, 2), (1, 3, 4), (3, 0, 5), (2, 2, 2), (0, 1, 4), (5, 3, 1), (0, 2, 5), (3, 1, 1), (0, 0, 0)]
# [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc']
tensor_mech1 = c2DIRmain.avrg_abc_tensor_new(mechanical_avrg_r[0], derivatives_data, setup.gammaCompsAll)
print(mechanical_avrg_r[0])
tests = []
for i in inds:
    # print(tensor_mech1[i])
    tot_hand = 0.
    for j in setup.gammaCompsAll:
        a, b, c, d = j
        tot_hand += derivatives_data['mu_Q'][i[0], b]*derivatives_data['alpha_Q'][i[1], a, d]*derivatives_data['mu_Q'][i[2], c]
    tot_hand = tot_hand/15.
    # print(tot_hand)
    # print(tensor_mech1[i]==tot_hand)
    tests.append(tensor_mech1[i]==tot_hand)
print(f'tensor_mech1 - {tests}')
print('---\n')

# [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc']
tensor_mech3 = c2DIRmain.avrg_abc_tensor_new(mechanical_avrg_r[2], derivatives_data, setup.gammaCompsAll)
print(mechanical_avrg_r[2])
tests = []
for i in inds:
    # print(tensor_mech3[i])
    tot_hand = 0.
    for j in setup.gammaCompsAll:
        a, b, c, d = j
        tot_hand += derivatives_data['mu_Q'][i[0], b]*derivatives_data['alpha_Q'][i[1], a, d]*derivatives_data['mu_Q'][i[0], c]
    tot_hand = tot_hand/15.
    # print(tot_hand)
    # print(tensor_mech3[i]==tot_hand)
    tests.append(tensor_mech3[i]==tot_hand)
print(f'tensor_mech3 - {tests}')
print('---\n')

# [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']
tensor_mech4 = c2DIRmain.avrg_abc_tensor_new(mechanical_avrg_r[3], derivatives_data, setup.gammaCompsAll)
print(mechanical_avrg_r[3])
tests = []
for i in inds:
    # print(tensor_mech4[i])
    tot_hand = 0.
    for j in setup.gammaCompsAll:
        a, b, c, d = j
        tot_hand += derivatives_data['mu_Q'][i[0], b]*derivatives_data['alpha_Q'][i[1], a, d]*derivatives_data['mu_Q'][i[1], c]
    tot_hand = tot_hand/15.
    # print(tot_hand)
    # print(tensor_mech4[i]==tot_hand)
    tests.append(tensor_mech4[i]==tot_hand)
print(f'tensor_mech4 - {tests}')
print('---\n')

# [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc']
tensor_mech5 = c2DIRmain.avrg_abc_tensor_new(mechanical_avrg_r[4], derivatives_data, setup.gammaCompsAll)
print(mechanical_avrg_r[4])
tests = []
for i in inds:
    # print(tensor_mech5[i])
    tot_hand = 0.
    for j in setup.gammaCompsAll:
        a, b, c, d = j
        tot_hand += derivatives_data['mu_Q'][i[0], b]*derivatives_data['alpha_Q'][i[0], a, d]*derivatives_data['mu_Q'][i[1], c]
    tot_hand = tot_hand/15.
    # print(tot_hand)
    # print(tensor_mech5[i]==tot_hand)
    tests.append(tensor_mech5[i]==tot_hand)
print(f'tensor_mech5 - {tests}')
print('---\n')
