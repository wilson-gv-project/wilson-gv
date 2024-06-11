#!/usr/bin/env python
from mock2D.spectrum import c2DIRmain

import numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=10)
import os

print(f"""Generated with: 
'getcwd:        {os.getcwd()}
'__file__:      {__file__}\n\n""")

# type of spectrum - what's on axes
w1mw2 = False

# start1, stop1, step = 1250., 1550., 50.
# start2, stop2, step = 2400., 3600., 50.

start1, stop1, step1 = 1250., 2250., 0.5
start2, stop2, step2 = 2470., 3250., 0.5

# ranges for 2 frequencies
omega1 = np.arange(start1, stop1, step1)
omega2 = np.arange(start2, stop2, step2)

y =  omega2 if not w1mw2 else omega2-omega1

# meshgrid for spectrum
x_mesh, y_mesh =  np.meshgrid(omega1, y)

g16files = {'log': '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_coh2hfoptanhramanDZ.out',
            '3quanta': '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_hfoptanhramanDZ_3q.out',}

from scriptsHPC.utils import parseGaussian
qq = parseGaussian.parse_frequencies(g16files['3quanta'])
# print(qq)
# for i in qq:
#     print(qq[i])

print('\n-----------------------------------\n')

# quit()

dimlessFile = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/QUADRATURE'

# spectrum is computing intensities on the grid of 2 frequencies
setup = c2DIRmain.SpectrumEVV(omega1, omega2, data={'source': 'gaussian',
                                                'type': 'log',
                                                'files':g16files})

print(setup.all_states)
print(setup.all_states[('0', '0', '0')])
# quit()

# collect derivatives
ders = setup.getDerivs()
# print(setup.fundamentals)
# print(setup.fundamentals_harmonic)

# quit()

# print(setup.all_states)
# add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
setup.addTerms(None, None, None, None)

# print(setup.electric_avrg, '\n')
print(setup.data, '\n')

# print derivatives
# c2DIRmain.printed2DIRtensors(setup)


gcmrec = 10.
gamma = c2DIRmain.rec_cm2hartree_amu_bohr_2(gcmrec)
print(c2DIRmain.rec_cm2hartree_amu_bohr_2(1660.), 1660)
print(c2DIRmain.rec_cm2hartree_amu_bohr_2(gcmrec), gcmrec)
# quit()

import time
start_time = time.time()
Z, savedict = setup.intensity(gamma, {})
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time - setup.intensity: {execution_time} seconds")


percent = 0.0
# f1 = setup.plot2Dplotly(Z, w1mw2, gamma, percent, step)
# c2DIRmain.makeHTML([f1], w1mw2, step, percent)

# setup.plot2Dmatplotlib(Z, w1mw2)
# setup.plt_matshow(Z, w1mw2)
# setup.plt_matshow_Skewed(Z, w1mw2, skew_factor=6, Gamma=gamma)

figfilename = f'./hcoh_HFcc_pVDZ_Gaussian_Gamma{gamma}_({start1}_{stop1-step1}_{step1})({start2}_{stop2-step2}_{step2}).svg'

setup.plot2Dmatplotlib(Z, w1mw2, figfilename)
