#!/usr/bin/env python

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

start1, stop1, step1 = 1150., 1780., 0.5
start2, stop2, step2 = 2870., 3160., 0.5

# ranges for 2 frequencies
omega1 = np.arange(start1, stop1, step1)
omega2 = np.arange(start2, stop2, step2)
print(len(omega1), len(omega2))
# quit()
y =  omega2 if not w1mw2 else omega2-omega1

# meshgrid for spectrum
x_mesh, y_mesh =  np.meshgrid(omega1, y)

g16files = {'log': '/home/vlew/scriptsHPC/input_data_info/dftGaussian/formaldehyde/g16_coh2hfoptanhramanDZ.out',
            '3quanta': '/home/vlew/scriptsHPC/input_data_info/dftGaussian/formaldehyde/g16_hfoptanhramanDZ_3q.out',}

from scriptsHPC.utils import parseGaussian
qq = parseGaussian.parse_frequencies(g16files['3quanta'])

print('\n-----------------------------------\n')

dimlessFile = '/home/vlew/scriptsHPC/input_data_info/coh2aldehyde_HFcc-pVTZ/QUADRATURE'

# spectrum is computing intensities on the grid of 2 frequencies
setup = c2DIRmain.SpectrumEVV(omega1, omega2, input_data_info={'source': 'gaussian',
                                                'type': 'log',
                                                'files':g16files})

# print(setup.all_states)

# collect derivatives
ders = setup.getDerivs()

# add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
setup.addTerms(None, None, None, None)

print(setup.dataInfo, '\n')
# print(setup.gammaCompsAll, '\n', len(setup.gammaCompsAll), '\n')
# quit()
# print derivatives
# c2DIRmain.printed2DIRtensors(setup)
# quit()


gcmrec = 10.
gamma = c2DIRmain.rec_cm2hartree_amu_bohr_2(gcmrec)

print(c2DIRmain.rec_cm2hartree_amu_bohr_2(1660.), 1660)
print(str(c2DIRmain.rec_cm2hartree_amu_bohr_2(gcmrec)), gcmrec)

# el_bool, mech_bool = True, False
# el_bool, mech_bool = False, True
el, mech = True, True

import time
start_time0 = time.time()
Z, savedict = setup.intensity(gamma, {}, el=el, mech=mech, printdata=False)
end_time0 = time.time()
execution_time0 = end_time0 - start_time0
print(f"Execution time - setup.intensity: {execution_time0} seconds")

# percent = 0.0
if mech and not el: contr = 'mech_bool'
elif el and not mech: contr = 'el_bool'
else: contr = 'both'
figfilename = f'./{contr}_Gamma{f'{gamma:.2e}'}__{start1}_{stop1-step1}_{step1}__{start2}_{stop2-step2}_{step2}.svg'

setup.plot2Dmatplotlib(Z, w1mw2, figfilename, dpi=200, contour_levels=10)
