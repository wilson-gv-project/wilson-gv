#!/usr/bin/env python

import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=10)
import os

print(f"""Generated with: 
'getcwd:        {os.getcwd()}
'__file__:      {__file__}\n\n""")

# type of spectrum - what's on axes
w1mw2 = False
# start1, stop1, step = 1250., 1750., 7.
# start2, stop2, step = 1900., 3150., 7.

start1, stop1, step = 1250., 1550., 50.
start2, stop2, step = 2400., 3600., 50.

# ranges for 2 frequencies
omega1 = np.arange(start1, stop1, step)
omega2 = np.arange(start2, stop2, step)

y =  omega2 if not w1mw2 else omega2-omega1

# meshgrid for spectrum
x_mesh, y_mesh =  np.meshgrid(omega1, y)

cfourdatafiles = {'out': '/home/vlew/scriptsHPC/input_data_info/cfourdata/hcoh/HFcc_pVDZ/out',
                  'cubic': '/home/vlew/scriptsHPC/input_data_info/cfourdata/hcoh/HFcc_pVDZ/cubic',
                  'dipolexyz': '/home/vlew/scriptsHPC/input_data_info/cfourdata/hcoh/HFcc_pVDZ/dipole',
                  'polar': '/home/vlew/scriptsHPC/input_data_info/cfourdata/hcoh/HFcc_pVDZ/polar.pkl'
                  }

dimlessFile = '/home/vlew/scriptsHPC/input_data_info/coh2aldehyde_HFcc-pVTZ/QUADRATURE'

# spectrum is computing intensities on the grid of 2 frequencies
setup = c2DIRmain.SpectrumEVV(omega1, omega2, input_data_info={'source': 'cfour',
                                                'type': 'out',
                                                'files':cfourdatafiles})
# collect derivatives
ders = setup.getDerivs()
# quit()
print(setup.fundamentals)
print(setup.all_states)
#quit()

# add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
setup.addTerms(None, None, None, None)

# print(setup.electric_avrg, '\n')
print(setup.data_info, '\n')

# print derivatives
c2DIRmain.printed2DIRtensors(setup)

quit()

gamma = 3.8
Z, savedict = setup.intensity(gamma, {})

# print(savedict.keys())
percent = 0.0
f1 = setup.plot2Dplotly(Z, w1mw2, gamma, percent, step)
# c2DIRmain.makeHTML([f1], w1mw2, step, percent)

# setup.plot2Dmatplotlib(Z, w1mw2)
# setup.plt_matshow(Z, w1mw2)
setup.plt_matshow_Skewed(Z, w1mw2, skew_factor=6)
