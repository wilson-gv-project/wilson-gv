#!/usr/bin/env python
from mock2D.spectrum import c2DIRmain

import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=10)
import os

print(f"""Generated with: 
'getcwd:        {os.getcwd()}
'__file__:      {__file__}\n\n""")

# type of spectrum - what's on axes
w1mw2 = False
start1, stop1, step = 1300., 1500., 20.
start2, stop2, step = 1300., 1500., 20.

# ranges for 2 frequencies
omega1 = np.arange(start1, stop1, step)
omega2 = np.arange(start2, stop2, step)

y =  omega2 if not w1mw2 else omega2-omega1

# meshgrid for spectrum
x_mesh, y_mesh =  np.meshgrid(omega1, y)

cfourdatafiles = {'vibdata': '../../scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/vibdata.pkl',
                  'cubic': '../../scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/cubic.pkl',
                  'dipole': '../../scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/dipolexyz.pkl',
                  'polar': '../../scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/polar.pkl'
                  }

dimlessFile = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/QUADRATURE'

# spectrum is computing intensities on the grid of 2 frequencies
setup = c2DIRmain.SpectrumEVV(omega1, omega2, data={'source': 'cfour',
                                                'type': 'pkl',
                                                'files':cfourdatafiles})

# ders = setup.getDerivs()

print(setup.data, '\n')
c2DIRmain.printed2DIRtensors(setup)
