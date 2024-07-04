#!/usr/bin/env python
import sys
import numpy as np
np.set_printoptions(linewidth=350, threshold=sys.maxsize, suppress=True, precision=12)

from pint import UnitRegistry
ureg = UnitRegistry()
SQRT_HBAR_OVER_2PIC = (np.sqrt(ureg.hbar / (2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1))).to('unified_atomic_mass_unit**0.5 * bohr').magnitude

import numpy as np
from src.wilson.retrievedata import CFOURdata, GaussianData

basis = 'T'

# gaussian_path = '/home/vlew/scriptsHPC/input_data_info/dftGaussian/formaldehyde/g16_hfoptanhramanDZ.out'
gaussian_path = '/home/vlew/scriptsHPC/input_data_info/dftGaussian/formaldehyde/g16_coh2hfanh_newopt_raman_newTZ.out'
q3 = '/home/vlew/scriptsHPC/input_data_info/dftGaussian/formaldehyde/g16_hfTZanh_newopt_raman_new_3quanta.out'
# gaussian_path = f'/home/vlew/scriptsHPC/input_data_info/dftGaussian/formaldehyde/g16_coh2hfoptanhraman{basis}Z.out'
data = {'source': 'gaussian', 'type': 'log', 'files': {'log': gaussian_path, '3quanta': q3}}
gaussianparser = GaussianData(data)

allstates_Gaussian, allstates_Gaussian_harm = gaussianparser.getAllStates()
#
funds = {k: v for k, v in allstates_Gaussian.items() if len(k) == 1}
sorted_data = {k: funds[k] for k in sorted(funds)}
freqs = np.array(list(sorted_data.values()))
print('freqs in cm-1', freqs)
#
funds_harm = {k: v for k, v in allstates_Gaussian_harm.items() if len(k) == 1}
sorted_data_harm = {k: funds_harm[k] for k in sorted(funds_harm)}
freqs_harm = np.array(list(sorted_data_harm.values()))
print('freqs in cm-1 harmonic', repr(freqs_harm))

dipole_derivs_Gaussian1, dipole_derivs_Gaussian2 = gaussianparser.getDipDers()

from scipy import constants
bohr_radius = constants.physical_constants['Bohr radius'][0]
debye_to_SI = 10**-21/constants.c
au_to_SI = constants.e * bohr_radius
debye_to_au = debye_to_SI / au_to_SI

dipole_matrix = dipole_derivs_Gaussian1 * debye_to_au

np.set_printoptions(linewidth=350, threshold=sys.maxsize, suppress=True, precision=12)
# print(dipole_matrix)

# Constants
# hbar = 1.0545718e-34  # Reduced Planck's constant, in joule second
# c = 2.99792458e8  # Speed of light, in meters per second
pi = np.pi
sqrt_two_pi_c = np.sqrt(constants.hbar / (2 * pi * constants.c))

print('-----------------------------')
# Calculate the transformed dipole moments
mu_T = (1 / np.sqrt(2)) * dipole_matrix * SQRT_HBAR_OVER_2PIC * (1 / np.sqrt(freqs)).reshape(-1, 1)
# mu_T = dipole_matrix * (1 / np.sqrt(freqs_hz)).reshape(-1, 1)
print('P1', dipole_matrix)
print('SQRT_HBAR_OVER_2PIC', SQRT_HBAR_OVER_2PIC)
print('1 / np.sqrt(freqs)', (1 / np.sqrt(freqs)).reshape(-1, 1))
print('-----------------------------')

# Print the resulting transformed dipole moments
print(mu_T)

vibdata_path = f'/home/vlew/scriptsHPC/input_data_info/cfourdata/hcoh/HFcc_pV{basis}Z/out'
cubic_path = f'/home/vlew/scriptsHPC/input_data_info/cfourdata/hcoh/HFcc_pV{basis}Z/cubic'
dipole_path = f'/home/vlew/scriptsHPC/input_data_info/cfourdata/hcoh/HFcc_pV{basis}Z/dipole'
# polar_path = '/home/vlew/scriptsHPC/input_data_info/coh2aldehyde_HFcc-pVDZ/polar.pkl'

files = {'out': vibdata_path, 'cubic': cubic_path, 'dipolexyz': dipole_path}
data = {'source': 'cfour', 'type': 'out', 'files': files}

cfourparser = CFOURdata(data)

dipole_derivs_CFOUR1, dipole_derivs_CFOUR2 = cfourparser.getDipDers()

print(dipole_derivs_CFOUR1)

allstates_CFOUR, allstates_CFOUR_harm = cfourparser.getAllStates()
funds_c4 = {k: v for k, v in allstates_CFOUR_harm.items() if len(k) == 1}
sorted_data_c4 = {k[0]: funds_c4[k] for k in sorted(funds_c4)}
freqs_c4 = np.array(list(sorted_data_c4.values()))
print('freqs in cm-1', freqs_c4)
print(sorted_data_c4)
print('---------------------------------------------------------------')

cff_gaussian = gaussianparser.getCFF()
print(type(cff_gaussian[0]))
print(cff_gaussian[0])

print('---------------------------------------------------------------')

cff_cfour = cfourparser.getCFF()
print(cfourparser.sourcetype, cfourparser.files)
print(cff_cfour)

from scriptsHPC.utils import parseCFOUR
cubi = parseCFOUR.pCubicORQuartic(cubic_path)
# print(cubi)

cubi_tensor = parseCFOUR.getCubicPost(sorted_data_c4, cubi, recipcm=True)
# print(cubi_tensor)

print('---------------------------------------------------------------')

cubi_tensor11 = parseCFOUR.getCubicPost(sorted_data_c4, cubi, recipcm=False)
print(cubi_tensor11)

