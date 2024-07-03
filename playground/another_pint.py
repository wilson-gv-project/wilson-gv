#!/usr/bin/env python
import copy

from pint import UnitRegistry
ureg = UnitRegistry()
import numpy as np
import sys

# print('\n----------------------------------------')
gamma_r = 2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1 / ureg.hbar
# print(gamma_r.to('Hz'))
# print(gamma_r.to('hartree', 'sp'))
# print(gamma_r.to('1/amu/angstrom**2'))
# print('gamma_r = 2*pi*w_i/hbar     ', gamma_r.to('1/amu/bohr**2'))
# print('gamma_r = 2*pi*w_i/hbar     ', gamma_r.to_base_units())

Q_r = 1 * ureg.amu**(1/2) * ureg.bohr
# print('Q_r                         ', Q_r.to_base_units())

# dimensionless !!
# print('Q_r * gamma_r**0.5          ', (Q_r * gamma_r**0.5).to_base_units())

# print('\n----------------------------------------')

Vwhat = 1 * ureg.hartree / ureg.hbar / ureg.speed_of_light
Vau = 1 * ureg.hartree

# print(Vwhat)
# print(Vwhat.to_base_units())

# print('\n----------------------------------------')

fijk = 1 * ureg.cm**-1
Fijk = 1 * ureg.hartree / ureg.bohr**3 / ureg.amu**1.5

Fijk_hc = Fijk / ureg.hbar / ureg.speed_of_light
# print(Fijk_hc)
# print(Fijk_hc.to_base_units())
# print('\n----------------------')

# print(Fijk)
# print(Fijk.to_base_units())
# print(fijk.to_base_units())

cubicpart = Q_r**3 * Fijk
# print(cubicpart)
# print(cubicpart.to_base_units())
# print(cubicpart.to_reduced_units())

cubicpart2 = Q_r**3 * gamma_r**1.5 * fijk
# print(cubicpart2.to_base_units())

# print('\n----------------------------------------')
ureg.default_system = 'atomic'
# print('gamma_r**1.5              ', (gamma_r**1.5).to_base_units())
# print('gamma_r**1.5              ', (gamma_r**1.5).to('1 / bohr ** 3 / unified_atomic_mass_unit ** 1.5'))

# print(dir(ureg.sys))
# print(dir(ureg.sys.Planck))
ureg.default_system = 'Planck'
# print('gamma_r**1.5              ', (gamma_r**1.5).to_base_units())

# print('planck_length             ', (1*ureg.planck_length).to('m'))
# print('planck_mass               ', (1*ureg.planck_mass).to('kg'))

# print('bohr                      ', (1*ureg.bohr).to('m'))
# print('electron_mass             ', (1*ureg.electron_mass).to('kg'))

# print('unified_atomic_mass_unit  ', (1*ureg.unified_atomic_mass_unit).to('kg'))

# print('\n----------------------------------------')
# print('g16_coh2b3lypoptanhramanDZ\n')
from wilson.spectrum.callbacks2DIR import CFOURdata, GaussianData
gaussian_path = '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_coh2hfoptanhramanDZ.out'
q3 = '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_hfoptanhramanDZ_3q.out'
data = {'source': 'gaussian', 'type': 'log', 'files': {'log': gaussian_path, '3quanta': q3}}
gaussianparser = GaussianData(data)
# quit()

allstates_Gaussian, allstates_Gaussian_harm = gaussianparser.getAllStates()
funds = {k: v for k, v in allstates_Gaussian.items() if len(k) == 1}

sorted_data = {k: funds[k] for k in sorted(funds)}
freqs = np.array(list(sorted_data.values()))
print('freqs in cm-1           ', repr(freqs))

funds_harm = {k: v for k, v in allstates_Gaussian_harm.items() if len(k) == 1}

sorted_data_harm = {k: funds_harm[k] for k in sorted(funds_harm)}
freqs_harm = np.array(list(sorted_data_harm.values()))
print('freqs in cm-1 harmonic  ', repr(freqs_harm))

# print('yo', np.array(list(sorted_data.keys())))
FREQS_SQRT = np.sqrt(copy.deepcopy(freqs))
FREQS_SQRT = FREQS_SQRT.reshape(-1, 1)

MATRIX_SQRT = np.outer(FREQS_SQRT, FREQS_SQRT)


# print('FREQS_SQRT', FREQS_SQRT)
# print(MATRIX_SQRT)

# quit()

# print(sorted_data)
# for k in sorted_data:
#     print(k[0], sorted_data[k])
# {(0,): 2955.965, (1,): 1974.481, (2,): 1621.105, (3,): 1320.264, (4,): 3047.464, (5,): 1351.518}
#                            1         2         3         4         5
#                           B1        B2        A1        A1        A1
#        Frequencies ---  1186.5203 1253.0558 1515.3650 1832.9310 2864.9798
#     Reduced masses ---     1.3739    1.3484    1.1008    7.5797    1.0453

#                            6
#                           B2
#        Frequencies ---  2917.3250
#     Reduced masses ---     1.1213

#  ----+------+------+------+------+------+------+
#  (H) |     1|     2|     3|     4|     5|     6|
#  (A) |     4|     6|     3|     2|     1|     5|
#  ----+------+------+------+------+------+------+

reduced_masses = np.array([7.5797, 1.1213, 1.1008, 1.3484, 1.3739, 1.0453])

gamma_r_array = 2 * np.pi * ureg.speed_of_light * freqs * ureg.cm**-1 / ureg.hbar
solution77 = (gamma_r_array**1.5).to('1 / bohr ** 3 / unified_atomic_mass_unit ** 1.5')
# print(solution77)

dipole_derivs_Gaussian1, dipole_derivs_Gaussian2 = gaussianparser.getDipDers()

from scipy import constants
bohr_radius = constants.physical_constants['Bohr radius'][0]
debye_to_SI = 10**-21/constants.c
au_to_SI = constants.e * bohr_radius
debye_to_au = debye_to_SI / au_to_SI
dipders_massweighted = dipole_derivs_Gaussian1 * debye_to_au
np.set_printoptions(linewidth=350, threshold=sys.maxsize, suppress=True, precision=12)
print(dipole_derivs_Gaussian1)
# dipders_massweighted = np.array([[ 0.            ,  0.            ,  0.253349422057],
#        [-0.            ,  0.            , -0.395267588879],
#        [-0.            ,  0.            , -0.130921397358],
#        [ 0.047024752964, -0.            , -0.            ],
#        [ 0.            , -0.343984346677,  0.            ],
#        [-0.            ,  0.150708579333,  0.            ]])

# Reshape the vector v to a column vector
v = solution77[:, np.newaxis]
rm = reduced_masses[:, np.newaxis]
freqs_matrix = freqs[:, np.newaxis]

# print(v)
print('\nwhere we started - Gaussian dipole derivatives - dipders_massweighted')
print(dipders_massweighted)

print('\nwhere we going')
# muT_a_(4) = 1/sqrt(2) * P1_a_(4) * sqrt(h-bar/w(4))
# w(4) will be the angular frequency 2*pi*c*nu(4)
interm = dipders_massweighted/np.sqrt(2)*(constants.hbar/(2*np.pi*constants.c*v))
# P1 - e/amu^(1/2)
unitcheck = ureg.e / ureg.amu**(1/2) * np.sqrt(ureg.hbar / (2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1))
unitsF1 = np.sqrt(ureg.hbar / (2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1))
unitsF2 = np.sqrt(ureg.speed_of_light**4/(2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1)/(2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1)/ureg.hbar**2)
unitsF3 = ureg.hbar / (2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1)

ureg.default_system = 'SI'
# print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', np.sqrt(1* ureg.hbar/(2 * np.pi * constants.c)).to_base_units())
factorG = unitcheck.to('e*bohr').magnitude
# print('unitsF1', unitsF1.to_base_units())
# print('unitsF1', unitsF1.to('unified_atomic_mass_unit**0.5 * bohr'))

SQRT_HBAR_OVER_2PIC = (np.sqrt(ureg.hbar / (2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1))).to('unified_atomic_mass_unit**0.5 * bohr').magnitude
# print('\n             >>>>>   herehere  ', SQRT_HBAR_OVER_2PIC)


# quit()

# print('unitsF2', unitsF2.to_base_units())
# print('unitsF2', unitsF2.to('1/unified_atomic_mass_unit'))
# print('unitsF3', unitsF3.to_base_units())
# print('unitsF3', unitsF3.to('unified_atomic_mass_unit*bohr**2'))
# print('factorG', factorG, '\n')

# print((1*ureg.e).dimensionality)
# print('unitcheck  ', unitcheck)
# print('unitcheck  ', unitcheck.to_base_units())
# print('unitcheck  ', unitcheck.to('e*bohr'))

print('\n--------------------------------------------')
interm = dipders_massweighted*factorG/np.sqrt(2)/np.sqrt(freqs_matrix)
# print(interm)
RESULT1 = dipders_massweighted * SQRT_HBAR_OVER_2PIC / FREQS_SQRT / np.sqrt(2)
print(RESULT1)
print('SQRT_HBAR_OVER_2PIC', SQRT_HBAR_OVER_2PIC)
print('FREQS_SQRT\n', FREQS_SQRT)
print('dipders_massweighted * (np.sqrt(ureg.hbar / (2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1))) / FREQS_SQRT / np.sqrt(2)')

print('\n--------------------------------------------')
# quit()

basis = 'D'
vibdata_path = f'/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pV{basis}Z/out'
cubic_path = f'/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pV{basis}Z/cubic'
dipole_path = f'/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pV{basis}Z/dipole'
# polar_path = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVDZ/polar.pkl'

files = {'out': vibdata_path, 'cubic': cubic_path, 'dipolexyz': dipole_path}
data = {'source': 'cfour', 'type': 'out', 'files': files}
cfourparser = CFOURdata(data)

dipole_derivs_CFOUR1, dipole_derivs_CFOUR2 = cfourparser.getDipDers()


print('\nwhere we want to go - reduced normal coords CFOUR dipole derivatives')
print(dipole_derivs_CFOUR1)

# print(targedvals/interm)

print('\n---------------------------------------------------------------------')
from wilson.spectrum.callbacks2DIR import getDimensionlessNM
dimlessFile = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVDZ/QUADRATURE'
# print(dqs, type(dqs))

dqs = getDimensionlessNM(dimlessFile)
# print('\ndimensionless - QUADRATURE')
# print(dqs[8])
undispl = np.array([     [  -0.0000000000   ,     0.0000000000    ,    1.1177168336],
     [  -0.0000000000   ,     0.0000000000   ,    -1.1160727840],
     [  -0.0000000000   ,     1.7621743928   ,    -2.2250449092],
      [  0.0000000000   ,    -1.7621743928   ,    -2.2250449092]])
# print(undispl)


from scriptsHPC.utils import parseCFOUR
moldenfile = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVDZ/MOLDEN'
one, two, three = parseCFOUR.pMOLDEN(moldenfile)
# print('\nnon-mass-weighted - MOLDEN')
# print(three[8])
# print(one)

normcofile = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVDZ/NORMCO'
massweighted = parseCFOUR.pNORMCO(normcofile)
# print('\nmass-weighted - NORMCO')
# print(massweighted)

nm8 = np.array([ [ -0.0000000000    ,    0.2765787733    ,    0.0000000000],
     [   0.0000000000    ,   -0.4454289293   ,    -0.0000000000],
     [  -0.0000000000    ,    0.2175862526   ,     0.5614310717],
     [  -0.0000000000    ,    0.2175862526   ,    -0.5614310717]])
# print(nm8)

geo = np.array([   [  -0.0000000000   ,     0.0000000000   ,     4.4701567776 ],
                    [ -0.0000000000    ,    0.0000000000    ,   -3.8661895336 ],
                    [  -0.0000000000    ,    1.7690554959    ,   -2.2337334724  ],
                    [  0.0000000000    ,   -1.7690554959    ,   -2.2337334724 ] ])
# print(geo)


#  ----+------+------+------+------+------+------+
#  (H) |     1|     2|     3|     4|     5|     6|
#  (A) |     4|     6|     3|     2|     1|     5|
#  (A) |     7|     8|     9|    10|    11|    12|
#  ----+------+------+------+------+------+------+

print('\n---------------------------------------------------------------------')
print('Dipole moment second derivatives\n')
dipders_massweighted2 = dipole_derivs_Gaussian2 * debye_to_au

print('\nwhere we started - Gaussian dipole derivatives')
# print(dipders_massweighted2[1, :, :])
# print(dipders_massweighted2)

print('\nwhere we going')
print('\n--------------------------------------------')
interm = dipders_massweighted2*factorG**2/np.sqrt(2)/freqs_matrix
# print(interm[1, :, :])
# print(interm)
RESHAPED_FREQS = copy.deepcopy(FREQS_SQRT).reshape(6, 1, 1)
M_weighted = dipders_massweighted2 / RESHAPED_FREQS
M_weighted2 = M_weighted / copy.deepcopy(FREQS_SQRT).reshape(1, 6, 1)
RESULT2 = M_weighted2 * SQRT_HBAR_OVER_2PIC**2 / np.sqrt(2)
# print(RESULT2)
print('\n--------------------------------------------')

print('\nwhere we want to go - reduced normal coords CFOUR dipole derivatives')
# print(dipole_derivs_CFOUR2[1, :, :])
# print(dipole_derivs_CFOUR2)

print('\n---------------------------------------------------------------------')
print('CFF conversions\n')

a = np.sqrt(constants.h/constants.c/constants.physical_constants['unified atomic mass unit'][0]/100)
b = 10**10/2/np.pi/constants.physical_constants['Bohr radius'][0]/10**10
Fact3R = (constants.physical_constants['hartree-joule relationship'][0]/constants.h/constants.c/100) * (a * b)**3
Fact4R = (constants.physical_constants['hartree-joule relationship'][0]/constants.h/constants.c/100) * (a * b)**4
print("a = np.sqrt(constants.h/constants.c/constants.physical_constants['unified atomic mass unit'][0]/100)")
print("b = 10**10/2/np.pi/constants.physical_constants['Bohr radius'][0]/10**10")
print('(a * b)/10**10  ', (a * b)/10**10)
print('(a * b)/10**30  ', ((a * b)/10**10)**3)
print('(a * b)         ', a * b)
print('ToJ/h/c/100     ', constants.physical_constants['hartree-joule relationship'][0]/constants.h/constants.c/100)
print('Fact3R          ', Fact3R)

print('\n---------------------------------------------------------------------')
print('unified atomic mass unit    ', constants.physical_constants['unified atomic mass unit'][0])
print('Bohr radius                 ', constants.physical_constants['Bohr radius'][0])
print('h                           ', constants.h)
print('c                           ', constants.c)
print('hartree-joule relationship  ', constants.physical_constants['hartree-joule relationship'][0])
print('electron mass               ', constants.physical_constants['electron mass'][0])

print('\n---------------------------------------------------------------------')
print(constants.e)
smth = constants.e**4 * constants.physical_constants['Bohr radius'][0]**4/constants.physical_constants['hartree-joule relationship'][0]**3/10**(-7*3)

smthUnits = 1. * ureg.e ** 4 * ureg.bohr ** 4 / ureg.hartree ** 3
print(smthUnits)

print(smth)


quit()


fc4_889 = -108.7633403123
# -107.51393     -0.53215     -0.01809
fg16_366 = -107.51393
# {(0,): 2706.412, (1,): 1808.055, (2,): 1483.967, (3,): 1167.449, (4,): 2651.062, (5,): 1233.413}

conv_fg16 = -0.01809 / np.sqrt(1483.967*1233.413**2) * Fact3R/10**30
print('conv_fg16', conv_fg16)
# -110.39225161467601

# 4      4      2           -27.52707     -0.14189     -0.00482
fg16_244 = -27.52707
conv_fg16 = -0.00482 / np.sqrt(1808.055*1167.449**2) * Fact3R/10**30
print('conv_fg16', conv_fg16)
# -28.15295394818791

# 6      5      1            -8.50829     -0.08835     -0.00300
fg16_156 = -8.50829
conv_fg16 = -0.00300 / np.sqrt(2706.412*2651.062*1233.413) * Fact3R/10**30
print('conv_fg16', conv_fg16)
# -9.246572182368077

reverse_conv = -107.51393 * np.sqrt(1483.967*1233.413**2) / Fact3R * 10**30
print('reverse_conv', reverse_conv)
# -0.01809

print('\n---------------------------------------------------------------------')

print('cfour quadratic force constants')
print('cfourdata/hcoh/HFcc_pVQZ\n')
#   7    7    7    7               0.0659236007          117.1411589855
#   8    8    7    7               0.0146126415           25.3200582441
#  12    8    7    7              -0.0020959839           -2.3959069767

#          Harmonic     Fundamental Anharmonic    Harmonic   Fundamental  Anharm
#   Mode   Frequency    Frequency   Contribution  Intensity  Intensity    Contrib
#     7   1338.0802   1320.8691      -17.2111      2.7409       2.9397      0.1988
#     8   1372.1917   1352.0687      -20.1230     21.6589      23.0811      1.4222
#     9   1652.1525   1620.2480      -31.9046     18.3353      17.0972     -1.2380
#    10   1996.5288   1971.3924      -25.1364    157.4113     164.5101      7.0987
#    11   3084.5396   2954.1708     -130.3688     64.9166      65.3682      0.4516
#    12   3152.9908   3016.5630     -136.4279    104.6780      98.5233     -6.1547

print('7_7_7_7', 0.0659236007/np.sqrt(1338.0802**4)*Fact4R, 117.1411589855)
print('8_8_7_7', 0.0146126415/np.sqrt(1338.0802**2*1372.1917**2)*Fact4R, 25.3200582441)
print('12_8_7_7', -0.0020959839/np.sqrt(1338.0802**2*1372.1917*3152.9908)*Fact4R, -2.3959069767)
print('\n==========================\n')
print('7_7_7_7', 117.1411589855*np.sqrt(1338.0802**4)/Fact4R, 0.0659236007)
print('8_8_7_7', 25.3200582441*np.sqrt(1338.0802**2*1372.1917**2)/Fact4R, 0.0146126415)
print('12_8_7_7', -2.3959069767*np.sqrt(1338.0802**2*1372.1917*3152.9908)/Fact4R, -0.0020959839)

print('\ndftGaussian/formaldehyde/g16_coh2hfoptanhramanQZ.out\n')
#       1      1      1      1    500.69864     83.24949      1.49736
#       2      1      1      1     -3.28330     -0.43920     -0.00790
#       2      2      1      1    -25.07870     -2.69895     -0.04854
#       2      2      2      1    -11.17135     -0.96725     -0.01740
#       2      2      2      2    155.93853     10.86248      0.19538
#       3      1      1      1     -1.22591     -0.14917     -0.00268
#       3      2      1      1     61.16985      5.98846      0.10771

print('1_1_1_1', 1.49736/np.sqrt(3084.5423**4)*Fact4R, 500.69864)
print('2_1_1_1', -0.00790/np.sqrt(3084.5423**3*1996.5300)*Fact4R, -3.28330)
print('2_2_1_1', -0.04854/np.sqrt(3084.5423**2*1996.5300**2)*Fact4R, -25.07870)
print('2_2_2_1', -0.01740/np.sqrt(3084.5423*1996.5300**3)*Fact4R, -11.17135)

print('\n==========================\n')

print('1_1_1_1', 500.69864*np.sqrt(3084.5423**4)/Fact4R, 1.49736)
print('2_1_1_1', -3.28330*np.sqrt(3084.5423**3*1996.5300)/Fact4R, -0.00790)
print('2_2_1_1', -25.07870*np.sqrt(3084.5423**2*1996.5300**2)/Fact4R, -0.04854)
print('2_2_2_1', -11.17135*np.sqrt(3084.5423*1996.5300**3)/Fact4R, -0.01740)

