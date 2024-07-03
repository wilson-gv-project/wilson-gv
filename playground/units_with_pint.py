#!/usr/bin/env python
from pint import UnitRegistry
ureg = UnitRegistry()
meterU = ureg.meter

length = 30.0 * meterU

# print(meterU)
# print(length)


volume = 4.3 * ureg.gal

# Returns True
vchk = volume.check('[volume]')
# print(vchk)
# print(volume.dimensionality)

Q_ = ureg.Quantity
cff_cm1 = Q_(214.0205431678, ureg.centimeter**(-1))

print(Q_(cff_cm1, ureg.centimeter**(-1)))


# Initialize a unit registry
# ureg = UnitRegistry()
# Define the conversion factor from cm⁻¹ to Hartrees
# This factor is based on the relation: 1 cm⁻¹ ≈ 4.5563e-6 Hartrees
# ureg.define('hartree = 4.3597447222071e-18 joule')  # Define Hartree in terms of Joules
# ureg.define('cm1_to_hartree = 4.5563e-6 hartree')  # Define the conversion factor
# Define your quantity in cm⁻¹
# force_constant_cm = 118.0273541127 * ureg('cm^-1')
# Convert to Hartrees
# force_constant_hartree = force_constant_cm.to('hartree')
# print(f"Force constant in atomic units (Hartrees): {force_constant_hartree}")


wavelength = 1550 * ureg.nm
frequency = (ureg.speed_of_light / wavelength).to('Hz')
print(frequency.dimensionality)
print(frequency.to_compact())

print(cff_cm1.to('Eh', 'spectroscopy'))
Vddhc = 1 * ureg.hartree / ureg.hbar / ureg.speed_of_light
Fijk = 1 * ureg.cm ** -1
dq = 1
prod = Fijk * dq**3


import numpy as np
plankh = 2*np.pi * ureg.hbar
print(plankh.to('J*s'))

print(Vddhc.to('cm^-1', 'sp').dimensionality)
print(Vddhc.dimensionality)

print('--------------------\n', prod.dimensionality)

lambdar = 4*np.pi**2*ureg.speed_of_light**2*(200*ureg.cm**-1)**2
print('lambdar           ', lambdar.to_base_units())
print('lambdar           ', lambdar.to_reduced_units())
# print(lambdar.dimensionality)


dQ = np.sqrt(1*ureg.amu) * ureg.bohr
print('dQ                ', dQ.to_base_units())
print('dQ                ', dQ.to_reduced_units())
# print(help(lambdar))
quadratic = lambdar * dQ.to_base_units() ** 2
print('quadratic         ', quadratic.to_base_units())
print('quadratic         ', quadratic.to_reduced_units())
# print('quadratic', quadratic.to('hartree'))

cubic = ureg.Quantity(0.005, 'hartree')
x= cubic / dQ**3
print('x                 ', x.to_base_units())
print('x                 ', x.to_reduced_units())
cubicrcm = 2000*ureg.cm**-1
fijk = 1 * ureg.cm**-1
# cubicrcm = fijk*xx**3*dQ**3
xx = (cubicrcm/fijk/dQ**3)**(1/3)
print('scaling to dimless', xx.to_reduced_units())
q = ureg.amu**(1/2)*ureg.bohr
omega = 2000 * ureg.cm**-1
# with ureg.context('spectroscopy'):
#      q1.to('Hz')
#      q2.to('Hz')
bohr2ang = ureg.Quantity(1, 'bohr').to('angstrom')
print(bohr2ang)
hartree2attoj = ureg.Quantity(1, 'hartree').to('attojoule')
print(hartree2attoj)
cubicau = cubicrcm * np.sqrt(omega**3) * bohr2ang**3 #/ hartree2attoj / 9.85501E+06
print(cubicau)
print(cubicau.to_reduced_units())
scl = (2*np.pi*ureg.speed_of_light*omega/ureg.hbar)**0.5
print(scl.to_base_units())
print(scl.to('1/amu**0.5/bohr'))
print(cubicrcm*scl.to('1/amu**0.5/bohr'))
# print(cubicau.to('hartree', 'sp'))
# print(cubicau.to_reduced_units().to('bohr**0.5 / hartree'))
print((2*np.pi*ureg.hbar*ureg.speed_of_light).dimensionality)
# print((2*np.pi*ureg.hbar*ureg.speed_of_light).to('hartree', 'sp'))
# hartree = hartree/bohr**3 * amu**1.5*bohr**3

print((cubicrcm.to('Hz', 'sp')*2*np.pi*ureg.hbar).to('hartree', 'sp'))


print('\n----------------------------------------')
gamma_r = 2 * np.pi * ureg.speed_of_light * 1 * ureg.cm**-1 / ureg.hbar
# print(gamma_r.to('Hz'))
# print(gamma_r.to('hartree', 'sp'))
print(gamma_r.to('1/amu/angstrom**2'))
print('gamma_r = 2*pi*w_i/hbar     ', gamma_r.to('1/amu/bohr**2'))
print('gamma_r = 2*pi*w_i/hbar     ', gamma_r.to_base_units())

Q_r = 1 * ureg.amu**(1/2) * ureg.bohr
print('Q_r                         ', Q_r.to_base_units())

# dimensionless !!
print('Q_r * gamma_r**0.5          ', (Q_r * gamma_r**0.5).to_base_units())

print('\n----------------------------------------')

Vwhat = 1 * ureg.hartree / ureg.hbar / ureg.speed_of_light
Vau = 1 * ureg.hartree

print(Vwhat)
print(Vwhat.to_base_units())

print('\n----------------------------------------')

fijk = 1 * ureg.cm**-1
Fijk = 1 * ureg.hartree / ureg.bohr**3 / ureg.amu**1.5

Fijk_hc = Fijk / ureg.hbar / ureg.speed_of_light
print(Fijk_hc)
print(Fijk_hc.to_base_units())
print('\n----------------------')

print(Fijk)
print(Fijk.to_base_units())
# print(fijk.to_base_units())

cubicpart = Q_r**3 * Fijk
# print(cubicpart)
print(cubicpart.to_base_units())
print(cubicpart.to_reduced_units())

cubicpart2 = Q_r**3 * gamma_r**1.5 * fijk
print(cubicpart2.to_base_units())

print('\n----------------------------------------')
ureg.default_system = 'atomic'
print('gamma_r**1.5              ', (gamma_r**1.5).to_base_units())
print('gamma_r**1.5              ', (gamma_r**1.5).to('1 / bohr ** 3 / unified_atomic_mass_unit ** 1.5'))

# print(dir(ureg.sys))
# print(dir(ureg.sys.Planck))
ureg.default_system = 'Planck'
print('gamma_r**1.5              ', (gamma_r**1.5).to_base_units())

print('planck_length             ', (1*ureg.planck_length).to('m'))
print('planck_mass               ', (1*ureg.planck_mass).to('kg'))

print('bohr                      ', (1*ureg.bohr).to('m'))
print('electron_mass             ', (1*ureg.electron_mass).to('kg'))

print('unified_atomic_mass_unit  ', (1*ureg.unified_atomic_mass_unit).to('kg'))

print('\n----------------------------------------')

freqs = np.array([2955.965, 1974.481, 1621.105, 1320.264, 3047.464, 1351.518])
# {(0,): 2955.965, (1,): 1974.481, (2,): 1621.105, (3,): 1320.264, (4,): 3047.464, (5,): 1351.518}
print('freques in cm-1', freqs)
#                          1         2         3         4         5
#                           B1        B2        A1        A1        A1
#        Frequencies ---  1198.1582 1262.8944 1530.1732 1813.0502 2884.6437
#     Reduced masses ---     1.3674    1.3428    1.1062    7.4759    1.0425

#                            6
#                           B2
#        Frequencies ---  2939.3539
#     Reduced masses ---     1.1210

#  ----+------+------+------+------+------+------+
#  (H) |     1|     2|     3|     4|     5|     6|
#  (A) |     4|     6|     3|     2|     1|     5|
#  ----+------+------+------+------+------+------+

reduced_masses = np.array([1.0425, 7.4759, 1.1062, 1.3674,  1.1210,   1.3428  ])

gamma_r_array = 2 * np.pi * ureg.speed_of_light * freqs * ureg.cm**-1 / ureg.hbar
solution77 = (gamma_r_array**1.5).to('1 / bohr ** 3 / unified_atomic_mass_unit ** 1.5')
print(solution77)

dipders_massweighted = np.array([[ 0.            ,  0.            ,  0.253349422057],
       [-0.            ,  0.            , -0.395267588879],
       [-0.            ,  0.            , -0.130921397358],
       [ 0.047024752964, -0.            , -0.            ],
       [ 0.            , -0.343984346677,  0.            ],
       [-0.            ,  0.150708579333,  0.            ]])

# Reshape the vector v to a column vector
v = solution77[:, np.newaxis]
rm = reduced_masses[:, np.newaxis]
print('\nwhere we started')
print(dipders_massweighted)

print('\nwhere we going')
# interm = dipders_massweighted/np.sqrt(rm)
interm = dipders_massweighted/np.sqrt(rm)/33.715258*v
# print(interm)
print(interm.magnitude)
# print(dipders_massweighted/np.sqrt(rm))

targedvals = np.array([[ 0.0141179696,  0.          ,  0.          ],
       [ 0.          , -0.0446649094,  0.          ],
       [ 0.          ,  0.          , -0.0353541339],
       [ 0.          ,  0.          , -0.0969882966],
       [ 0.          ,  0.          , -0.0500448721],
       [ 0.          , -0.0672003057,  0.          ]])

print('\nwhere we want to go')
print(targedvals)

# print(targedvals/interm)

print('\n---------------------------------------------------------------------')
from wilson.spectrum.callbacks2DIR import getDimensionlessNM
dimlessFilepkl = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVQZ/dimensionless.pkl'
dimlessFile = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVQZ/QUADRATURE'
dqs = getDimensionlessNM(dimlessFilepkl)
# print(dqs, type(dqs))

dqs = getDimensionlessNM(dimlessFile)
print(dqs[8])

from scriptsHPC.utils import parseCFOUR
moldenfile = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVQZ/MOLDEN'
one, two, three = parseCFOUR.pMOLDEN(moldenfile)
print(three[8])

normcofile = ''