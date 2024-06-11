#!/usr/bin/env python
import sys
import numpy as np
np.set_printoptions(linewidth=350, threshold=sys.maxsize, suppress=True, precision=12)

import copy
from pint import UnitRegistry
ureg = UnitRegistry()
# mu - e·a_0
# alpha - a_0**3
# gamma_electric = 1/(ureg.cm**-2) * (1/(ureg.cm**-2) * (ureg.e * ureg.bohr)**2 * ureg.bohr**3).to('cm**2/(erg*bohr**2)').magnitude
gamma_electric = 1/(ureg.cm**-2) * (1/(ureg.cm**-2) * (ureg.e * ureg.bohr)**2 * ureg.bohr**3)
# gamma_electric = 1/(ureg.cm**-2) * (1/(ureg.cm**-2) * (ureg.e * ureg.bohr)**2 * ureg.bohr**3).dimensionality
print(gamma_electric)
print(gamma_electric.dimensionality)

gamma_mechanical = 1/(ureg.cm**-6) * (ureg.e * ureg.bohr)**2 * ureg.bohr**3 * 1 * ureg.cm**-1
print(gamma_mechanical)
print(gamma_mechanical.dimensionality)

paper = 1 * ureg.cm**3 / ureg.erg
print(paper)
print(paper.dimensionality)
print(paper.to_base_units())
# print(paper.to('cm**4', 'sp'))

eqs = 1 * ureg.cm**4 * ureg.e**2 * ureg.bohr**5
print('\n', eqs)
print(eqs.dimensionality)
# print(eqs.to('cm**5*e**2*bohr**5', 'sp'))
# print(eqs.to('cm**5*e**2*bohr**5'))

#     values_in_wavenumbers = values * constants.hartree2J / (4 * np.pi ** 2 * constants.c ** 2 * constants.amu2kg
#                                                             * constants.bohr2cm ** 2)
#     values_in_wavenumbers = np.sqrt(np.absolute(values_in_wavenumbers))

print('\n------------------------------')
vv = (1 * ureg.hartree / ureg.amu / ureg.bohr**2) / (4 * np.pi**2 * ureg.c**2)
print('np.sqrt(vv)', np.sqrt(vv))
print('np.sqrt(vv)', np.sqrt(vv).dimensionality)
# omega**2 = (2pi*c*nu)**2
print('(1 * ureg.hartree / ureg.amu / ureg.bohr**2)', (1 * ureg.hartree / ureg.amu / ureg.bohr**2).dimensionality)
print('(1 * ureg.hartree / ureg.bohr**2)', (1 * ureg.hartree / ureg.bohr**2).dimensionality)
print(np.sqrt(vv).to_base_units(), '\n')
print(np.sqrt(vv).to_reduced_units())
print(np.sqrt(vv).to('cm**-1'))

print('\n------------------------------')
#  ureg.amu*ureg.bohr**2*
vvconv = (1 / ureg.m)**2 * (4 * np.pi**2 * ureg.c**2) * ureg.kg**2 / ureg.J**2 *  ureg.m**2
print(vvconv)
print(vvconv.dimensionality)
# print(vvconv.to('hartree/(amu*bohr**2)'))

si = 1 * ureg.C * ureg.m**2 / ureg.V
print('\n', si)
print(si.dimensionality)
print(si.to_base_units())
# ureg.default_system = 'cgs'
# print(si.to_base_units())

print('hello', si.to('e**2*bohr**2/hartree'))
from scipy.constants import *
factorPol = 10**6/4/np.pi/epsilon_0
facVol = 4*np.pi*epsilon_0
print((facVol*ureg.m**3).to('bohr**3'))

print('-------')
print((1*ureg.bohr).to('m'))
print((1*ureg.bohr**3).to('m**3'))
si_pol = (1*ureg.bohr**3).to('m**3')*4*np.pi*epsilon_0
print(si_pol.magnitude*(1 * ureg.C * ureg.m**2 / ureg.V).to('e**2*bohr**2/hartree'))
# print()

print('-------')
au= 1 * ureg.e**2 * ureg.bohr**2 / ureg.hartree
print(au.dimensionality)
# print(epsilon_0)
# print(factorPol)
# print(si.to('e**2*bohr**2/hartree')/factorPol)
# print(1/factorPol*si.to('e**2*bohr**2/hartree').magnitude)


