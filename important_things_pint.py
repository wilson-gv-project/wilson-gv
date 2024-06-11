#!/usr/bin/env python
# import copy
import numpy as np

from pint import UnitRegistry
ureg = UnitRegistry()
# import sys
from scipy import constants


bohr_radius = constants.physical_constants['Bohr radius'][0]
debye_to_SI = 10**-21/constants.c
au_to_SI = constants.e * bohr_radius
debye_to_au = debye_to_SI / au_to_SI
print('debye_to_au: ', debye_to_au)

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
print('smthUnits:                  ', smthUnits)

ureg.default_system = 'cgs'
print('smthUnits in cgs units:     ', smthUnits.to_base_units())
ureg.default_system = 'SI'
print('smthUnits in SI units:      ', smthUnits.to_base_units())

print('smthUnits dimensionality:   ', smthUnits.dimensionality)
print((1*ureg.N_A).to_base_units())
print(smth)
print('\n---------------------------------------------------------------------')

Kwak_chi = 1. * ureg.cm ** 3 / ureg.erg
print('Kwak_chi:                   ', Kwak_chi)
print('Kwak_chi in SI units:       ', Kwak_chi.to_base_units())
print('Kwak_chi dimensionality:    ', Kwak_chi.dimensionality)

print('planck constant:            ', (1*ureg.planck_constant).to_base_units())

print('\n---------------------------------------------------------------------')

ok = smthUnits*1.*ureg.planck_constant**2
print('ok:                         ', ok)
print('ok in SI units:             ', ok.to_base_units())
print('ok dimensionality:          ', ok.dimensionality)

print('\n---------------------------------------------------------------------')
gammaAsresponse = 1. * ureg.coulomb * ureg.m / (ureg.volt/ureg.m)**3
print('gammaAsresponse:            ', gammaAsresponse)
print('gammaAsresponse in SI units:', gammaAsresponse.to_base_units())

print('\n---------------------------------------------------------------------')

gamma_au = 1* ureg.e**4*ureg.bohr**4/ureg.hartree**3
print('gamma_au:                   ', gamma_au)
print('gamma_au in SI units:       ', gamma_au.to_base_units())
print('gamma_au dimensionality:    ', gamma_au.dimensionality)
ureg.default_system = 'cgs'
print('gamma_au in cgs units:      ', gamma_au.to_base_units())
ureg.default_system = 'SI'
# print(gamma_au.to('cm^3/erg'))
print('\n---------------------------------------------------------------------')

draft = 1*ureg.statC**4*ureg.cm**4/ureg.erg**3
print('draft:                      ', draft)
print('draft in SI units:          ', draft.to_base_units())

# print('draft in cm^3/erg:          ', draft.to('cm^3/erg'))
print('\n---------------------------------------------------------------------')
print((1*ureg.statC*ureg.statV).to_base_units())
ureg.default_system = 'atomic'
print((1*ureg.statC*ureg.statV).to_base_units())

print('\n---------------------------------------------------------------------')
print('dfaft in atomic units:      ', draft.to_base_units())
ureg.default_system = 'cgs'
print('draft in cgs units:         ', draft.to_base_units())
print('draft dimensionality:       ', draft.dimensionality)

# print('draft in cgs units:         ', draft.to('cm^3/erg'))
print('\n---------------------------------------------------------------------')

new = 1*ureg.hbar/ureg.hartree**0.5/ureg.bohr**2
print('new:                        ', new)
print('new in SI units:            ', new.to_base_units())
ureg.default_system = 'atomic'
print('new in atomic units:        ', new.to_base_units())

print('\n---------------------------------------------------------------------')

hbar = (1*ureg.hbar).to('hartree s')
print('hbar:                       ', hbar)

hbaroveromega = hbar/(1/ureg.s)
print('hbaroveromega:              ', hbaroveromega)

print('\n---------------------------------------------------------------------')

from scipy import constants
hartree2J = constants.physical_constants['hartree-joule relationship'][0]

t = 100 * constants.h * constants.c / hartree2J
print('conversion factor rcm_rs from cm^-1 to s^-1:  ', t)
print('1/rcm_rs', 1/t)
