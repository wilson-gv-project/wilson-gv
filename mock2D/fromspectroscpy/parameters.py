# SpectroscPy x.x.x
# SpectroscPy is a script package developed by and containing contributions from

    # Karen Oda Hjorth Minde Dundas
    # Magnus Ringholm
    # Yann Cornation
    # Benedicte Ofstad

# The package is released under a LGPL licence.
# For questions, please contact on karen.o.dundas@uit.no

from math import sqrt

# Conversion factors
hartree_to_coulomb     = 1.60217656535e-19
bohr_to_meter          = 5.29177210922e-11
hartree_to_joule       = 4.3597441775e-18

# Constants
vacuum_permittivity    = 8.854187817e-12
hbar                   = 1.054571726e-34
avogadros_constant     = 6.02214075862e23
plancs_constant        = 6.62607015e-34
speed_of_light         = 299792458
boltzmann_constant_SI  = 1.38064852e-23
sqrt2                  = sqrt(2)

# mass of one electron in kg, or in other words conversion factor from au to kg
electron_mass = 9.1093829140e-31
# (1/12)th of the mass of a carbon atom in kg, or in other words conversion from amu to kg
one_twelfth_carbon = 1.66053904020e-27

#DALTON
debye_to_coulomb_meter = 10**(-21)/speed_of_light
angstrom_to_meter      = 1.0e-10

# Here converted from 1/cm to J since that is what is needed in the routine.
fermi_threshold  = 200.0*100*plancs_constant*speed_of_light
martin_threshold = 1.0*100*plancs_constant*speed_of_light

hess_operator = ['GEO', 'GEO']
cubic_operator = ['GEO', 'GEO', 'GEO']
quartic_operator = ['GEO', 'GEO', 'GEO', 'GEO']

dip_grad_operator = ['GEO', 'EL']
dip_hess_operator = ['GEO', 'GEO', 'EL']
dip_cubic_operator = ['GEO', 'GEO', 'GEO', 'EL']

polariz_grad_operator = ['GEO', 'EL', 'EL']
polariz_hess_operator = ['GEO', 'GEO', 'EL', 'EL']
polariz_cubic_operator = ['GEO', 'GEO', 'GEO', 'EL', 'EL']

hyper_polariz_grad_operator = ['GEO', 'EL', 'EL', 'EL']
hyper_polariz_hess_operator = ['GEO', 'GEO', 'EL', 'EL', 'EL']
hyper_polariz_cubic_operator = ['GEO', 'GEO', 'GEO', 'EL', 'EL', 'EL']

