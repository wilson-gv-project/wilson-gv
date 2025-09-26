from .wilson_main import abstractions as abst_main

# QC calculations/vibana parameters
calc_setup = abst_main.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc-pVQZ')
calc_setup1 = abst_main.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc-pVTZ')
calc_setup2 = abst_main.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc-pVDZ')
