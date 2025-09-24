from .wilson_main import abstractions as abst_main
from .fixtures import mol_system
from .calcsetups import calc_setup

vibanasetup_none = abst_main.VibAnaSetup(system=mol_system, regime='GVPT2', 
                                         vibana_own_analysis='none')
vibanasetup_anharm = abst_main.VibAnaSetup(system=mol_system, regime='GVPT2', 
                                           vibana_own_analysis='anharm')