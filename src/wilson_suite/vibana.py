from .wilson_main import abstractions as abst_main
from .fixtures import mol_system
from .calcsetups import calc_setup

vibanasetup_none = abst_main.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_prop_need='none',
                                         allow_skip_eigvec=True, external_fill_from=calc_setup)
vibanasetup_anharm = abst_main.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_prop_need='anharm',
                                           allow_skip_eigvec=True, external_fill_from=calc_setup)