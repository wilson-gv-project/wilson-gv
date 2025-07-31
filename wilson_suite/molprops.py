import copy
import wilson_main.abstractions as abst_main
from .calcsetups import calc_setup, calc_setup1, calc_setup2

props_evv_none_nocalc_novals = [abst_main.MolecularProperty(prop_spec={'ops': ('g', 'f'), 'freq': (0.0, 0.0)}, trivial_name='dipgrad'), 
                      abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'f', 'f'), 'freq': (0.0, 0.0, 0.0, 0.0)}, trivial_name='polhess',), 
                      abst_main.MolecularProperty(prop_spec={'ops': ('g', 'f', 'f'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='polgrad'), 
                      abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'f'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='diphess'), 
                      abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'g'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='cff')]

props_evv_anharm_nocalc_novals = [abst_main.MolecularProperty(prop_spec={'ops': ('g', 'f'), 'freq': (0.0, 0.0)}, trivial_name='dipgrad'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'f', 'f'), 'freq': (0.0, 0.0, 0.0, 0.0)}, trivial_name='polhess'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'f', 'f'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='polgrad'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'f'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='diphess'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'g'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='cff'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'g', 'g'), 'freq': (0.0, 0.0, 0.0, 0.0)}, trivial_name='qff'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('r',), 'freq': 0.0}, trivial_name='B'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'r'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='coriolis')]

props_evv_anharm_wcalc_novals = copy.deepcopy(props_evv_anharm_nocalc_novals)


for prop in props_evv_anharm_wcalc_novals:
    prop.addCalcSetup(calc_setup)

eval_prop_specify = {'cff': calc_setup2, 'qff': calc_setup2, 
                     'dipgrad': calc_setup1, 'diphess': calc_setup2, 
                     'polgrad': calc_setup1, 'polhess': calc_setup2}
