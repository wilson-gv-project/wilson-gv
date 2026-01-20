import wilson_suite as ws
from ...fixtures import evv_experiment

def test_find_props():
    experiment_a = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=experiment_a)
    props = ws.main.main_functions.find_props(terms=terms)

    for p in props:
        print(p)

def test_find_residual_vib_info():

    vib_ana = ws.main.abstractions.VibAnaSetup(system='', regime='GVPT2', vibana_own_analysis='anharm')
    props = ws.main.main_functions.find_residual_vib_info(vib_ana=vib_ana)

    for p in props:
        print(p)

def test_find_props_and_max_state_lvl():
    print()
    vib_ana = ws.main.abstractions.VibAnaSetup(system='', regime='GVPT2', vibana_own_analysis='anharm')
    
    experiment_a = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=experiment_a)
    props, _, _ = ws.main.main_functions.find_props_and_max_state_lvl(terms=terms, vib_ana=vib_ana)
    print([k.h(1) for k in props], '\n')
    print(set([k.h(1) for k in props]), '\n')
    
    for p in props:
        print(p)
    if not 'f' in 'fjk':
        print()

def test_get_data_for_vibanalysers():
    print()
    vib_ana = ws.main.abstractions.VibAnaSetup(system='', regime='GVPT2', vibana_own_analysis='anharm')

    from wilson_suite.wilson_utils.paths import SUITE_ROOT
    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                     lvl_theory='HF', 
                                                     basis_set='STO-3G', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_h2o_HF_STO3G.out')
    
    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    vib_ana, props = ws.main.main_functions.get_data_for_vibanalysers(vib_ana=vib_ana, 
                                                     calc_setup=calc_setup, 
                                                     obtainer=wilson_data_obtainer)
    
    states, diagn = ws.intensities.anharmonic_treatment.anharm_analyzer_data(props=props,
                                                                             nc_sqrt_eigval=vib_ana.nc_sqrt_eigval,
                                                                             regime=vib_ana.regime,
                                                                             exclude_modes=None)
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    for k,v in st_dict.items():
        print(k.ljust(10), v)
    print(diagn)

    states_1quantum_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states if len(list(s.harm_quanta_coeffs.keys())[0])==1}
    print('\n1 quantum levels')
    for k,v in vib_ana.nc_sqrt_eigval.items():
        print(k, '--', v, '--', states_1quantum_corrected[k])