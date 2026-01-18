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