import pytest

from ....wilson_utils.printing import printtest, separatorprint
from ....fixtures import evv_experiment
from ....wilson_system.system_data import DataOriginInfo, MolSystemData
import wilson_suite as ws
from importlib.resources import files
data_dir = files('wilson_suite').joinpath('data_for_tests')

WATER = DataOriginInfo(source_type='gaussian', lvl_theory='HF', basis_set='STO-3G',
                       base_file_loc=data_dir / 'g16_h2o_HF_STO3G.out')
VPT2_INPUTS = ['cff_rc', 'qff_rc', 'B', 'coriolis']     # force constants in cm-1, rotational, Coriolis constants

def test_anharm_analyzer_wilsonsim():
    """
    Anharmonic analyzer (using vpt2.py module) integration test

    to put back in future:
    import logging
    from ....wilson_utils.logger import setup_logger
    setup_logger("wilson_suite.", level=logging.DEBUG)
    logging.getLogger('wilson_suite.').setLevel(logging.DEBUG)
    """
    separatorprint()

    from .... import wilson_main as ws_main

    mol_system = ws_main.abstractions.MolecularSystem(name='h2o', natoms=3)
    vib_ana = ws_main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='anharm')

    printtest(f'vibana.vibana_own_analysis: {vib_ana.vibana_own_analysis}')
    experiment_a = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=experiment_a)
    axes_choice = experiment_a.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}
    calc_setup = ws_main.abstractions.DataOriginInfo(source_type='gaussian',
                                                     lvl_theory='HF', 
                                                     basis_set='STO-3G', 
                                                     base_file_loc=data_dir / 'g16_h2o_HF_STO3G.out')

    sim = ws_main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(experiment_a)
    sim.addTerms(terms=terms)
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana_setup=vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)

    # should be careful with props, because props are needed for vib analyzer
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    printtest(f'[i.triv_name for i in sim.props] {[i.trivial_name for i in sim.props]}')    
    sim.dressPropsWithSetup()

    sim.setAxisChoiceAndTranslateTerms(axes_choice)
    
    from wilson_suite.wilson_system.data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)
        
    print('\n')
    printtest(f'nc_sqrt_eigval: {sim.vib_ana_setup.nc_sqrt_eigval}') # vibana_own_analysis='all' -> nc_sqrt_eigval is None
    print('\n')

    states, diagn = ws.intensities.anharmonic_treatment.anharm_analyzer_data(props=sim.props,
                                                                             nc_sqrt_eigval=sim.vib_ana_setup.nc_sqrt_eigval,
                                                                             regime=sim.vib_ana_setup.regime,
                                                                             exclude_modes=None)
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    for k,v in st_dict.items():
        print(k.ljust(10), v)
    print(diagn)
    
'''
anharm_analyzer_data() returns list[Vibstate] and diagn dict. Vibstates energies are with anharmonic corrections

anharm_analyzer_data() needs:
    - props as MolPropsCollection: cff_rc, qff_rc (cm-1), B, coriolis
    - nc_sqrt_eigval as dict
    - regime info - str
    - exclude_modes list if needed
'''

def test_anharm_analyzer_vibana():
    """
    Water, read with MolSystemData.load. GVPT2 moves the fundamentals away from the harmonic
    frequencies, to the anharmonic fundamentals the file prints (3 decimals).
    """
    separatorprint()

    molsys = MolSystemData.load(WATER, VPT2_INPUTS)
    printtest(f'eigenvals: {molsys.eigenvals}')

    # ---- do analysis
    states, diagn = ws.intensities.anharmonic_treatment.anharm_analyzer_data(props=molsys.mol_props,
                                                                             nc_sqrt_eigval=molsys.eigenvals,
                                                                             regime='GVPT2',
                                                                             exclude_modes=None)
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    states_1quantum_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states if len(list(s.harm_quanta_coeffs.keys())[0])==1}
    assert molsys.eigenvals != states_1quantum_corrected

    file_fundamentals = {k: molsys.states['anharmonic'].get_state_by_label(str(k)).energy for k in molsys.eigenvals}
    assert states_1quantum_corrected == pytest.approx(file_fundamentals, abs=2e-3)

    for k,v in st_dict.items():
        print(k.ljust(10), v)
    print(diagn)

    print('\n1 quantum levels')
    for k,v in molsys.eigenvals.items():
        print(k, '--', v, '--', states_1quantum_corrected[k])



def test_anharm_analyzer_vibana_excludemodes():
    """
    Water, read with MolSystemData.load, GVPT2 with mode 1 left out: no state of mode 1 comes back,
    and the diagnostics name the mode. An unknown mode raises.
    """
    separatorprint()

    molsys = MolSystemData.load(WATER, VPT2_INPUTS)
    printtest(f'eigenvals: {molsys.eigenvals}')

    # ---- do analysis
    states, diagn = ws.intensities.anharmonic_treatment.anharm_analyzer_data(props=molsys.mol_props,
                                                                             nc_sqrt_eigval=molsys.eigenvals,
                                                                             regime='GVPT2',
                                                                             exclude_modes=[1])
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    states_1quantum_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states if len(list(s.harm_quanta_coeffs.keys())[0])==1}
    assert set(states_1quantum_corrected) == {0, 2}
    assert not [key for s in states for key in s.harm_quanta_coeffs if '1' in key]
    assert diagn['exclude_modes'] == (1,)

    for k,v in st_dict.items():
        print(k.ljust(10), v)
    print(diagn)

    print('\n1 quantum levels')
    for k,v in molsys.eigenvals.items():
        print(k, '--', v, '--', states_1quantum_corrected.get(k))

    with pytest.raises(ValueError, match=r'exclude_modes \[3\]'):
        ws.intensities.anharmonic_treatment.anharm_analyzer_data(props=molsys.mol_props,
                                                                 nc_sqrt_eigval=molsys.eigenvals,
                                                                 regime='GVPT2', exclude_modes=[3])

'''
def test_anharm_analyzer_vibana():
    """
    Anharmonic analyzer (using vpt2.py module) integration test 
    without wilson simulation

    OMG... it's very complicated. i give up now
    """
    separatorprint()
    import logging
    from ....wilson_utils.logger import setup_logger
    setup_logger("wilson.", level=logging.DEBUG)
    logging.getLogger('wilson.wilson.spectrum.vpt2').setLevel(logging.INFO)

    from CQCParse.logger import setup_logger as set_loggerCQCP
    set_loggerCQCP('CQCParse', level=logging.ERROR)

    from ...anharmonic_treatment.anharmonic_analyzer import anharm_analyzer_data
    from .... import wilson_main as ws_main

    mol_system = ws_main.abstractions.MolecularSystem(name='FORM', natoms=4)
    calc_setup = ws_main.abstractions.DataOriginInfo(source_type='gaussian', lvl_theory='B3LYP', basis_set='cc-pVQZ')

    vibana = ws_main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2',
                                              vibana_own_analysis='anharm', # should this vary? take minimal needed for regime unless specified? 
                                              )
    printtest(f'vibana.vibana_own_analysis: {vibana.vibana_own_analysis}')
    # FIXME: is it intentionally not possible for VibAnaSetup to get results from files?

    props, residual_vib_info, vibana.max_state_lvl = \
        ws_main.main_functions.find_props_and_max_state_lvl(terms, vibana)

    data_dict = {}
    ws_main.main_functions.request_props(props, data_dict=data_dict)
    ws_main.main_functions.request_residual_vib_info(residual_vib_info, data_dict)

    ws_main.main_functions.fill_props_results(props)
    ws_main.main_functions.fill_residual_vib_info_results(vibana)
    
    printtest(f'nc_sqrt_eigval: {vibana.nc_sqrt_eigval}') # vibana_own_analysis='all' -> nc_sqrt_eigval is None
    print(f'props: {props}') # vibana_own_analysis='all' -> nc_sqrt_eigval is None

    try:
        # FIXME: should be done internally with WilsonSimulation somehow? or when?
        res = ws_main.main_functions.do_anharmonic_analysis(vib_ana=vibana, 
                                                            props=props,
                                                            anharmonic_analyzer=anharm_analyzer_data)
    except Exception as e:
        assert False, f"Test failed due to an exception: {e}"
'''