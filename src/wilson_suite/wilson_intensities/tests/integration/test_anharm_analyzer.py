from ....wilson_utils.printing import printtest, separatorprint
from ....fixtures import evv_experiment
import wilson_suite as ws

def test_anharm_analyzer():
    """
    Anharmonic analyzer (using vpt2.py module) integration test
    """
    separatorprint()
    import logging
    from ....wilson_utils.logger import setup_logger
    setup_logger("wilson_suite.", level=logging.DEBUG)
    logging.getLogger('wilson_suite.').setLevel(logging.DEBUG)

    from ....wilson_utils.paths import SUITE_ROOT
    from .... import wilson_main as ws_main

    mol_system = ws.main.abstractions.MolecularSystem(name='h2o', natoms=3)
    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='anharm')

    printtest(f'vibana.vibana_own_analysis: {vib_ana.vibana_own_analysis}')
    experiment_a = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=experiment_a)
    axes_choice = experiment_a.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}
    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                     lvl_theory='HF', 
                                                     basis_set='STO-3G', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_h2o_HF_STO3G.out')

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
    
    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)
    
    print('\n')
    printtest(f'nc_sqrt_eigval: {sim.vib_ana_setup.nc_sqrt_eigval}') # vibana_own_analysis='all' -> nc_sqrt_eigval is None
    print('\n')
    
    # for p in sim.props:
    #     if p.trivial_name == 'cff':
    #         printtest(p.vals)

    states, diagn = ws.intensities.anharmonic_treatment.anharm_analyzer_data(system=sim.system,
                                                                             props=sim.props,
                                                                             nc_sqrt_eigval=sim.vib_ana_setup.nc_sqrt_eigval,
                                                                             regime=sim.vib_ana_setup.regime,
                                                                             regime_subinfo=sim.vib_ana_setup.regime_subinfo,
                                                                             exclude_modes=None)
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    for k,v in st_dict.items():
        print(k.ljust(10), v)
    print(diagn)
    

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