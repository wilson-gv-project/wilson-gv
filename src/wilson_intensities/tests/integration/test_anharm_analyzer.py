from wilson_utils.printing import printtest, separatorprint

def evv_experiment():
    import wilson_experiment as ws_experiment

    pulse_ir_1 = ws_experiment.abstractions.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2 = ws_experiment.abstractions.EmPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1 = ws_experiment.abstractions.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)

    pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]

    field_a = ws_experiment.abstractions.ElectricField(pulses)
    order = len(pulses)

    field_a.findEpochs()

    detector_a = ws_experiment.abstractions.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                                        detection_polarization=[0.0, 0.0, 1.0],
                                                        detection_range=[0.003 + 0.0001*i for i in range(101)],
                                                        wv_filter=[{1: [-1], 2: [1], 3: [1]}]) #, {1: [-1], 2: [1], 3: [1]}

    # Push one carrier freq
    scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a = [0.0001*i for i in range(101)]
    scan_a = ws_experiment.abstractions.SpecScan(scan_obj_a, scan_range_a)
    experiment_a = ws_experiment.abstractions.VibExperiment(order, field_a, detector_a, [scan_a], magn_conditions=[[-1, 2]])
    return experiment_a

def test_anharm_analyzer():
    """
    Anharmonic analyzer (using vpt2.py module) integration test
    """
    separatorprint()
    import logging
    from wilson_utils.logger import setup_logger
    setup_logger("wilson.", level=logging.DEBUG)
    logging.getLogger('wilson.wilson.spectrum.vpt2').setLevel(logging.INFO)

    from CQCParse.logger import setup_logger as set_loggerCQCP
    set_loggerCQCP('CQCParse', level=logging.ERROR)

    from wilson_utils.paths import SUITE_ROOT
    from wilson.spectrum.anharmonic_analyzer import anharm_analyzer_data
    import wilson_main as ws_main
    import wilson_derive as ws_derive

    mol_system = ws_main.abstractions.MolecularSystem(name='FORM', natoms=4)
    calc_setup = ws_main.abstractions.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVQZ')
    vibana = ws_main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2',
                                              vibana_prop_need='anharm', # should this vary? take minimal needed for regime unless specified? 
                                              allow_skip_eigvec=True, 
                                              external_fill_from=calc_setup)
    printtest(f'vibana.vibana_prop_need: {vibana.vibana_prop_need}')
    experiment_a = evv_experiment()

    sim = ws_main.abstractions.WilsonSimulation()
    sim.addExperiment(experiment_a)
    sim.getTerms(ws_derive.main.get_fully_enhanced_terms)
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana_setup=vibana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)

    # should be careful with props, because props are needed for vib analyzer
    sim.findPropsAndMaxStateLvl() # setting up self.props/sim.props
    printtest(f'[i.triv_name for i in sim.props] {[i.trivial_name for i in sim.props]}')
    
    sim.dressPropsWithSetup()
    sim.makeCalculationBatches()
    
    database_csv = SUITE_ROOT+'/wilson_intensities/tests/test_database/mini_files_database.csv'
    from CQCParse.relay import DataVault
    vault = DataVault(database_csv)

    # looks like VibAnaSetup can't get data without WilsonSimulation? CalculationBatches?
    sim.getResultsFromCalculationBatches(source_type='vault',
                                        datavault=vault, source_loc=SUITE_ROOT+'/wilson_intensities/tests')
    
    printtest(f'nc_sqrt_eigval: {sim.vib_ana_setup.nc_sqrt_eigval}') # vibana_prop_need='all' -> nc_sqrt_eigval is None
    printtest(sim.props)

    try:
        # FIXME: should be done internally with WilsonSimulation somehow? or when?
        sim.vib_ana_setup.doAnharmonicAnalysis(sim.props, anharmonic_analyzer=anharm_analyzer_data)
    except Exception as e:
        assert False, f"Test failed due to an exception: {e}"


def test_anharm_analyzer_vibana():
    """
    Anharmonic analyzer (using vpt2.py module) integration test 
    without wilson simulation
    """
    separatorprint()
    import logging
    from wilson_utils.logger import setup_logger
    setup_logger("wilson.", level=logging.DEBUG)
    logging.getLogger('wilson.wilson.spectrum.vpt2').setLevel(logging.INFO)

    from CQCParse.logger import setup_logger as set_loggerCQCP
    set_loggerCQCP('CQCParse', level=logging.ERROR)

    from wilson_utils.paths import SUITE_ROOT
    database_csv = SUITE_ROOT+'/wilson_intensities/tests/test_database/mini_files_database.csv'

    from wilson.spectrum.anharmonic_analyzer import anharm_analyzer_data
    import wilson_main as ws_main

    mol_system = ws_main.abstractions.MolecularSystem(name='FORM', natoms=4)
    calc_setup = ws_main.abstractions.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVQZ')
    vibana = ws_main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2',
                                              vibana_prop_need='anharm', # should this vary? take minimal needed for regime unless specified? 
                                              allow_skip_eigvec=True, external_fill_from=calc_setup)
    printtest(f'vibana.vibana_prop_need: {vibana.vibana_prop_need}')
    props = vibana.tellNeededProps()

    for i in props:
        i.addCalcSetup(calc_setup)

    calc_batch = ws_main.abstractions.CalculationBatch(system=mol_system, calc_setup=calc_setup, properties=props)    
    
    database_csv = SUITE_ROOT+'/wilson_intensities/tests/test_database/mini_files_database.csv'
    from CQCParse.relay import DataVault
    vault = DataVault(database_csv)

    # needs dressed props with calc setup
    calc_batch.getResults(props_to_fill=props, vib_ana_setup_to_fill=vibana,
                          source_type='vault',
                          datavault=vault, source_loc=SUITE_ROOT+'/wilson_intensities/tests')

    
    printtest(f'nc_sqrt_eigval: {vibana.nc_sqrt_eigval}') # vibana_prop_need='all' -> nc_sqrt_eigval is None

    try:
        # FIXME: should be done internally with WilsonSimulation somehow? or when?
        vibana.doAnharmonicAnalysis(props, anharmonic_analyzer=anharm_analyzer_data)
    except Exception as e:
        assert False, f"Test failed due to an exception: {e}"
