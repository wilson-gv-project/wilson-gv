from wilson.utils.debug import printtest, separator_print

def test_anharm_analyzer_vibana():
    """
    Trying to isolate anharm_analyzer_data function
    and test it
    """
    separator_print()
    import logging
    from wilson_utils.logger import setup_logger
    setup_logger("wilson", level=logging.DEBUG)
    logging.getLogger('wilson.wilson.spectrum.vpt2').setLevel(logging.INFO)

    from CQCParse.logger import setup_logger as set_loggerCQCP
    set_loggerCQCP('CQCParse', level=logging.ERROR)

    from wilson.spectrum.anharmonic_analyzer import anharm_analyzer_data
    # context = {'system': None, 'props': None, 'nc_sqrt_eigval': None,
    #            'regime': None, 'regime_subinfo': None, 'exclude_modes': None}
    context = {'system': None, 'props': None, 'nc_sqrt_eigval': None,
            'regime': None, 'regime_subinfo': None}
    
    # how it's used in VibAnaSetup().doAnharmonicAnalysis
    anharm_analyzer_data(**context)


    # from wilson_utils.paths import SUITE_ROOT
    # database_csv = SUITE_ROOT+'/wilson_intensities/tests/test_database/mini_files_database.csv'

    # from wilson.spectrum.anharmonic_analyzer import anharm_analyzer_data
    # import wilson_main as ws_main

    # mol_system = ws_main.abstractions.MolecularSystem(name='FORM', natoms=4)
    # calc_setup = ws_main.abstractions.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVQZ')
    # vibana = ws_main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2',
    #                                           vibana_prop_need='anharm', # should this vary? take minimal needed for regime unless specified? 
    #                                           allow_skip_eigvec=True, external_fill_from=calc_setup)
    # printtest(f'vibana.vibana_prop_need: {vibana.vibana_prop_need}')
    # props = vibana.tellNeededProps()

    # # FIXME? WilsonSimulation: def dressPropsWithSetup(self) - can't dressProps without WilsonSimulation - now done with:
    
    # ws_main.abstractions.dressPropsWithSetup(props, eval_uniform=calc_setup, eval_by_prop_name=None)
    
    # calc_batch = ws_main.abstractions.CalculationBatch(system=mol_system, calc_setup=calc_setup, properties=props)    
    # # needs dressed props with calc setup
    # calc_batch.getResultsFromVault(props_to_fill=props, vib_ana_setup_to_fill=vibana,
    #                                source_loc=database_csv)
    
    # printtest(f'nc_sqrt_eigval: {vibana.nc_sqrt_eigval}') # vibana_prop_need='all' -> nc_sqrt_eigval is None

    # try:
    #     # FIXME: should be done internally with WilsonSimulation somehow? or when?
    #     vibana.doAnharmonicAnalysis(props, anharmonic_analyzer=anharm_analyzer_data)
    # except Exception as e:
    #     assert False, f"Test failed due to an exception: {e}"