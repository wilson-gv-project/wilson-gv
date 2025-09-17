"""
Turning scripts into pytests
"""
from wilson_suite.wilson_utils.paths import SUITE_ROOT
from wilson_suite.wilson_utils.serialization import unpickle_smth_from
import numpy as np
from wilson_suite.wilson_utils.printing import separatorprint

def test_evv_tester_dataclasses():
    """
    simple run with no pickling
    """
    import evv_tester as evv_tester
    evv_tester.TO_PICKLES = []
    evv_tester.PREP_ONLY = False
    wilsim = evv_tester.run()

    assert hasattr(wilsim, 'spec')

def test_evv_terms_tester():
    import evv_terms_tester

    assert True


def test_evv_tester_dataclasses_vibexp():

    """
    pickle and unpickle VibExperiment after initialization in the integration script.
    pickled VibExperiment is then unpickled in the script and calculation continues.
    Now comparing loaded from file VibExperiment to the one saved in WilsonSimulation instance (after all calculation is done).
    """
    import evv_tester as evv_tester
    
    topickles_1 = ['VibExperiment']
    evv_tester.TO_PICKLES = topickles_1
    
    # WilsonSimulation object after spectrum was calculated and rendered
    wilsim = evv_tester.run()
    # loading saved VibExperiment in that calculation from the file (file name was saved in a dict)
    load_vibexp = unpickle_smth_from(filenamepkl=evv_tester.PKL_FILES['VibExperiment'], load_from=SUITE_ROOT+'/tests/')
    
    # now this is not working: evv_tester_dataclasses.PKL_FILES['VibExperiment'] == wilsim.exp; comparison not implemented but it could be # TODO?
    
    assert wilsim.exp.order == load_vibexp.order
    assert load_vibexp.detector.detection_range == wilsim.exp.detector.detection_range
    assert load_vibexp.epochs == wilsim.exp.epochs

    from wilson_suite.wilson_experiment.abstractions import EmPulse, ElectricField
    # field_ref comes from print: print(wilsim.exp.field)
    field_ref = ElectricField(pulses=[EmPulse(env='ideal', maxstr=1e-05, tc=50.0, cf=0.0, cf_uv=0.0, dev=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1), 
                                      EmPulse(env='impulsive', maxstr=1e-05, tc=100.0, cf=None, cf_uv=0.0, dev=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2), 
                                      EmPulse(env='ideal', maxstr=1e-05, tc=120.0, cf=0.0, cf_uv=0.072, dev=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)])
    assert load_vibexp.field == field_ref
    assert wilsim.exp.field == field_ref


def test_evv_tester_dataclasses_calcsetup():
    """
    pickle and unpickle ExternalCalcSetup after initialization in the integration script.
    pickled ExternalCalcSetup is then unpickled in the script and calculation continues.
    Now comparing loaded from file ExternalCalcSetup to the one saved in WilsonSimulation instance (after all calculation is done).
    """
    import evv_tester as evv_tester

    topickles_2 = ['ExternalCalcSetup']
    evv_tester.TO_PICKLES = topickles_2
    
    # WilsonSimulation object after spectrum was calculated and rendered
    wilsim2 = evv_tester.run()

    # loading saved ExternalCalcSetup in that calculation from the file (file name was saved in a dict)
    load_calcsetup = unpickle_smth_from(filenamepkl=evv_tester.PKL_FILES['ExternalCalcSetup'], load_from=SUITE_ROOT+'/tests/')
    assert load_calcsetup == wilsim2.eval_uniform

    from wilson_suite.wilson_main.abstractions import ExternalCalcSetup
    calcsetup_ref = ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVQZ', other_setup={}, other_setup_identifier={})

    assert load_calcsetup == calcsetup_ref
    assert wilsim2.eval_uniform == calcsetup_ref


def test_evv_tester_dataclasses_wilsonsim():
    """
    In the script can save WilsonSim instance three times: 
        after initialization, just before the evaluation and after evaluation.
    Loading them:
        load_wilsonsim_init - from pickle after initialization
        load_wilsonsim_mid - from pickle just before evaluation
        load_wilsonsim_final - from pickle after evaluation
    
    load_wilsonsim_init would not have some atributes which were set later in script

    """
    import evv_tester as evv_tester

    topickles_2 = ['WilsonSimulation_init', 'WilsonSimulation_final']
    evv_tester.TO_PICKLES = topickles_2
    evv_tester.PREP_ONLY = False
    
    # WilsonSimulation object after spectrum was calculated and rendered
    wilsim3 = evv_tester.run()
    print(evv_tester.PKL_FILES)
    # loading saved WilsonSimulation in that calculation from the file (file name was saved in a dict)
    load_wilsonsim_init = unpickle_smth_from(filenamepkl=evv_tester.PKL_FILES['WilsonSimulation_init'], load_from=SUITE_ROOT+'/tests/')
    load_wilsonsim_final = unpickle_smth_from(filenamepkl=evv_tester.PKL_FILES['WilsonSimulation_final'], load_from=SUITE_ROOT+'/tests/')

    assert load_wilsonsim_init.spec is None
    assert isinstance(load_wilsonsim_final.spec, np.ndarray)
    assert isinstance(wilsim3.spec, np.ndarray)
    assert np.allclose(np.abs(load_wilsonsim_final.spec), np.abs(wilsim3.spec))

    # optional pickling at init and before evaluation
    topickles_2 = ['WilsonSimulation_init', 'WilsonSimulation_mid', 'WilsonSimulation_final']
    evv_tester.TO_PICKLES = topickles_2
    
    # WilsonSimulation object after spectrum was calculated and rendered
    wilsim3 = evv_tester.run()

    # loading saved WilsonSimulation in that calculation from the file (file name was saved in a dict)
    load_wilsonsim_init = unpickle_smth_from(filenamepkl=evv_tester.PKL_FILES['WilsonSimulation_init'], load_from=SUITE_ROOT+'/tests/')
    load_wilsonsim_mid = unpickle_smth_from(filenamepkl=evv_tester.PKL_FILES['WilsonSimulation_mid'], load_from=SUITE_ROOT+'/tests/')
    load_wilsonsim_final = unpickle_smth_from(filenamepkl=evv_tester.PKL_FILES['WilsonSimulation_final'], load_from=SUITE_ROOT+'/tests/')

    assert load_wilsonsim_init.spec is None
    assert load_wilsonsim_mid.spec is None

    assert isinstance(load_wilsonsim_final.spec, np.ndarray)
    assert isinstance(wilsim3.spec, np.ndarray)

    assert np.allclose(np.abs(load_wilsonsim_final.spec), np.abs(wilsim3.spec))

    from wilson_suite.wilson_main.abstractions import VibAnaSetup, MolecularSystem, ExternalCalcSetup
    vibanasetup_ref = VibAnaSetup(regime='GVPT2', system=MolecularSystem(name='FORM', natoms=4, geo=None, geo_extra=None, linear=False), 
                                  regime_subinfo=None, max_state_lvl=3, nc_sqrt_eigval={0: 2878.687, 1: 1820.416, 2: 1534.549, 3: 1203.179, 4: 2933.526, 5: 1268.91}, 
                                  nc_eigvec=None, allow_skip_eigvec=True, vibana_prop_need='none', 
                                  external_fill_from=ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVQZ', other_setup={}, other_setup_identifier={}), 
                                  exclude_modes=[])
    assert vibanasetup_ref.system == wilsim3.vib_ana_setup.system
    assert vibanasetup_ref.external_fill_from == wilsim3.vib_ana_setup.external_fill_from

    assert vibanasetup_ref.system == load_wilsonsim_mid.vib_ana_setup.system
    assert vibanasetup_ref.external_fill_from == load_wilsonsim_mid.vib_ana_setup.external_fill_from

    assert vibanasetup_ref.nc_sqrt_eigval == wilsim3.vib_ana_setup.nc_sqrt_eigval
    assert vibanasetup_ref.nc_sqrt_eigval == load_wilsonsim_final.vib_ana_setup.nc_sqrt_eigval

def test_logger_evv_tester_file():
    import logging
    from wilson_suite.wilson_utils.logger import setup_logger
    setup_logger("wilson", level=logging.DEBUG, log_to_file=SUITE_ROOT+'/tests/out.log')
    
    import evv_tester
    evv_tester.run()

    with open(SUITE_ROOT+"/tests/out.log", "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [i for i in lines if i!='']
    
    info_line = [line for line in lines if "INFO" in line and "evv_tester_dataclasses.py" in line]
    assert info_line, "No INFO log line with evv_tester_dataclasses.py found"

    debug_line = [line for line in lines if "DEBUG" in line and "np.max(intensities): 4.2029e+11" in line]
    assert debug_line is not None, "DEBUG log with np.max(intensities) line not found"

def test_logger_evv_tester_terminal():
    separatorprint()
    import logging
    from wilson_suite.wilson_utils.logger import setup_logger
    setup_logger("wilson", level=logging.DEBUG)
    
    import evv_tester
    evv_tester.PREP_ONLY = False
    evv_tester.run()
