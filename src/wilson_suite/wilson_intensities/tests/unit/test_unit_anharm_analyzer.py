"""
anharm_analyzer/anharm_analyzer_data are pure functions
Need to test:
    -[] Expected output
    -[] Edge cases
    -[] Properties or invariants
    -[] Reference comparison

"""
from ....wilson_utils.printing import printtest, separatorprint
from ....wilson_main import abstractions as wm_abst


def test_anharm_analyzer_vibana():
    """
    Trying to isolate anharm_analyzer_data function
    and test it

    TODO: not passing now because unfinished
    """
    separatorprint()
    import logging
    from ....wilson_utils.logger import setup_logger
    setup_logger("wilson", level=logging.DEBUG)
    logging.getLogger('wilson.wilson.spectrum.vpt2').setLevel(logging.INFO)

    from CQCParse.logger import setup_logger as set_loggerCQCP
    set_loggerCQCP('CQCParse', level=logging.ERROR)

    from ...anharmonic_treatment.anharmonic_analyzer import anharm_analyzer_data

    context = {'system': wm_abst.MolecularSystem(name='FORM', natoms=4, geo=None, geo_extra=None, linear=False), 
               'props': [wm_abst.MolecularProperty(prop_spec={'ops': ('g', 'g', 'g'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='cff'), 
                         wm_abst.MolecularProperty(prop_spec={'ops': ('g', 'g', 'g', 'g'), 'freq': (0.0, 0.0, 0.0, 0.0)}, trivial_name='qff'), 
                         wm_abst.MolecularProperty(prop_spec={'ops': ('r',), 'freq': 0.0}, trivial_name='B'), 
                         wm_abst.MolecularProperty(prop_spec={'ops': ('g', 'g', 'r'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='coriolis')],
                         'regime': 'GVPT2', 'regime_subinfo': None,
                         'nc_sqrt_eigval': {0: 2878.687, 1: 1820.416, 2: 1534.549, 3: 1203.179, 4: 2933.526, 5: 1268.91}, 
                         'exclude_modes': []}
    
    # how it's used in VibAnaSetup().doAnharmonicAnalysis
    anharm_analyzer_data(**context)
