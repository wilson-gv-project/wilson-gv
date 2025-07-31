import wilson_main.abstractions as wm_abst
from wilson_utils.logger import setup_logger

from wilson_suite import fixtures as fixt
from wilson_suite import molprops as propsfixt
from wilson_suite import calcsetups as calcsetupfixt
from wilson_suite import vibana as vibanafixt
from wilson_suite import parsing as parse_fixt

import pytest
from wilson_utils.printing import printtest

import logging
setup_logger("wilson", level=logging.DEBUG)


def test_CalculationBatch_getResults():
    # getting results from source
    # fill in props input and vibanasetup
    # could be a function, needs some id-ing of results
    """
    def getResults(self, props_to_fill: list[MolecularProperty],
                vib_ana_setup_to_fill: VibAnaSetup=None, source_type: str='',
                source_types: list[str]=[], source_loc: Any=None, datavault: Any = None):
        
    is suposed to fill in :
    vib_ana_setup_to_fill and props_to_fill -> adds values there, modifies incoming objects
            
    minimum setup to test functionality:
        None
    but needs to be initialized with system and calc_setup (both not optional)
    """
    dummyBatch = wm_abst.CalculationBatch(system=fixt.mol_system, calc_setup=calcsetupfixt.calc_setup)
    
    prop_vals_check = [i.vals for i in propsfixt.props_evv_anharm_wcalc_novals]
    assert all(elem is None for elem in prop_vals_check), 'Not a valid test for getting properties values'

    try:
        dummyBatch.getResults(props_to_fill=propsfixt.props_evv_anharm_wcalc_novals, 
                            vib_ana_setup_to_fill=vibanafixt.vibanasetup_anharm)
        pytest.fail('Should have raised an error without source_type specified')
    except NotImplementedError as e:
        assert "Data retrieval for this source_type [] is not implemented" in str(e), \
            f"Unexpected error message: {e}"


def test_CalculationBatch_getResults_vault():
    """
    is suposed to fill in :
    vib_ana_setup_to_fill and props_to_fill -> adds values there, modifies incoming objects
    """
    dummyBatch = wm_abst.CalculationBatch(system=fixt.mol_system, calc_setup=calcsetupfixt.calc_setup)
    
    prop_vals_check = [i.vals for i in propsfixt.props_evv_anharm_wcalc_novals]
    assert all(elem is None for elem in prop_vals_check), 'Not a valid test for getting properties values'

    dummyBatch.getResults(props_to_fill=propsfixt.props_evv_anharm_wcalc_novals, 
                          vib_ana_setup_to_fill=vibanafixt.vibanasetup_anharm,
                          source_type='vault', datavault=parse_fixt.vault)

    prop_vals_check = [i.vals for i in propsfixt.props_evv_anharm_wcalc_novals]

    assert all(elem is not None for elem in prop_vals_check), 'Some props did not get values?'

    assert vibanafixt.vibanasetup_anharm.vibana_prop_need == 'anharm'
    assert vibanafixt.vibanasetup_anharm.regime == 'GVPT2'
    # should not be None?
    assert vibanafixt.vibanasetup_anharm.states is None


def test_CalculationBatch_getResultsFromOutputs():
    pass
