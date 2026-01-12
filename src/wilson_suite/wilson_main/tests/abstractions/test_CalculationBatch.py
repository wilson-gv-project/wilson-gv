# from ... import abstractions as wm_abst
# from ....wilson_utils.logger import setup_logger

# from wilson_suite import fixtures as fixt
# from wilson_suite import molprops as propsfixt
# from wilson_suite import calcsetups as calcsetupfixt
# from wilson_suite import vibana as vibanafixt
# from wilson_suite import parsing as parse_fixt

# from ....wilson_utils.paths import SUITE_ROOT
# from CQCParse.utils import PKG_ROOT as CQCPARSE_ROOT
# import pytest

# import logging
# setup_logger("wilson", level=logging.DEBUG)


# def test_CalculationBatch_getResults():
#     # getting results from source
#     # fill in props input and vibanasetup
#     # could be a function, needs some id-ing of results
#     """
#     def getResults(self, props_to_fill: list[MolecularProperty],
#                 vib_ana_setup_to_fill: VibAnaSetup=None, source_type: str='',
#                 source_types: list[str]=[], source_loc: Any=None, datavault: Any = None):
        
#     is suposed to fill in :
#     vib_ana_setup_to_fill and props_to_fill -> adds values there, modifies incoming objects
            
#     minimum setup to test functionality:
#         None
#     but needs to be initialized with system and calc_setup (both are not optional)
#     """
#     # importing fresh props

#     dummyBatch = wm_abst.CalculationBatch(system=fixt.mol_system, calc_setup=calcsetupfixt.calc_setup)
    
#     prop_vals_check = [i.vals for i in propsfixt.props_evv_anharm_wcalc_novals]
#     assert all(elem is None for elem in prop_vals_check), 'Not a valid test for getting properties values'

#     try:
#         dummyBatch.getResults(props_to_fill=propsfixt.props_evv_anharm_wcalc_novals, 
#                             vib_ana_setup_to_fill=vibanafixt.vibanasetup_anharm)
#         pytest.fail('Should have raised an error without source_type specified')
#     except NotImplementedError as e:
#         assert "Data retrieval for this source_type [] is not implemented" in str(e), \
#             f"Unexpected error message: {e}"


# def test_CalculationBatch_getResults_vault():
#     """
#     is suposed to fill in :
#     vib_ana_setup_to_fill and props_to_fill -> adds values there, modifies incoming objects
#     """
#     # importing fresh props

#     dummyBatch = wm_abst.CalculationBatch(system=fixt.mol_system, calc_setup=calcsetupfixt.calc_setup)

#     props_mock = propsfixt.makeMockProps('evv_anharm_wcalc_uni')
#     prop_vals_notNone = [i.trivial_name for i in props_mock if i.vals is not None]
#     assert not prop_vals_notNone, 'Not a valid test for getting properties values. Values are already present'

#     assert hasattr(dummyBatch, 'parser_obj'), 'No self.parser_obj in this CalculationBatch instance'

#     dummyBatch.getResults(props_to_fill=props_mock, 
#                           vib_ana_setup_to_fill=vibanafixt.vibanasetup_anharm,
#                           source_type='vault', datavault=parse_fixt.vault, 
#                           source_loc=SUITE_ROOT+'/wilson_intensities/tests')

#     prop_vals_notNone = [i.trivial_name for i in props_mock if i.vals is not None]
#     assert prop_vals_notNone, 'Some props did not get values?'

#     assert vibanafixt.vibanasetup_anharm.vibana_own_analysis == 'anharm'
#     assert vibanafixt.vibanasetup_anharm.regime == 'GVPT2'
    
#     # TODO in WilsonSim? should not be None? - if OK then need to get states later; 
#     # should have a check in WilsonSim?
#     assert vibanafixt.vibanasetup_anharm.states is None


# def test_CalculationBatch_getResultsFromOutputs():
#     """
#     is suposed to fill in :
#     vib_ana_setup_to_fill and props_to_fill -> adds values there, modifies incoming objects
#     """
#     # importing fresh props

#     dummyBatch = wm_abst.CalculationBatch(system=fixt.mol_system, calc_setup=calcsetupfixt.calc_setup)
#     props_mock = propsfixt.makeMockProps('evv_anharm_wcalc_uni')
#     prop_vals_notNone = [i.trivial_name for i in props_mock if i.vals is not None]
#     assert not prop_vals_notNone, 'Not a valid test for getting properties values. Values are already present'

#     assert hasattr(dummyBatch, 'parser_obj'), 'No self.parser_obj in this CalculationBatch instance'

#     datadict = {'source': 'gaussian', 'type': 'log', 
#                 'files': {'mol_code': 'FORM', 'method': 'B3LYP', 'basis': 'cc_pVQZ', 
#                           'log': SUITE_ROOT+'/wilson_intensities/tests/test_database/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out'}}
    
#     datadict = {'source': 'gaussian', 'type': 'log', 
#             'files': {'mol_name': 'FORM', 'method': 'B3LYP', 'basis': 'cc-pVQZ', 
#             'log': CQCPARSE_ROOT + '/CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out'}}
    
#     from CQCParse.parsing import GaussianOutput as OutFiles
#     out = OutFiles(molecule=dummyBatch.system.name, 
#                     conformer=dummyBatch.system.conformer,
#                     method=dummyBatch.calc_setup.lvl_theory, 
#                     basis=dummyBatch.calc_setup.basis, program=dummyBatch.calc_setup.program)
#     out.log_file = datadict['files']['log']
    
#     dummyBatch.parser_obj.relevant_files = out # shouldn't need this or datadict

#     dummyBatch.getResultsFromOutputs(props_to_fill=props_mock, 
#                                      vib_ana_setup_to_fill=vibanafixt.vibanasetup_anharm,
#                                      datafilesdict=datadict)

#     prop_vals_notNone = [i.trivial_name for i in props_mock if i.vals is not None]
#     assert prop_vals_notNone, 'Some props did not get values?'

#     assert vibanafixt.vibanasetup_anharm.vibana_own_analysis == 'anharm'
#     assert vibanafixt.vibanasetup_anharm.regime == 'GVPT2'
    
#     # TODO in WilsonSim? should not be None? - if OK then need to get states later; 
#     # should have a check in WilsonSim?
#     assert vibanafixt.vibanasetup_anharm.states is None