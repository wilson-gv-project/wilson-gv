from unittest.mock import Mock, patch

import wilson_suite as ws
from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationWorkflow
import numpy as np


def test_workflow_run_with_keep_intermediates():
    """Test workflow.run(keep_intermediates) behavior"""
    
    # Create mock simulation
    mock_sim = Mock()
    mock_sim.terms = {}
    mock_sim.system = Mock()
    mock_sim.exp = Mock()
    mock_sim.vib_ana_setup = Mock()
    mock_sim.props = []
    mock_sim.spec_eval_setup = Mock()
    mock_sim.spec_eval_setup.ev_info = Mock()
    mock_sim.spec_eval_setup.ev_info.Gamma = 0.1
    mock_sim.spec_eval_setup.ev_info.spectral_window = Mock()
    mock_sim.spec_eval_setup.ev_info.grid_resolution = 100
    
    # Mock all workflow steps to return simple data
    with patch('wilson_suite.wilson_intensities.amplitudes.eval_wf.EvaluationWorkflow._prep_terms', return_value=['term1', 'term2']):
        with patch('wilson_suite.wilson_intensities.amplitudes.eval_wf.EvaluationWorkflow._prep_data', return_value=(['state1'], {}, {})):
            with patch('wilson_suite.wilson_intensities.amplitudes.eval_wf.EvaluationWorkflow._process_resonances', return_value=({}, {})):
                with patch('wilson_suite.wilson_intensities.amplitudes.eval_wf.EvaluationWorkflow._calc_coefficients', return_value={'c1': 1.0}):
                    with patch('wilson_suite.wilson_intensities.amplitudes.eval_wf.EvaluationWorkflow._extract_features', return_value=['feat1', 'feat2']):
                        with patch('wilson_suite.wilson_intensities.amplitudes.eval_wf.EvaluationWorkflow._place_in_specwindow', return_value=Mock()):
                            with patch('wilson_suite.wilson_intensities.amplitudes.eval_wf.EvaluationWorkflow._evaluate_spectrum', return_value='SPECTRUM'):
                                
                                workflow = EvaluationWorkflow(mock_sim)
                                
                                # Test WITHOUT intermediates
                                spectrum1, info1 = workflow.run(keep_intermediates=False)
                                assert spectrum1 == 'SPECTRUM'
                                assert 'timing' in info1
                                assert 'total_time' in info1
                                assert 'intermediates' not in info1  # Not kept
                                
                                # Test WITH intermediates
                                workflow2 = EvaluationWorkflow(mock_sim)
                                spectrum2, info2 = workflow2.run(keep_intermediates=True)
                                assert spectrum2 == 'SPECTRUM'
                                assert 'timing' in info2
                                assert 'intermediates' in info2  # Now included!
                                assert info2['intermediates']['prep_terms'] == ['term1', 'term2']
                                assert info2['intermediates']['all_features'] == ['feat1', 'feat2']
                                assert len(info2['intermediates']) == 7  # All steps


def test_workflow_run_with_keep_intermediates_real():
    """Test workflow.run(keep_intermediates) behavior"""
    from wilson_suite.wilson_main.workflow_abstractions import WilsonSimulation
    from wilson_suite.wilson_derive.main import get_fully_enhanced_terms
    from ....fixtures import evv_experiment
    from CQCParse.utils import PKG_ROOT as CQCPARSE_ROOT
    
    terms = get_fully_enhanced_terms(experiment=evv_experiment())
    axes_choice = evv_experiment().valid_axis_combs[((-1,), (2,))][3] # {'A': [(2,)], 'B': [(-1,), (2,)]}
    evv_terms =  ws.derive.term_var_translate.translate_terms_to_axis_variables(terms, axes_choice)

    mol_system = ws.main.abstractions.MolecularSystem(name='FORM', natoms=4)
    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian', 
                                                     lvl_theory='B3LYP', 
                                                     basis_set='cc-pVQZ', 
                                                     base_file_loc=CQCPARSE_ROOT+'/CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out')
    bounds_dict = {'A': (0., 5000.), 'B': (0., 5000.)}
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 4.7, 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 10, 'B': 10}})
    mock_sim = WilsonSimulation()
    mock_sim.terms = evv_terms
    mock_sim.system = mol_system
    mock_sim.exp = evv_experiment()
    mock_sim.vib_ana_setup = vib_ana
    
    mock_sim.addPropEvalSetup(eval_uniform=calc_setup)
    mock_sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    mock_sim.dressPropsWithSetup()

    mock_sim.spec_eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    mock_sim.getResults(obtainer=wilson_data_obtainer)


    workflow = EvaluationWorkflow(mock_sim)
    
    # Test WITHOUT intermediates
    spectrum1, info1 = workflow.run(keep_intermediates=False)

    np.set_printoptions(linewidth=280, precision=3)

    assert 'timing' in info1
    assert 'total_time' in info1
    assert 'intermediates' not in info1  # Not kept
    
    # Test WITH intermediates
    workflow2 = EvaluationWorkflow(mock_sim)
    spectrum2, info2 = workflow2.run(keep_intermediates=True)

    assert 'timing' in info2
    assert 'intermediates' in info2  # Now included!
    assert len(info2['intermediates']['prep_terms']) == 14
    assert len(info2['intermediates']['all_features']) == 60
    assert len(info2['intermediates']) == 7  # All steps
    
    np.set_printoptions(linewidth=280, precision=3)

    print(info2.keys())


def test_workflow_run_with_keep_intermediates_real_wfsim():
    """Test workflow.run(keep_intermediates) behavior"""
    from wilson_suite.wilson_main.wf import WilsonSimulation
    # from wilson_suite.wilson_main.workflow_abstractions import WilsonSimulation

    from wilson_suite.wilson_derive.main import get_fully_enhanced_terms
    from ....fixtures import evv_experiment
    from CQCParse.utils import PKG_ROOT as CQCPARSE_ROOT
    
    terms = get_fully_enhanced_terms(experiment=evv_experiment())
    axes_choice = evv_experiment().valid_axis_combs[((-1,), (2,))][3] # {'A': [(2,)], 'B': [(-1,), (2,)]}
    evv_terms =  ws.derive.term_var_translate.translate_terms_to_axis_variables(terms, axes_choice)

    mol_system = ws.main.abstractions.MolecularSystem(name='FORM', natoms=4)
    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian', 
                                                     lvl_theory='B3LYP', 
                                                     basis_set='cc-pVQZ', 
                                                     base_file_loc=CQCPARSE_ROOT+'/CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out')
    bounds_dict = {'A': (0., 5000.), 'B': (0., 5000.)}
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 4.7, 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 10, 'B': 10}})
    mock_sim = WilsonSimulation()
    mock_sim.terms = evv_terms
    mock_sim.system = mol_system
    mock_sim.exp = evv_experiment()
    mock_sim.vib_ana_setup = vib_ana
    
    mock_sim.addPropEvalSetup(eval_uniform=calc_setup)
    mock_sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    mock_sim.dressPropsWithSetup()

    mock_sim.spec_eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    mock_sim.getResults(obtainer=wilson_data_obtainer)

    print(mock_sim.is_ready)
    print(mock_sim.is_configured)
    
    mock_sim.evaluate(keep_intermediates=True)

    assert 'timing' in mock_sim.diagn
    assert 'intermediates' in mock_sim.diagn  # Now included!
    assert len(mock_sim.diagn['intermediates']['prep_terms']) == 14
    assert len(mock_sim.diagn['intermediates']['all_features']) == 60
    assert len(mock_sim.diagn['intermediates']) == 11  # All steps
    
    np.set_printoptions(linewidth=280, precision=3)

    print(mock_sim.diagn.keys())