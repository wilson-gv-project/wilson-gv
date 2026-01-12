import wilson_suite as ws
from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationWorkflow, make_evaluation_inputs
import numpy as np

def test_workflow_run_with_keep_intermediates_real():
    """Test workflow.run(keep_intermediates) behavior"""
    from wilson_suite.wilson_main.workflow_abstractions import WilsonSimulation
    from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
    from ....fixtures import evv_experiment
    from CQCParse.utils import PKG_ROOT as CQCPARSE_ROOT
    
    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}

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
    mock_sim.terms = terms
    mock_sim.system = mol_system
    mock_sim.exp = evv_experiment()
    mock_sim.vib_ana_setup = vib_ana
    
    mock_sim.addPropEvalSetup(eval_uniform=calc_setup)
    mock_sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    mock_sim.dressPropsWithSetup()
    mock_sim.setAxisChoiceAndTranslateTerms(axes_choice)


    mock_sim.spec_eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    mock_sim.getResults(obtainer=wilson_data_obtainer)

    eval_inputs = make_evaluation_inputs(simulation=mock_sim)
    workflow = EvaluationWorkflow(inputs=eval_inputs)
    
    # Test WITHOUT intermediates
    spectrum1 = workflow.run()
    info1 = workflow.artifacts

    np.set_printoptions(linewidth=280, precision=3)

    # assert 'timing' in info1
    # assert 'total_time' in info1
    # assert 'intermediates' not in info1  # Not kept
    
    # Test WITH intermediates
    workflow2 = EvaluationWorkflow(inputs=eval_inputs)
    spectrum2 = workflow2.run()
    info2 = workflow2.artifacts

    # assert 'timing' in info2
    # assert 'intermediates' in info2  # Now included!
    # assert len(info2['intermediates']['prep_terms']) == 14
    # assert len(info2['intermediates']['all_features']) == 60
    # assert len(info2['intermediates']) == 11  # All steps
    
    np.set_printoptions(linewidth=280, precision=3)


def test_workflow_run_with_keep_intermediates_real_wfsim():
    """Test workflow.run(keep_intermediates) behavior"""
    from wilson_suite.wilson_main.wf import WilsonSimulation
    # from wilson_suite.wilson_main.workflow_abstractions import WilsonSimulation

    from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
    from ....fixtures import evv_experiment
    from CQCParse.utils import PKG_ROOT as CQCPARSE_ROOT
    
    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}

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
    mock_sim.terms = terms
    mock_sim.system = mol_system
    mock_sim.exp = evv_experiment()
    mock_sim.vib_ana_setup = vib_ana
    
    mock_sim.addPropEvalSetup(eval_uniform=calc_setup)
    mock_sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    mock_sim.dressPropsWithSetup()
    mock_sim.setAxisChoiceAndTranslateTerms(axes_choice)

    mock_sim.spec_eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    mock_sim.getResults(obtainer=wilson_data_obtainer)

    print(mock_sim.is_ready)
    print(mock_sim.is_configured)
    
    print(mock_sim.vib_ana_setup.nc_sqrt_eigval)
    mock_sim.evaluate()

    # assert 'timing' in mock_sim.diagn
    # assert 'intermediates' in mock_sim.diagn  # Now included!
    # assert len(mock_sim.diagn['intermediates']['prep_terms']) == 14
    # assert len(mock_sim.diagn['intermediates']['all_features']) == 60
    # assert len(mock_sim.diagn['intermediates']) == 11  # All steps
    
    np.set_printoptions(linewidth=280, precision=3)

    # print(mock_sim.diagn.keys())

