import wilson_suite as ws
import numpy as np
np.set_printoptions(linewidth=280, precision=1)


def test_full_integration():
    print()
    from ...fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    # ?? why no other axes choice works?
    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}
    print()
    evv_exp.valid_axis_combs[0].present_spectral_axis_choices()
    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian', 
                                                     lvl_theory='B3LYP', 
                                                     basis_set='cc-pVQZ', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')

    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms) # terms

    mol_system = ws.main.abstractions.MolecularSystem(name='FORM', natoms=4)

    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    sim.setAxisChoiceAndTranslateTerms(axes_choice)

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    
    bounds_dict = {'A': (1000., 3100.), 'B': (-100., 2500.)}

    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 24.7, 'Gamma_unit': 'cm-1',
                                                          'dynamic_range': 1000,
                                                          'grid_resolution': {'A': 70, 'B': 100}})
    
    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    sim.addSpecEvalSetup(eval_setup)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)

    sim.evaluate()

    style_config = ws.main.spectrum_abstractions.PlotConfig(tick_step=100.)
    rnd = ws.main.spectrum_abstractions.RenderingInfo(filename='f_hcoh.svg', 
                                                      style_config=style_config)
    sim.spec_eval_setup.rnd_info = rnd

    print(f"\nMean: {np.mean(np.abs(sim.spec)**2):.3e}")
    print(f"Standard Deviation: {np.std(np.abs(sim.spec)**2):.3e}")
    print(f"Minimum Value: {np.min(np.abs(sim.spec)**2):.3e}")
    print(f"Maximum Value: {np.max(np.abs(sim.spec)**2):.3e}")

    sim.render(renderer=ws.analysis.render.render_spectrum)

def test_full_integration_other_axes_choice():
    print()
    from ...fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    # ?? why no other axes choice works?
    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[0] # {'A': [(2,)], 'B': [(-1,), (2,)]}
    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian', 
                                                     lvl_theory='B3LYP', 
                                                     basis_set='cc-pVQZ', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')

    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms) # terms

    mol_system = ws.main.abstractions.MolecularSystem(name='FORM', natoms=4)

    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    sim.setAxisChoiceAndTranslateTerms(axes_choice)

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    
    bounds_dict = {'A': (1000., 3100.), 'B': (-100., 2500.)}

    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 24.7, 'Gamma_unit': 'cm-1',
                                                          'dynamic_range': 1000,
                                                          'grid_resolution': {'A': 70, 'B': 100}})
    
    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    sim.addSpecEvalSetup(eval_setup)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)
    import pytest
    with pytest.raises(ValueError) as error:
        sim.evaluate()
    assert str(error.value) == "Failed at 'place_in_specwindow': This SpectralWindow does not contain any features. Change the bounds of the window or use different terms. EvaluationWorkflow instanse was saved to `eval_wf.pkl`."


def test_smth():
    from wilson_suite.wilson_utils.serialization import unpickle_smth_from
    wf: ws.intensities.amplitudes.evaluation_wf.EvaluationWorkflow = unpickle_smth_from('eval_wf.pkl')
    print(wf.artifacts.features)


def test_full_integration_H2O_molecule():
    print()
    from ...fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}

    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                     lvl_theory='HF', 
                                                     basis_set='STO-3G', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_h2o_HF_STO3G.out')

    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms) # terms

    mol_system = ws.main.abstractions.MolecularSystem(name='h2o', natoms=3)

    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    sim.setAxisChoiceAndTranslateTerms(axes_choice)

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    
    bounds_dict = {'A': (0., 5000.), 'B': (0., 5000.)}

    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 4.7, 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 10, 'B': 10}})
    
    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    sim.addSpecEvalSetup(eval_setup)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)

    sim.evaluate()

    style_config = ws.main.spectrum_abstractions.PlotConfig(tick_step=50.)
    rnd = ws.main.spectrum_abstractions.RenderingInfo(filename='f_h2o.svg', 
                                                      style_config=style_config)
    sim.spec_eval_setup.rnd_info = rnd

    print(f"\nMean: {np.mean(np.abs(sim.spec)**2):.3e}")
    print(f"Standard Deviation: {np.std(np.abs(sim.spec)**2):.3e}")
    print(f"Minimum Value: {np.min(np.abs(sim.spec)**2):.3e}")
    print(f"Maximum Value: {np.max(np.abs(sim.spec)**2):.3e}")

    sim.render(renderer=ws.analysis.render.render_spectrum)

