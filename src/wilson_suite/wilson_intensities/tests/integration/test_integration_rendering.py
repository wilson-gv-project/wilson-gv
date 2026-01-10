import wilson_suite as ws
import numpy as np

def test_full_integration():
    print()
    from ....fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_experiment = evv_experiment()
    terms = ws.derive.main.get_fully_enhanced_terms(experiment=evv_experiment)
    axes_choice = evv_experiment.valid_axis_combs[((-1,), (2,))][3] # {'A': [(2,)], 'B': [(-1,), (2,)]}
    evv_terms =  ws.derive.term_var_translate.translate_terms_to_axis_variables(terms, axes_choice)

    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian', 
                                                     lvl_theory='B3LYP', 
                                                     basis_set='cc-pVQZ', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')

    from wilson_suite.wilson_main.wf import WilsonSimulation
    sim = WilsonSimulation()

    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_experiment)
    sim.addTerms(terms=evv_terms) # terms

    mol_system = ws.main.abstractions.MolecularSystem(name='FORM', natoms=4)

    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    
    bounds_dict = {'A': (1100., 5000.), 'B': (-150., 3000.)}

    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 54.7, 'Gamma_unit': 'cm-1',
                                                          'dynamic_range': 1000,
                                                          'grid_resolution': {'A': 150, 'B': 180}})
    
    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    sim.addSpecEvalSetup(eval_setup)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)

    sim.evaluate()
    print(sim._workflow.artifacts.spec_window.full_features)

    style_config = ws.main.spectrum_abstractions.PlotConfig(tick_step=100.)
    rnd = ws.main.spectrum_abstractions.RenderingInfo(filename='f_hcoh.svg', 
                                                      style_config=style_config)
    sim.spec_eval_setup.rnd_info = rnd
    sim.spec_eval_setup.grid = {'A': sim.spec['A'], 'B': sim.spec['B']}
    sim.spec = sim.spec['result']

    feats_in_window = sim._workflow.artifacts.spec_window.full_features
    for i in [(f.feat_box, f.amplitude_coeff) for f in feats_in_window]:
        print(i[0].bounds, i[1])

    sim.render(renderer=ws.analysis.render.render_spectrum)


def test_full_integration_H2O_molecule():
    print()
    from ....fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_experiment = evv_experiment()
    terms = ws.derive.main.get_fully_enhanced_terms(experiment=evv_experiment)
    axes_choice = evv_experiment.valid_axis_combs[((-1,), (2,))][3] # {'A': [(2,)], 'B': [(-1,), (2,)]}
    evv_terms =  ws.derive.term_var_translate.translate_terms_to_axis_variables(terms, axes_choice)

    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian', 
                                                     lvl_theory='HF', 
                                                     basis_set='STO-3G', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_h2o_HF_STO3G.out')

    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_experiment)
    sim.addTerms(terms=evv_terms)

    mol_system = ws.main.abstractions.MolecularSystem(name='h2o', natoms=3)

    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    bounds_dict = {'A': (3400., 5000.), 'B': (-100., 2400.)}
    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 24.7, 'Gamma_unit': 'cm-1',
                                                          'dynamic_range': 30,
                                                          'grid_resolution': {'A': 100, 'B': 100}})
    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)
    sim.addSpecEvalSetup(eval_setup)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)

    sim.evaluate()

    style_config = ws.main.spectrum_abstractions.PlotConfig(tick_step=50.)
    rnd = ws.main.spectrum_abstractions.RenderingInfo(filename='f_h2o.svg', 
                                                      style_config=style_config)
    sim.spec_eval_setup.rnd_info = rnd
    sim.spec_eval_setup.grid = {'A': sim.spec['A'], 'B': sim.spec['B']}
    sim.spec = sim.spec['result']

    feats_in_window = sim._workflow.artifacts.spec_window.full_features
    for i in [(f.feat_box, f.amplitude_coeff) for f in feats_in_window]:
        print(i[0].bounds, i[1])

    sim.render(renderer=ws.analysis.render.render_spectrum)

