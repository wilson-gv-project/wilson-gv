import wilson_suite as ws
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_suite.wilson_main.abstractions import VibAnaSetup

def test_terms_evaluator_general_compilation():
    print()
    from ..unit.test_domains import get_data_evaluators_tests
    datadict = get_data_evaluators_tests()
    # 'system', 'vib_ana_setup', 'derived_terms', 'props', 
    # 'experiment', 'spec_eval_setup', 'domain_distance_thresholds'
    
    np.set_printoptions(linewidth=180, precision=3)

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
    bounds_dict = {'B': (900., 900.), 'A': (1864., 1864.)}
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 1., 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 1, 'B': 1}})
    mock_sim = WilsonSimulation()
    mock_sim.terms = evv_terms
    mock_sim.system = mol_system
    mock_sim.exp = evv_experiment()
    mock_sim.vib_ana_setup = vib_ana
    
    mock_sim.addPropEvalSetup(eval_uniform=calc_setup)
    mock_sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    mock_sim.dressPropsWithSetup()

    mock_sim.spec_eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    # use simple model data
    mock_sim.system = datadict['system']
    mock_sim.props = datadict['props']
    mock_sim.vib_ana_setup: 'VibAnaSetup' = datadict['vib_ana_setup']

    print(mock_sim.is_ready)

    mock_sim.evaluate()

    # print(mock_sim._workflow.artifacts.spec_window.full_features)
    # print(len(mock_sim._workflow.artifacts.spec_window.full_features))

    # domain = mock_sim._workflow.artifacts.spec_window.find_clusters_by_featboxes()[0]
    # print(domain)

    region = mock_sim._workflow.artifacts.regions[0]
    feat_coeff = region.domain.full_features[0].amplitude_coeff

    np.set_printoptions(linewidth=280, precision=1)
    for k,v in mock_sim._workflow.artifacts.grid_manager.full_grid.items():
        print(k)
        print(v)
    print('\n==========')
    
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    r_res = ws.intensities.amplitudes.evaluation_wf.evaluate_region(region, 
                                                            mock_sim._workflow.artifacts.vib_data, 
                                                            mock_sim._workflow.artifacts.vibdiff_cache, 
                                                            convNu2Ene(mock_sim.spec_eval_setup.ev_info.Gamma))
    ref_res = np.array([1/(-1j*convNu2Ene(1.))/(-1j*convNu2Ene(1.)) * feat_coeff])
    assert np.allclose(r_res, ref_res)
    
    assert np.allclose(ref_res, mock_sim.spec['result'])


def test_full_integration():
    print()
    from ....fixtures import evv_experiment
    from CQCParse.utils import PKG_ROOT as CQCPARSE_ROOT

    evv_experiment = evv_experiment()
    terms = ws.derive.main.get_fully_enhanced_terms(experiment=evv_experiment)
    axes_choice = evv_experiment.valid_axis_combs[((-1,), (2,))][3] # {'A': [(2,)], 'B': [(-1,), (2,)]}
    evv_terms =  ws.derive.term_var_translate.translate_terms_to_axis_variables(terms, axes_choice)

    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian', 
                                                     lvl_theory='B3LYP', 
                                                     basis_set='cc-pVQZ', 
                                                     base_file_loc=CQCPARSE_ROOT+'/CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out')

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

    from wilson_suite.wilson_analysis.render.spectrum_renderer import PlotConfig, NormalizationType
    style_config = PlotConfig(
        figsize=(35, 45),
        label_fontsize=30,
        font_dict={'size': 24},
        colormap='hot_r',  # Better contrast colormap
        saturation_color='#FF00FF',
        dpi=350,
        tick_step=200.0,  # Step size for both axes ticks
        equal_aspect=True,  # Force equal aspect ratio for axes
        no_data_color='#E0E0E0',  # Light gray
        below_range_color='#F8F8F8',  # Very light gray
        data_edge_color='black',
        data_edge_width=0.75,
        y_min=0,
        y_max=4500,
        colorbar_main_label="Intensity",
        colorbar_padding=0.02,
        show_top_ticks=True,
        show_right_ticks=True,
        x_tick_rotation=45,
        colormap_spacing='log',
        colormap_power=0.5,
    )

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    
    bounds_dict = {'A': (0., 5000.), 'B': (0., 5000.)}

    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 4.7, 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 10, 'B': 10}})
    rndi = ws.main.spectrum_abstractions.RenderingInfo(**{'intensity_normalization_type': NormalizationType.LOG_RATIO,
                                                 'dynamic_range': 500, 
                                                 'num_levels': 15, 
                                                 'reference_max': None,
                                                 'spec_data_operations': 'abs()**2',
                                                 'projection': '2d', 
                                                 'filename': 'smth.svg',
                                                 'backend': 'matplotlib',
                                                 'to_save': True,
                                                 'style_config': style_config})
    
    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi, rnd_info=rndi)

    sim.addSpecEvalSetup(eval_setup)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)

    sim.evaluate()

    np.set_printoptions(linewidth=280, precision=1)

    print(sim.spec['A'].T)
    print(sim.spec['B'].T)
    print(sim.spec['result'].T)

    import matplotlib.pyplot as plt

    plt.imshow(np.log(abs(sim.spec['result'])**2).T, interpolation='none', origin="upper") # now folows A,B arrays - down and to the left
    plt.show()

