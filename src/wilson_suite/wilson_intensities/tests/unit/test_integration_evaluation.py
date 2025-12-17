import wilson_suite.wilson_intensities.amplitudes.evaluators as evaluators
import wilson_suite as ws
import numpy as np


def test_terms_evaluator_general_compilation():
    print()
    from .test_domains import get_data_evaluators_tests
    datadict = get_data_evaluators_tests()

    np.set_printoptions(linewidth=180, precision=3)

    r = evaluators.terms_evaluator_general_compilation(**datadict)
    print(r['result'])

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
                                                          'grid_resolution': {'A': 100, 'B': 100}})
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

    from ...amplitudes.evaluators import terms_evaluator_general_compilation
    np.set_printoptions(linewidth=280, precision=3)
    sim.evaluateSpectrum(evaluator=terms_evaluator_general_compilation)

    _, ax_dict = spectral_window.sample_grid(sim.spec_eval_setup.ev_info.grid_resolution)

    from ....wilson_analysis.render.simple_plot import render_spectrum, set_figure, prep_levels
    dyn_range = 1000

    # # render_spectrum(abs(sim.spec)**2, ax_dict['A'], ax_dict['B'], 'integrtest.svg', dyn_range, num_level_ticks=10, nicetitle='yes')
    # levels_nums, levels_ticks, levels_nums_str = prep_levels(
    #                             d_max=np.max(abs(sim.spec)**2),
    #                             dynamic_range=dyn_range,
    #                             num_level_ticks=num_level_ticks
    #                             )
    # fig, ax = set_figure(figsize=(35, 45), font_dict={'size': 20}, to_save=True)
    # cont = ax.contourf(ax_dict['A'], ax_dict['B'], abs(sim.spec)**2,
    #                 levels=levels, cmap=cmap  #'hot_r'
    #                 # , norm=colorbar_norm
    #                 , extend='max'
    #                 )
    
    import matplotlib.pyplot as plt

    plt.imshow(np.log(abs(sim.spec)**2).T, interpolation='none', origin="lower")
    plt.show()
