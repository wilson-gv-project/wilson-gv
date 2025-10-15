import wilson_suite as ws
from wilson_suite.wilson_main.main_functions import do_anharmonic_analysis
from wilson_suite.wilson_utils.paths import SUITE_ROOT
from CQCParse.utils import PKG_ROOT as CQCPARSE_ROOT

'''

pulse_1 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.04, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=1)
pulse_2 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.04, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=2)
pulse_3 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.08, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=3)
pulse_4 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.08, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=4)
pulse_5 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.12, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=5)
pulse_6 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.12, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=6)
pulse_7 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.0, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=7)
pulse_8 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=100.0, cf=0.0, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=8)
pulse_9 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=120.0, cf=0.0, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=9)
pulse_10 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=120.0, cf=0.0, cf_uv=0.072,
                                                   wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=10)
pulse_11 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=120.0, cf=0.0, cf_uv=0.072,
                                                   wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=11)
pulse_12 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=120.0, cf=0.0, cf_uv=0.072,
                                                   wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=12)

pulses = [pulse_1, pulse_2, pulse_3, pulse_4, pulse_5, pulse_6, pulse_7, pulse_8, pulse_9, pulse_10, pulse_11, pulse_12]

field_a = ws.experiment.abstractions.ElectricField(pulses)
order = len(pulses)

detector_a = ws.experiment.abstractions.SpecDetector(detection_method='freq',
                                                     detector_location=[0.0, 0.0, 1.0],
                                                     detection_polarization=[0.0, 0.0, 1.0],
                                                     detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                                     wv_filter=[
                                                         {1: 1, 2: -1, 3: 1, 4: -1, 5: 1, 6: -1, 7: 1, 8: 1, 9: 1, 10: 1, 11: -1, 12: 1}])
'''
'''
pulse_1 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.04, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=1)
pulse_2 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.04, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=2)
pulse_3 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.08, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=3)
pulse_4 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.08, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=4)
pulse_5 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.0, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=5)
pulse_6 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=100.0, cf=0.0, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=6)
pulse_7 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=120.0, cf=0.0, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=7)
pulse_8 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=120.0, cf=0.0, cf_uv=0.072,
                                                   wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=8)
pulse_9 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=120.0, cf=0.0, cf_uv=0.072,
                                                   wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=9)
pulse_10 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=120.0, cf=0.0, cf_uv=0.072,
                                                   wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=10)

pulses = [pulse_1, pulse_2, pulse_3, pulse_4, pulse_5, pulse_6, pulse_7, pulse_8, pulse_9, pulse_10]

field_a = ws.experiment.abstractions.ElectricField(pulses)
order = len(pulses)

detector_a = ws.experiment.abstractions.SpecDetector(detection_method='freq',
                                                     detector_location=[0.0, 0.0, 1.0],
                                                     detection_polarization=[0.0, 0.0, 1.0],
                                                     detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                                     wv_filter=[
                                                         {1: 1, 2: -1, 3: 1, 4: -1, 5: 1, 6: 1, 7: 1, 8: 1, 9: -1, 10: 1}])
'''
'''
pulse_1 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.04, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=1)
pulse_2 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.04, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=2)
pulse_3 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.08, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=3)
pulse_4 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.08, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=4)
pulse_5 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.0, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=5)
pulse_6 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=100.0, cf=0.0, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=6)

pulses = [pulse_1, pulse_2, pulse_3, pulse_4, pulse_5, pulse_6]



field_a = ws.experiment.abstractions.ElectricField(pulses)
order = len(pulses)

detector_a = ws.experiment.abstractions.SpecDetector(detection_method='freq',
                                                     detector_location=[0.0, 0.0, 1.0],
                                                     detection_polarization=[0.0, 0.0, 1.0],
                                                     detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                                     wv_filter=[
                                                         {1: 1, 2: -1, 3: 1, 4: -1, 5: 1, 6: 1}])

'''
'''
pulse_1 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.04, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=1)
pulse_2 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.04, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=2)
pulse_3 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.08, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=3)
pulse_4 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.08, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=4)
pulse_5 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=80.0, cf=0.0, cf_uv=0.02, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=5)

pulses = [pulse_1, pulse_2, pulse_3, pulse_4, pulse_5]

field_a = ws.experiment.abstractions.ElectricField(pulses)
order = len(pulses)

detector_a = ws.experiment.abstractions.SpecDetector(detection_method='freq',
                                                     detector_location=[0.0, 0.0, 1.0],
                                                     detection_polarization=[0.0, 0.0, 1.0],
                                                     detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                                     wv_filter=[
                                                         {1: 1, 2: -1, 3: 1, 4: -1, 5: 1}])
'''
pulse_1 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=50.0, cf=0.0, cf_uv=0.00, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=1)
pulse_2 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=58.0, cf=0.0, cf_uv=0.00, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=2)
pulse_3 = ws.experiment.abstractions.EmPulse(env='impulsive', maxstr=1.0e-5, tc=110.0, cf=0.0, cf_uv=0.08, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=3)

pulses = [pulse_1, pulse_2, pulse_3]



field_a = ws.experiment.abstractions.ElectricField(pulses)
order = len(pulses)

detector_a = ws.experiment.abstractions.SpecDetector(detection_method='freq',
                                                     detector_location=[0.0, 0.0, 1.0],
                                                     detection_polarization=[0.0, 0.0, 1.0],
                                                     detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                                     wv_filter=[
                                                         {1: -1, 2: 1, 3: 1}])

# Push one carrier freq
scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
scan_range_a = [0.0001 * i for i in range(101)]
scan_a = ws.experiment.abstractions.SpecScan(scan_objs=scan_obj_a, range=scan_range_a)

experiment_a = ws.experiment.abstractions.VibExperiment(order=order, field=field_a,
                                                        detector=detector_a,
                                                        scans=[scan_a],
                                                        magn_conditions=[[-1, 2]])



epochs = ws.experiment.abstractions.find_epochs(field_a)

#print('Valid axis combinations', experiment_a.valid_axis_combs)

# TODO: Include ind vars and axes in experiment class as post-init calculated attribute DONE
# TODO: Get terms based on canonical axes (send c.a. as argument to get_fully_enhanced_terms) UPD: DO NOT DO; KEEP CURR FORM
# TODO: Translate to chosen axes (DO THIS IN main or as utility in derive? Decision: In derive) OK
# FOR THAT NEED: Translator, variable equivalence finder in terms of chosen axes DONE
# TODO: Finish new spectral grid class in wilson-main
# TODO: Bring new axis functionality into wilson-main
# ALSO NEED: Function to tell which values need to be specified with chosen axes (that is, indep vars not chosen as axes)
# TODO: Docstrings for all new fns
# TODO: Unit tests for all new fns
# TODO (possibly not on this branch): Full functional consistency in wilson-experiment
# TODO (possibly not on this branch): Make Docstrings for by this new fns, update existing ones wrt. ditto changes
# TODO (possibly not on this branch): Unit tests for all fns thus separated out
# TODO (possibly not on this branch): Full functional consistency in wilson-experiment



terms = ws.derive.main.get_fully_enhanced_terms(experiment=experiment_a)

print(experiment_a.valid_axis_combs)


translated_terms = ws.derive.term_var_translate.translate_terms_to_axis_variables(terms, experiment_a.valid_axis_combs[((-1,), (2,))][3])


calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                 lvl_theory='B3LYP',
                                                 basis_set='cc-pVQZ',
                                                 base_file_loc=CQCPARSE_ROOT + '/CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out')

sim = ws.main.workflow_abstractions.WilsonSimulation()

sim.addExperiment(experiment_a)
sim.addTerms(terms=translated_terms)

# ! 1.1 transform terms from derive to evaluate form
dict_terms = ws.wilson_utils.termdict_from_symb_term.derived_terms_dict_to_dicts(translated_terms)

for i in dict_terms:
    print('\n\n')
    print(i)

quit()

mol_system = ws.main.abstractions.MolecularSystem(name='FORM', natoms=4)
vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
sim.addSystem(mol_system)
sim.addVibAnaSetup(vib_ana)
sim.addPropEvalSetup(eval_uniform=calc_setup)


axis1 = ws.main.spectrum_abstractions.SpectralAxis({'w1': 1})
axis2 = ws.main.spectrum_abstractions.SpectralAxis({'w2': 1})

start = {'x': 250, 'y': 100}
end = {'x': 3850, 'y': 7550}
spacer = {'x': 3.8, 'y': 3.8}

spec_grid = ws.main.spectrum_abstractions.SpectralGrid({'x': axis1, 'y': axis2}, range_style='uniform',
                                                       start=start, end=end, spacer=spacer)

eval_vars = {'w1': ws.main.spectrum_abstractions.EvaluationVariable(range_style='uniform', start=250., end=3850,
                                                                    spacer=3.8).range,
             'w2': ws.main.spectrum_abstractions.EvaluationVariable(range_style='uniform', start=100., end=7550,
                                                                    spacer=3.8).range}
import numpy as np
meshgrids = np.meshgrid(*eval_vars.values(), indexing='ij')

eval_vars_meshgrids = {}
for i, key in enumerate(eval_vars.keys()):
    eval_vars_meshgrids[key] = meshgrids[i]

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

evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'freq_variables': eval_vars_meshgrids,
                                                      'Gamma': 4.7, 'Gamma_unit': 'cm-1'})
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

eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)

sim.addSpecEvalSetup(eval_setup)
sim.setPropsAndMaxStateLvl()  # setting up self.props/sim.props
sim.dressPropsWithSetup()


from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
sim.getResults(obtainer=wilson_data_obtainer)


import numpy as np
np.set_printoptions(precision=4)

from wilson_suite.wilson_intensities.spectrum.evaluators import terms_evaluator
sim.evaluateAsResponseFunction(evaluator=terms_evaluator)
intensities_spec = np.abs(sim.spec) ** 2

hist, bin_edges = np.histogram(intensities_spec, bins=10)

sim.render(renderer=ws.analysis.render.render_spectrum)
