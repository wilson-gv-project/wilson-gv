"""
Single EVV spectrum workflow.

1. Set up an experiment -> get derived EVV terms
    - phasematching condition - should be set as config option?
    - axes choices options
2. Configure calculation:
    - molecular system - label
    - vibrational analysis: regime='GVPT2', vibana_own_analysis='anharm'
    - calc_setup: to retrieve data from the QC program outputs (or submit, get outputs then retrieve)
    - 

EVV_EXPERIMENT.magn_conditions -- w2 > w1


core_paper1_setup = ''

-- fixed
experiment: EVV + phasematching (-1, 2, 3) + magn_condition 'w2>w1'
vib_analysis: anharmonic GVPT2
axes_choice: based on experiment but here either (w1,w2) or (w1,w2-w1)
SpecEvalSetup fixed: 
    PlotConfig
    RenderingInfo[all except reference max]
    EvaluationInfo[all except Gamma and dynamic range?]

-- variables:
system
calc_setup - DataOriginInfo
SpecEvalSetup variables: 
    RenderingInfo[reference max]
    EvaluationInfo[Gamma and dynamic range]

"""
import wilson_suite as ws
from wilson_suite.fixtures import evv_experiment
from wilson_suite.wilson_utils.some_reprs import make_SpectralAxisSet
from wilson_suite.wilson_experiment.indep_vars_and_axes import PhaseMatchingCondition, SignedPulseTuple
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer

# ---------- PREPARE PARTS FOR WilsonSimulation
# assuming this function will set up and return a correct EVV experiment
EVV_EXPERIMENT = evv_experiment()


# TODO: need to have an API for phasematching condition choice as a part of configs? 
# TODO: also, after the axis choice phasematching condition will be defined as well?
EVV_PHASEMATCH_COND = PhaseMatchingCondition(pulses=SignedPulseTuple(pulse_refs=(-1, 2, 3)), id=0)
assert EVV_EXPERIMENT.relevant_phasematch[0] == EVV_PHASEMATCH_COND

# BTW: valid_axis_combs[0] is unclear API from the POV of the user
# EVV_EXPERIMENT.valid_axis_combs[0].present_spectral_axis_choices()

# setting up a SpectralAxisSet - axes based on possible combination of independent variables
axes_choice: ws.main.spectrum_abstractions.SpectralAxisSet = make_SpectralAxisSet({'A': [-1], 'B': [-1, 2]})
# now would be useful to check wheather constructed SpectralAxisSet makes sense here - is it in valid_axis_combs?


# original derived terms with independent variables - needed for the manuscript
DERIVED_EVV_TERMS = ws.derive.derive.get_fully_enhanced_terms(experiment=EVV_EXPERIMENT)
# next step is to translate terms wrt axes_choice

# would always be a user input - ?
molecular_system = ws.main.abstractions.MolecularSystem(name='h2o', natoms=3)

# user configs
vib_ana = ws.main.abstractions.VibAnaSetup(regime='GVPT2', vibana_own_analysis='anharm')
# doesn't have to have a molecular system

# set up SpectralWindow -- should be flexible/changable in wilsonsim object
bounds_dict = {'A': (1000., 3100.), 'B': (-100., 2500.)}
spec_box = ws.intensities.amplitudes.spectrum_composition.Box(bounds=bounds_dict)
spectral_window = ws.intensities.amplitudes.spectrum_composition.SpectralWindow(box=spec_box)

# set up EvaluationInfo
evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                        'Gamma': 24.7, 'Gamma_unit': 'cm-1',
                                                        'dynamic_range': 1000,
                                                        'grid_resolution': {'A': 70, 'B': 100}})

# put configs together in SpecEvalSetup
eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

# DataOriginInfo - to get data from QC program outputs
base_filepath = ''
calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian', 
                                                    lvl_theory='B3LYP', 
                                                    basis_set='cc-pVQZ', 
                                                    base_file_loc=base_filepath)

# ---------- WilsonSimulation
sim = ws.main.workflow_abstractions.WilsonSimulation()

# -- setting attributes
sim.addExperiment(experiment=EVV_EXPERIMENT)
sim.addTerms(terms=DERIVED_EVV_TERMS)
sim.addSystem(system=molecular_system)
sim.addVibAnaSetup(vib_ana)
sim.addPropEvalSetup(eval_uniform=calc_setup)
sim.addSpecEvalSetup(eval_setup)


# ---- chng of state
sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
# ---- chng of state
sim.dressPropsWithSetup()
# ---- chng of state
sim.setAxisChoiceAndTranslateTerms(axes_choice) # set axes and prepare terms for evaluation 

# ---- chng of state
sim.getResults(obtainer=wilson_data_obtainer)

# ---- chng of state? or just setting attributes?
sim.evaluate()

fig_file = '.svg'
style_config = ws.main.spectrum_abstractions.PlotConfig(tick_step=50.)
rnd = ws.main.spectrum_abstractions.RenderingInfo(filename=fig_file, style_config=style_config)

sim.spec_eval_setup.rnd_info = rnd # add RenderingInfo to SpecEvalSetup

# ---- just setting attributes?
sim.render(renderer=ws.analysis.render.render_spectrum)
