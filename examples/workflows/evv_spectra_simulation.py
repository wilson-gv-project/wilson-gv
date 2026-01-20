"""
Single EVV spectrum workflow.

1. Set up an experiment -> get derived EVV terms
2. Configure calculation:
    - molecular system - label
    - vibrational analysis: regime='GVPT2', vibana_own_analysis='anharm'
    - calc_setup: to retrieve data from the QC program outputs (or submit, get outputs then retrieve)
    - 
"""
import wilson_suite as ws
from wilson_suite.fixtures import evv_experiment
from wilson_suite.wilson_utils.some_reprs import make_SpectralAxisSet

# assuming this function will set up and return a correct EVV experiment
EVV_EXPERIMENT = evv_experiment()
# BTW: valid_axis_combs[0] is unclear API from the POV of the user
# TODO: need to have an API for phasematching condition choice or what?
EVV_EXPERIMENT.valid_axis_combs[0].present_spectral_axis_choices()

# setting up a SpectralAxisSet - axes based on possible combination of independent variables
axes_choice: ws.main.spectrum_abstractions.SpectralAxisSet = make_SpectralAxisSet({'A': [-1], 'B': [-1, 2]})
# now would be useful to check wheather constructed SpectralAxisSet makes sense here - is it in valid_axis_combs?
print(axes_choice)
exit()

DERIVED_EVV_TERMS = ws.derive.derive.get_fully_enhanced_terms(experiment=EVV_EXPERIMENT)
axes_choice = EVV_EXPERIMENT.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}
print()


# would always be a user input - ?
MOLECULAR_SYSTEM = ws.main.abstractions.MolecularSystem(name='h2o', natoms=3)

# user configs
vib_ana = ws.main.abstractions.VibAnaSetup(regime='GVPT2', vibana_own_analysis='anharm')

from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box

bounds_dict = {'A': (1000., 3100.), 'B': (-100., 2500.)}

spectral_window = SpectralWindow(box=Box(bounds_dict))

evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                        'Gamma': 24.7, 'Gamma_unit': 'cm-1',
                                                        'dynamic_range': 1000,
                                                        'grid_resolution': {'A': 70, 'B': 100}})

eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)
