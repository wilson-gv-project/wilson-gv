from .wilson_main import abstractions as abst_main
from . import wilson_derive as ws_derive
from .wilson_derive.abstractions import VibPerturbedTerm
from . import wilson_experiment as ws_experiment
from . import wilson_utils as ws_utils

import logging
# wilson. - for hierarchy of loggers
logger = logging.getLogger("wilson.")


def evv_experiment() -> ws_experiment.abstractions.VibExperiment:
    """
    Returns VibExperiment instance for EVV experiment
    """
    pulse_ir_1 = ws_experiment.abstractions.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2 = ws_experiment.abstractions.EmPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1 = ws_experiment.abstractions.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)

    pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]

    field_a = ws_experiment.abstractions.ElectricField(pulses)
    order = len(pulses)

    field_a.findEpochs()

    detector_a = ws_experiment.abstractions.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                                        detection_polarization=[0.0, 0.0, 1.0],
                                                        detection_range=[0.003 + 0.0001*i for i in range(101)],
                                                        wv_filter=[{1: [-1], 2: [1], 3: [1]}]) #, {1: [-1], 2: [1], 3: [1]}

    # Push one carrier freq
    scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a = [0.0001*i for i in range(101)]
    scan_a = ws_experiment.abstractions.SpecScan(scan_obj_a, scan_range_a)
    experiment_a = ws_experiment.abstractions.VibExperiment(order, field_a, detector_a, [scan_a], magn_conditions=[[-1, 2]])
    return experiment_a

def evv_terms() -> list[VibPerturbedTerm]:
    """
    Returns EVV terms derived with wilson_derive
    """
    sim = abst_main.WilsonSimulation()
    sim.addExperiment(experiment=evv_experiment())
    sim.getTerms(ws_derive.main.get_fully_enhanced_terms)
    return sim.terms

def SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH():
    # TODO for MR
    import wilson_suite as ws

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
    
    terms = ws.derive.main.get_fully_enhanced_terms(experiment=experiment_a)
    translated_terms = ws.derive.term_var_translate.translate_terms_to_axis_variables(terms, experiment_a.valid_axis_combs[((-1,), (2,))][3])

    return translated_terms



# # QC calculations/vibana parameters
# mol_system = abst_main.MolecularSystem(name='FORM', natoms=4)


# # spectrum eval/render parameters

# def spectral_grid():
#     axis1 = abst_main.SpectralAxis({1: 1})
#     axis2 = abst_main.SpectralAxis({1: 1, 2: -1})
#     start = {1: 250, 2: 100}
#     end = {1: 3850, 2: 7550}
#     spacer = {1: 230.8, 2: 230.8}
#     return abst_main.SpectralGrid({1: axis1, 2: axis2}, range_style='uniform',
#                                   start=start, end=end, spacer=spacer)

# def spec_eval_setup(spec_grid):
#     evi = {'dynrange': 500, 'Gamma': 4.7, 'diag_margin': 5., 'maxmax': None}
#     rndi = {'num_level_ticks': 15}
#     return abst_main.SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)


# def makeWilsonSimInstance(experiment, mol_system, vibanasetup, calc_setup, eval_prop_specify: dict, 
#                   spec_eval_setup: abst_main.SpecEvalSetup = None):
#     """
#     Should create an instance of a WilsonSimulation 
#     with a certain state based on the needs
    
#     """
#     # sim = abst_main.WilsonSimulation()

#     raise NotImplementedError('This fixture-making function is not yet implemented.')

# """
# 1. Ready for evaluation, WilsonSimulation instance should have:
#     a. self.system, self.terms, self.props, self.spec_eval_setup, self.vib_ana_setup
#     b. self.exp if evaluateFull()

# 2. Ready for rendering, WilsonSimulation instance should have:
#     a. self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup

# 3. Ready to load numerical data (states, properties) (getResultsFromCalculationBatches()):
#     a. self.calc_batches, self.vib_ana_setup., self.props
#     b. [if self.vib_ana_setup.external_fill_from is not None: -- getResults() needs to call vibanalysis if no external fill]
#     c. getResultsFrom...() should not be needing any self. access? static methods?
#     d. getResults of CalculationBatch
     
# 4. Ready to makeCalculationBatches(): - this method is copying MolProps to batches where they are saved
#     a. self.props - props has all possible props. not getting calc batch for getting states? if requesting own vibana for stetes, how to get hessian?
# """