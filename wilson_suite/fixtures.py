import wilson_main.abstractions as abst_main
import wilson_derive as ws_derive
from wilson_derive.abstractions import VibPerturbedTerm
import wilson_experiment as ws_experiment
import wilson_utils as ws_utils

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


# QC calculations/vibana parameters
mol_system = abst_main.MolecularSystem(name='FORM', natoms=4)
calc_setup = abst_main.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVQZ')
calc_setup1 = abst_main.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVTZ')
calc_setup2 = abst_main.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVDZ')

vibanasetup_none = abst_main.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_prop_need='none',
                                         allow_skip_eigvec=True, external_fill_from=calc_setup)
vibanasetup_anharm = abst_main.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_prop_need='anharm',
                                           allow_skip_eigvec=True, external_fill_from=calc_setup)

eval_prop_specify = {'cff': calc_setup2, 'qff': calc_setup2, 
                     'dipgrad': calc_setup1, 'diphess': calc_setup2, 
                     'polgrad': calc_setup1, 'polhess': calc_setup2}


mol_props_evv_none = [abst_main.MolecularProperty(prop_spec={'ops': ('g', 'f'), 'freq': (0.0, 0.0)}, trivial_name='dipgrad'), 
                      abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'f', 'f'), 'freq': (0.0, 0.0, 0.0, 0.0)}, trivial_name='polhess',), 
                      abst_main.MolecularProperty(prop_spec={'ops': ('g', 'f', 'f'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='polgrad'), 
                      abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'f'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='diphess'), 
                      abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'g'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='cff')]

mol_props_evv_amharm = [abst_main.MolecularProperty(prop_spec={'ops': ('g', 'f'), 'freq': (0.0, 0.0)}, trivial_name='dipgrad'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'f', 'f'), 'freq': (0.0, 0.0, 0.0, 0.0)}, trivial_name='polhess'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'f', 'f'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='polgrad'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'f'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='diphess'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'g'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='cff'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'g', 'g'), 'freq': (0.0, 0.0, 0.0, 0.0)}, trivial_name='qff'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('r',), 'freq': 0.0}, trivial_name='B'), 
                        abst_main.MolecularProperty(prop_spec={'ops': ('g', 'g', 'r'), 'freq': (0.0, 0.0, 0.0)}, trivial_name='coriolis')]

# spectrum eval/render parameters

def spectral_grid():
    axis1 = abst_main.SpectralAxis({1: 1})
    axis2 = abst_main.SpectralAxis({1: 1, 2: -1})
    start = {1: 250, 2: 100}
    end = {1: 3850, 2: 7550}
    spacer = {1: 3.8, 2: 3.8}
    return abst_main.SpectralGrid({1: axis1, 2: axis2}, range_style='uniform',
                                  start=start, end=end, spacer=spacer)

def spec_eval_setup(spec_grid):
    evi = {'dynrange': 500, 'Gamma': 4.7, 'diag_margin': 5., 'maxmax': None}
    rndi = {'num_level_ticks': 15}
    return abst_main.SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)


def makeWilsonSim(experiment, mol_system, vibanasetup, calc_setup, eval_prop_specify: dict, 
                  spec_eval_setup: abst_main.SpecEvalSetup = None):

    sim = abst_main.WilsonSimulation()

    # terms for calculation
    sim.addExperiment(experiment)
    sim.getTerms(ws_derive.main.get_fully_enhanced_terms)

    # QC data/vibana parameters
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vibanasetup)
    sim.addPropEvalSetup(eval_uniform=calc_setup, eval_by_prop_name=eval_prop_specify)

    sim.findPropsAndMaxStateLvl() # setting up self.props/sim.props
    logger.debug(f'\nafter findPropsAndMaxStateLvl {sim.props}\n')

    sim.dressPropsWithSetup()
    sim.makeCalculationBatches()
    
    sim.getResultsFromCalculationBatches(source_type='vault',
                                        source_loc=ws_utils.paths.SUITE_ROOT
                                                    + '/wilson_intensities/tests/test_database/mini_files_database.csv' )
    

    if spec_eval_setup is not None:
        # spectrum eval/render parameters
        sim.addSpecEvalSetup(spec_eval_setup)
    
    return sim

"""
1. Ready for evaluation, WilsonSimulation instance should have:
    a. self.system, self.terms, self.props, self.spec_eval_setup, self.vib_ana_setup
    b. self.exp if evaluateFull()

2. Ready for rendering, WilsonSimulation instance should have:
    a. self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup

3. Ready to load numerical data (states, properties) (getResultsFromCalculationBatches()):
    a. self.calc_batches, self.vib_ana_setup., self.props
    b. [if self.vib_ana_setup.external_fill_from is not None: -- getResults() needs to call vibanalysis if no external fill]
    c. getResultsFrom...() should not be needing any self. access? static methods?
    d. getResults of CalculationBatch
     
4. Ready to makeCalculationBatches(): - this method is copying MolProps to batches where they are saved
    a. self.props - props has all possible props. not getting calc batch for getting states? if requesting own vibana for stetes, how to get hessian?
"""