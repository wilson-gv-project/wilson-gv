"""
Notes

Procedure:
1. Set up an experiment.
2. Set up calculation parameters/settings.
3. Set up a simulation object which would control the whole procedure.
4. getResultsFromCalculationBatches - here first call for using already existing data

sim.findPropsAndMaxStateLvl() - ??
sim.dressPropsWithSetup() - ??
sim.makeCalculationBatches() - ??


sim.evaluate(evaluator); evaluator is a function
evaluator(self.system, self.exp, self.terms, self.props, self.spec_eval_setup, self.vib_ana_setup)

"""
import wilson_suite as ws
from wilson_utils.serialization import pickle_this_to, unpickle_smth_from
from wilson_utils.paths import SUITE_ROOT

import logging
logger = logging.getLogger("wilson.")

logger.info('evv_tester_dataclasses.py')

TO_PICKLES = ['VibExperiment', 'ExternalCalcSetup', 'WilsonSimulation']
PKL_FILES = {}

def run():

    pulse_ir_1 = ws.experiment.abstractions.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2 = ws.experiment.abstractions.EmPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1 = ws.experiment.abstractions.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)

    pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]

    field_a = ws.experiment.abstractions.ElectricField(pulses)
    order = len(pulses)

    field_a.findEpochs()

    detector_a = ws.experiment.abstractions.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                                        detection_polarization=[0.0, 0.0, 1.0],
                                                        detection_range=[0.003 + 0.0001*i for i in range(101)],
                                                        wv_filter=[{1: [-1], 2: [1], 3: [1]}]) #, {1: [-1], 2: [1], 3: [1]}

    # Push one carrier freq
    scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a = [0.0001*i for i in range(101)]
    scan_a = ws.experiment.abstractions.SpecScan(scan_obj_a, scan_range_a)

    experiment_a = ws.experiment.abstractions.VibExperiment(order, field_a, detector_a, [scan_a], magn_conditions=[[-1, 2]])
    if 'VibExperiment' in TO_PICKLES:
        pickle_this_to(obj=experiment_a, filenamepkl='vibexp.pkl', save_to=SUITE_ROOT+'/tests/')

        experiment_a = unpickle_smth_from(filenamepkl='vibexp.pkl', load_from=SUITE_ROOT+'/tests/')
        PKL_FILES['VibExperiment'] = 'vibexp.pkl'
    
    calc_setup = ws.main.abstractions.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVQZ')
    if 'ExternalCalcSetup' in TO_PICKLES:
        pickle_this_to(obj=calc_setup, filenamepkl='calcsetup.pkl', save_to=SUITE_ROOT+'/tests/')

        calc_setup = unpickle_smth_from(filenamepkl='calcsetup.pkl', load_from=SUITE_ROOT+'/tests/')
        PKL_FILES['ExternalCalcSetup'] = 'calcsetup.pkl'

    sim = ws.main.abstractions.WilsonSimulation()
    if 'WilsonSimulation_init' in TO_PICKLES:
        pickle_this_to(obj=sim, filenamepkl='sim_init.pkl', save_to=SUITE_ROOT+'/tests/')

        sim = unpickle_smth_from(filenamepkl='sim_init.pkl', load_from=SUITE_ROOT+'/tests/')
        PKL_FILES['WilsonSimulation_init'] = 'sim_init.pkl'
    
    sim.addExperiment(experiment_a)
    sim.getTerms(ws.derive.main.get_fully_enhanced_terms) # here terms are derived
    mol_system = ws.main.abstractions.MolecularSystem(name='FORM', natoms=4)
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_prop_need='none',
                                                        allow_skip_eigvec=True, external_fill_from=calc_setup))
    sim.addPropEvalSetup(eval_uniform=calc_setup)

    axis1 = ws.main.abstractions.SpectralAxis({1: 1})
    axis2 = ws.main.abstractions.SpectralAxis({1: 1, 2: -1})
    start = {1: 250, 2: 100}
    end = {1: 3850, 2: 7550}
    spacer = {1: 3.8, 2: 3.8}
    spec_grid = ws.main.abstractions.SpectralGrid({1: axis1, 2: axis2}, range_style='uniform',
                                                start=start, end=end, spacer=spacer)
    evi = {'dynrange': 500, 'Gamma': 4.7, 'diag_margin': 5., 'maxmax': None}
    rndi = {'num_level_ticks': 15}
    eval_setup = ws.main.abstractions.SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)
    sim.addSpecEvalSetup(eval_setup)

    sim.findPropsAndMaxStateLvl() # setting up self.props/sim.props
    logger.debug(f'\nafter findPropsAndMaxStateLvl {sim.props}\n')

    sim.dressPropsWithSetup()
    sim.makeCalculationBatches()
    sim.getResultsFromCalculationBatches(source_type='vault',
                                        source_loc=ws.intensities.utils.get_package_root()
                                                    + '/../tests/test_database/mini_files_database.csv' )
    logger.debug(f'\nafter getResultsFromCalculationBatches {sim.props}\n')


    import numpy as np
    np.set_printoptions(precision=4)

    if 'WilsonSimulation_mid' in TO_PICKLES:
        pickle_this_to(obj=sim, filenamepkl='sim_mid.pkl', save_to=SUITE_ROOT+'/tests/')

        sim = unpickle_smth_from(filenamepkl='sim_mid.pkl', load_from=SUITE_ROOT+'/tests/')
        PKL_FILES['WilsonSimulation_mid'] = 'sim_mid.pkl'

    logger.info('  >>> Going to evaluate now...\n')
    sim.evaluateAsResponseFunction(evaluator=ws.intensities.spectrum.evaluators.terms_evaluator)
    intensities_spec = np.abs(sim.spec)**2
    logger.info(f'np.max(np.abs(sim.spec)**2) {np.max(np.abs(sim.spec)**2)}')

    logger.debug('\n=====================================================')
    logger.info('\n  >>> And now rendering...\n')

    from wilson_analysis.render import render_spectrum

    hist, bin_edges = np.histogram(intensities_spec, bins=10)
    logger.debug(f"Histogram counts: {hist}")
    logger.debug(f"Bin edges: {bin_edges}\n")

    logger.debug(f'np.max(intensities): {np.max(intensities_spec):.4e}')

    dict_meshes = spec_grid.make_mesh_numpy()
    render_spectrum(intensities_spec, dict_meshes[1], dict_meshes[2],
                    filename='yo_terms_derive_ACAC.svg', dynamic_range=100,
                    nicetitle='TermsEvaluator')

    from wilson_utils.serialization import check_if_jsonsafe
    check_if_jsonsafe(sim.to_dict())

    ws_root = ws.intensities.utils.get_package_root() + '/../../'

    # writing JSON file
    sim.writeToJsonFile(ws_root+'/tests/WilsonSimulation.json')

    # pickling
    import pickle
    with open(ws_root+"/tests/wilsonsim0.pkl", "wb") as f:
        pickle.dump(sim, f)

    with open(ws_root+"/tests/wilsonsim0.pkl", "rb") as f:
        loaded_wilsonsim0 = pickle.load(f)
    
    if 'WilsonSimulation_final' in TO_PICKLES:
        pickle_this_to(obj=sim, filenamepkl='sim_final.pkl', save_to=SUITE_ROOT+'/tests/')

        sim = unpickle_smth_from(filenamepkl='sim_final.pkl', load_from=SUITE_ROOT+'/tests/')
        PKL_FILES['WilsonSimulation_final'] = 'sim_final.pkl'
    
    # apparently numpy can't compare complex numbers
    assert np.allclose(np.abs(sim.spec), np.abs(loaded_wilsonsim0.spec))

    return sim