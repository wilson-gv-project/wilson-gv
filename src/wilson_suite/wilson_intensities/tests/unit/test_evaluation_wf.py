from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import build_experiment_context, build_axis_context
from wilson_suite.wilson_utils.serialization import unpickle_smth_from
import wilson_suite as ws
import numpy as np

def test_build_experiment_context():

    print()
    from ....fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)

    from wilson_suite.wilson_utils.some_reprs import make_SpectralAxisSet
    axes_choice: ws.main.spectrum_abstractions.SpectralAxisSet = make_SpectralAxisSet({'A': [1], 'B': [-1,2]}) # {'A': [(1,)], 'B': [(-1,), (2,)]}

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

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    
    bounds_dict = {'A': (0., 5000.), 'B': (0., 5000.)}

    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 4.7, 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 7, 'B': 10},
                                                          'minimum_box_padding': 30.})
    
    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    sim.addSpecEvalSetup(eval_setup)

    sim.vib_ana_setup.set_include_modes_list()

    print('simulation.exp.polarization_avg_vector', sim.exp.polarization_avg_vector)
    sim.evaluate()

    wf1 = ws.intensities.amplitudes.evaluation_wf.EvaluationWorkflow(sim)

    print(type(sim.terms))
    r = build_experiment_context(sim)
    print(r.__dict__.keys())
    print()
    for k,v in r.need_precalc.items():
        print(k)
        print(v)
        print('-------')
    
    results  = wf1.evaluate()
    print(results.__dict__.keys())
    
    np.set_printoptions(suppress=True)
    print(results.result)
    assert np.allclose(results.result, sim.spec)
    # print(np.abs(results.result)**2)
