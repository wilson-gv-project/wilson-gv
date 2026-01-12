import wilson_suite as ws
import numpy as np


def test_evaluation_general_customdata_1elterm():
    print()
    from ..unit.test_domains import get_data_evaluators_tests
    datadict = get_data_evaluators_tests()
    # 'system', 'vib_ana_setup', 'derived_terms', 'props', 
    # 'experiment', 'spec_eval_setup', 'domain_distance_thresholds'
    
    np.set_printoptions(linewidth=180, precision=3)

    from wilson_suite.wilson_main.wf import WilsonSimulation
    # from wilson_suite.wilson_main.workflow_abstractions import WilsonSimulation

    from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
    from ....fixtures import evv_experiment

    evv_exp = evv_experiment()

    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}

    bounds_dict = {'B': (900., 900.), 'A': (1864., 1864.)}
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 1., 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 1, 'B': 1}})
    mock_sim = WilsonSimulation()
    mock_sim.terms = get_fully_enhanced_terms(experiment=evv_exp)

    mock_sim.exp = evv_exp

    mock_sim.setAxisChoiceAndTranslateTerms(axes_choice)

    mock_sim.spec_eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    # use simple model data
    mock_sim.system = datadict['system']
    mock_sim.props = datadict['props']
    mock_sim.vib_ana_setup = datadict['vib_ana_setup']
    mock_sim.vib_ana_setup.max_state_lvl = 3 # there is an issue for the underlying reason for this

    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_flat
    flat_dict = derived_terms_flat(mock_sim.terms, tolistonly=False)

    # TODO: how to update the terms for evaluation? how to make a selection of them after derivation?
    mock_sim.terms = [flat_dict['1_(1, 0)']]

    # FIXME: This printing appears to need update wrt. changes in wilson-derive, made issue
    #print('\n', flat_dict['1_(1, 0)'].to_latex())

    print(mock_sim.vib_ana_setup.max_state_lvl)

    mock_sim.evaluate()
    
    for f in mock_sim._workflow.artifacts.features:
        print(f.location, f.term_contributions[0].term_ids)
    
    region = mock_sim._workflow.artifacts.regions[0]
    feat1 = region.domain.full_features[0]
    feat_coeff = feat1.amplitude_coeff
    term_contributions = feat1.term_contributions

    print('\nterm_contributions[0].term_ids', term_contributions[0].term_ids, '\n')
    print('feat_coeff', feat_coeff)

    np.set_printoptions(linewidth=280, precision=1)
    for k,v in mock_sim._workflow.artifacts.grid_manager.full_grid.items():
        print(k,v)
    print('\n==========')
    
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    r_res = ws.intensities.amplitudes.evaluation_wf.evaluate_region(region, 
                                                            mock_sim._workflow.artifacts.vib_data, 
                                                            mock_sim._workflow.artifacts.vibdiff_cache, 
                                                            convNu2Ene(mock_sim.spec_eval_setup.ev_info.Gamma))
    ref_res = np.array([1/(-1j*convNu2Ene(1.))/(-1j*convNu2Ene(1.)) * feat_coeff])

    assert np.allclose(r_res, ref_res)
    assert np.allclose(ref_res, mock_sim.spec['result'])

def test_evaluation_general_customdata_1mechterm():
    print()
    from ..unit.test_domains import get_data_evaluators_tests
    datadict = get_data_evaluators_tests()
    # 'system', 'vib_ana_setup', 'derived_terms', 'props', 
    # 'experiment', 'spec_eval_setup', 'domain_distance_thresholds'
    
    np.set_printoptions(linewidth=180, precision=3)

    from wilson_suite.wilson_main.wf import WilsonSimulation
    # from wilson_suite.wilson_main.workflow_abstractions import WilsonSimulation

    from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
    from ....fixtures import evv_experiment
    from CQCParse.utils import PKG_ROOT as CQCPARSE_ROOT
    
    evv_exp = evv_experiment()
    terms = get_fully_enhanced_terms(experiment=evv_exp)
    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}

    bounds_dict = {'B': (900., 900.), 'A': (1864., 1864.)}
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 1., 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 1, 'B': 1}})
    mock_sim = WilsonSimulation()
    mock_sim.terms = terms

    mock_sim.exp = evv_exp
    mock_sim.setAxisChoiceAndTranslateTerms(axes_choice)


    mock_sim.spec_eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    # use simple model data
    mock_sim.system = datadict['system']
    mock_sim.props = datadict['props']
    mock_sim.vib_ana_setup = datadict['vib_ana_setup']
    mock_sim.vib_ana_setup.max_state_lvl = 3 # there is an issue for the underlying reason for this

    print('\nmock_sim.is_ready', mock_sim.is_ready)

    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_flat
    flat_dict = derived_terms_flat(mock_sim.terms, tolistonly=False)

    # TODO: how to update the terms for evaluation? how to make a selection of them after derivation?

    # FIXME: Cannot find this term
    #mock_sim.terms = [flat_dict['8_(0, 1)']]
    #print('\n', flat_dict['8_(0, 1)'].to_latex())

    mock_sim.evaluate()

    for f in mock_sim._workflow.artifacts.features:
        print(f.location, f.term_contributions[0].term_ids)

    region = mock_sim._workflow.artifacts.regions[0]
    feat1 = region.domain.full_features[0]
    feat_coeff = feat1.amplitude_coeff
    term_contributions = feat1.term_contributions
    
    print('\nterm_contributions[0].term_ids', term_contributions[0].term_ids, '\n')
    print('feat_coeff', feat_coeff)

    np.set_printoptions(linewidth=280, precision=1)
    for k,v in mock_sim._workflow.artifacts.grid_manager.full_grid.items():
        print(k,v)
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
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}

    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian', 
                                                     lvl_theory='B3LYP', 
                                                     basis_set='cc-pVQZ', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')

    from wilson_suite.wilson_main.wf import WilsonSimulation
    sim = WilsonSimulation()

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

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    
    bounds_dict = {'A': (0., 5000.), 'B': (0., 5000.)}

    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 4.7, 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 7, 'B': 10}})
    
    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    sim.addSpecEvalSetup(eval_setup)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)

    print('simulation.exp.polarization_avg_vector', sim.exp.polarization_avg_vector)
    sim.evaluate()

    np.set_printoptions(linewidth=280, precision=1)


    print(sim.spec['A'])
    print(sim.spec['B'])
    print(sim.spec['result'])

    import matplotlib.pyplot as plt

    Z = np.log(np.abs(sim.spec['result'])**2)
    x = np.unique(sim.spec['A'])
    y = np.unique(sim.spec['B'])

    # if Z.shape == (len(y), len(x)) -> no transpose; if Z.shape == (len(x), len(y)) -> transpose
    # matplotlib expects [y, x] ordering for images
    toplot = Z.T

    plt.pcolormesh(x, y, toplot, shading="auto")
    plt.xlabel('A')
    plt.ylabel('B')
    plt.colorbar(label='log intensity')
    #plt.show()


def test_full_integration_H2O_molecule():
    print()
    from ....fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[3] # {'A': [(2,)], 'B': [(-1,), (2,)]}

    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                     lvl_theory='HF', 
                                                     basis_set='STO-3G', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_h2o_HF_STO3G.out')

    from wilson_suite.wilson_main.wf import WilsonSimulation
    sim = WilsonSimulation()

    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms) # terms

    mol_system = ws.main.abstractions.MolecularSystem(name='h2o', natoms=3)

    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    sim.setAxisChoiceAndTranslateTerms(axes_choice)

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box
    
    bounds_dict = {'A': (0., 5000.), 'B': (0., 5000.)}

    spectral_window = SpectralWindow(box=Box(bounds_dict))

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 4.7, 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 10, 'B': 10}})
    
    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    sim.addSpecEvalSetup(eval_setup)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)

    sim.evaluate()

    np.set_printoptions(linewidth=280, precision=1)

    print(sim.spec['A'])
    print(sim.spec['B'])
    print(sim.spec['result'])

    import matplotlib.pyplot as plt

    Z = np.log(np.abs(sim.spec['result'])**2)
    x = np.unique(sim.spec['A'])
    y = np.unique(sim.spec['B'])

    # if Z.shape == (len(y), len(x)) -> no transpose; if Z.shape == (len(x), len(y)) -> transpose
    # matplotlib expects [y, x] ordering for images
    toplot = Z.T

    plt.pcolormesh(x, y, toplot, shading="auto")
    plt.xlabel('A')
    plt.ylabel('B')
    plt.colorbar(label='log intensity')
    #plt.show()
