import wilson_suite.wilson_intensities.amplitudes.evaluators as evaluators #terms_evaluator_general, terms_evaluator_general_compilation
from wilson_suite.wilson_intensities.amplitudes import domains as domfuncs
from ...amplitudes.numerical_abstractions import NumericalResonanceCondition, NumericalResonanceMotif
from ....wilson_main import abstractions as wm_abst
import numpy as np

def generate_props_data_Nmodes(N_modes):
    return {'dipgrad': np.ones((N_modes, 3)), 
            'diphess': np.zeros((N_modes, N_modes, 3)),
            'polgrad': np.zeros((N_modes, 3, 3)), 
            'polhess': np.zeros((N_modes, N_modes, 3, 3)),
            'cff':     np.ones((N_modes, N_modes, N_modes)),
            'qff':     np.zeros((N_modes, N_modes, N_modes, N_modes)),
            }

def test_terms_evaluator_general():
    print()
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()

    t_inds = [0, 1,-1, -2, -3]
    terms_select: list['wm_abst.VibPerturbedTerm'] = [terms_fuller_flat[tID] for tID in t_inds]

    # vib_ana_setup needs to have vibstates
    vibana = wm_abst.VibAnaSetup()
    vibana.setStates((
                        wm_abst.VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
                        wm_abst.VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
                        wm_abst.VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=1234., harmonic_WF=True),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 0):1.}, state_label='0,0', energy=1864., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 1):1.}, state_label='0,1', energy=2255., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 1):1.}, state_label='1,1', energy=2368., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 2):1.}, state_label='1,2', energy=2360., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(2, 2):1.}, state_label='2,2', energy=2362., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 2):1.}, state_label='0,2', energy=2274., harmonic_WF=False),
                        # Adding 3-quanta states:
                        wm_abst.VibState(harm_quanta_coeffs={(0, 0, 0):1.}, state_label='0,0,0', energy=2685., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 1, 1):1.}, state_label='1,1,1', energy=3581., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(2, 2, 2):1.}, state_label='2,2,2', energy=3690., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 1, 2):1.}, state_label='0,1,2', energy=3742., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 2, 2):1.}, state_label='0,2,2', energy=3680., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 0, 1):1.}, state_label='0,0,1', energy=3155., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 0, 2):1.}, state_label='0,0,2', energy=3498., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 1, 2):1.}, state_label='1,1,2', energy=3594., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 2, 2):1.}, state_label='1,2,2', energy=3642., harmonic_WF=False),
                    ))
    system = wm_abst.MolecularSystem(name='mock', natoms=3, linear=False)

    props_data = generate_props_data_Nmodes(system.Nnmodes)
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['dipgrad'][1, 0] = 0.45
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polgrad'][1, 1, 0] = 0.3
    props_data['polgrad'][0, 1, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15
    
    props_data['cff'][0, 1, 1] = 0.7
    
    props = []
    
    # props needs to have properties tensors, values    
    for trname in props_data:
        p = wm_abst.MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)

    print(system.Nnmodes)
    
    evaluators.terms_evaluator_general(system=system, 
                            vib_ana_setup=vibana, 
                            derived_terms=terms_select, 
                            props=props)


def test_terms_evaluator_general_degen_states():
    print()
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()

    t_inds = [0, 1,-1, -2, -3]
    terms_select: list['wm_abst.VibPerturbedTerm'] = [terms_fuller_flat[tID] for tID in t_inds]

    # vib_ana_setup needs to have vibstates
    vibana = wm_abst.VibAnaSetup()
    vibana.setStates((
    wm_abst.VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
    wm_abst.VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
    wm_abst.VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=1234., harmonic_WF=True),
    wm_abst.VibState(harm_quanta_coeffs={(0, 0):1.}, state_label='0,0', energy=1864., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 1):1.}, state_label='0,1', energy=2255., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 1):1.}, state_label='1,1', energy=2368., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 2):1.}, state_label='1,2', energy=2360., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(2, 2):1.}, state_label='2,2', energy=2362., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 2):1.}, state_label='0,2', energy=2274., harmonic_WF=False),
    # Adding 3-quanta states:
    wm_abst.VibState(harm_quanta_coeffs={(0, 0, 0):1.}, state_label='0,0,0', energy=2685., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 1, 1):1.}, state_label='1,1,1', energy=3581., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(2, 2, 2):1.}, state_label='2,2,2', energy=3690., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 1, 2):1.}, state_label='0,1,2', energy=3742., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 2, 2):1.}, state_label='0,2,2', energy=3680., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 0, 1):1.}, state_label='0,0,1', energy=3155., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 0, 2):1.}, state_label='0,0,2', energy=3498., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 1, 2):1.}, state_label='1,1,2', energy=3594., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 2, 2):1.}, state_label='1,2,2', energy=3642., harmonic_WF=False),
))
    system = wm_abst.MolecularSystem(name='mock', natoms=3, linear=False)

    props_data = generate_props_data_Nmodes(system.Nnmodes)
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['dipgrad'][1, 0] = 0.45
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polgrad'][1, 1, 0] = 0.3
    props_data['polgrad'][0, 1, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15
    
    props_data['cff'][0, 1, 1] = 0.7
    
    props = []
    
    # props needs to have properties tensors, values    
    for trname in props_data:
        p = wm_abst.MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)

    print(system.Nnmodes)
    
    evaluators.terms_evaluator_general(system=system, 
                            vib_ana_setup=vibana, 
                            derived_terms=terms_select, 
                            props=props)

def test_evaluate_domain_on_grid():
    print()
    from .test_domains import get_features_from_terms_for_eval
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()
    t_inds = [0, 1,-1, -2, -3]
    # terms_select: list['wm_abst.VibPerturbedTerm'] = [terms_fuller_flat[tID] for tID in t_inds]
    terms_select: list['wm_abst.VibPerturbedTerm'] = terms_fuller_flat

    vibana = wm_abst.VibAnaSetup()
    vibana.setStates((
    wm_abst.VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
    wm_abst.VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
    wm_abst.VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=1234., harmonic_WF=True),
    wm_abst.VibState(harm_quanta_coeffs={(0, 0):1.}, state_label='0,0', energy=1864., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 1):1.}, state_label='0,1', energy=2255., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 1):1.}, state_label='1,1', energy=2368., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 2):1.}, state_label='1,2', energy=2360., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(2, 2):1.}, state_label='2,2', energy=2362., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 2):1.}, state_label='0,2', energy=2274., harmonic_WF=False),
    # Adding 3-quanta states:
    wm_abst.VibState(harm_quanta_coeffs={(0, 0, 0):1.}, state_label='0,0,0', energy=2685., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 1, 1):1.}, state_label='1,1,1', energy=3581., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(2, 2, 2):1.}, state_label='2,2,2', energy=3690., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 1, 2):1.}, state_label='0,1,2', energy=3742., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 2, 2):1.}, state_label='0,2,2', energy=3680., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 0, 1):1.}, state_label='0,0,1', energy=3155., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(0, 0, 2):1.}, state_label='0,0,2', energy=3498., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 1, 2):1.}, state_label='1,1,2', energy=3594., harmonic_WF=False),
    wm_abst.VibState(harm_quanta_coeffs={(1, 2, 2):1.}, state_label='1,2,2', energy=3642., harmonic_WF=False),
))
    system = wm_abst.MolecularSystem(name='mock', natoms=3, linear=False)

    props_data = generate_props_data_Nmodes(system.Nnmodes)
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['dipgrad'][1, 0] = 0.45
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polgrad'][1, 1, 0] = 0.3
    props_data['polgrad'][0, 1, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15    
    props_data['cff'][0, 1, 1] = 0.7
    props = []
    for trname in props_data:
        p = wm_abst.MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)
    
    features = get_features_from_terms_for_eval(system=system, 
                                                vib_ana_setup=vibana, 
                                                derived_terms=terms_select, 
                                                props=props)
    print('features len', len(features))
    rec_domains_dict = domfuncs.find_feature_clusters_by_distance(features=features, 
                                                         distance_thresholds={'A': 10., 'B': 10.}, 
                                                         linkage='single')
    count = 0
    # print(rec_domains_dict)
    print(features)
    for i in rec_domains_dict:
        print(i, len(rec_domains_dict[i].features))
        count += len(rec_domains_dict[i].features)
    print('count', count)
    print('features len', len(features))
    # print(rec_domains_dict[4])

    evaluators.evaluate_domain_on_grid(domain=rec_domains_dict[4])

def test_terms_evaluator_general_compilation():
    print()
    from .test_domains import get_data_evaluators_tests
    datadict = get_data_evaluators_tests()
    print('datadict', list(datadict.keys()))

    evaluators.terms_evaluator_general_compilation(**datadict)

from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm

def get_necessary_data(terms_select: list[VibPerturbedTerm]):
    import wilson_suite.wilson_intensities.amplitudes.full_amplitude_coeff as fac
    from wilson_suite.wilson_main.abstractions import MolPropsCollection, MolecularProperty, VibState
    from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData, EvaluationDataAndConfigs
    from .test_full_coeff import generate_props_data4modes
    props_data = generate_props_data4modes()
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['dipgrad'][1, 0] = 0.45
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polgrad'][1, 1, 0] = 0.3
    props_data['polgrad'][0, 1, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15
    
    props_data['cff'][0, 1, 1] = 0.7

    props = []
    for trname in props_data:
        p = MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)

    vibdata = VibStatesData(allstates=(VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=3644., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
                                       ),
                                   harmonic_osc_states_labels=(0, 1, 2))

    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)

    from .test_domains import get_data_evaluators_tests
    datadict = get_data_evaluators_tests()
    print('\ndatadict', datadict.keys())

    # settings = EvaluationDataAndConfigs(props_data=MolPropsCollection(props),
    #                                     vibstates_data=vibdata,
    #                                     number_of_nmodes=4)
    data_and_configs = EvaluationDataAndConfigs(props_data=props,
                                                vibstates_data=vibdata,
                                                number_of_nmodes=4,
                                                nm_inds_choices=[0,1],
                                                pulse_polarization_vector=datadict['experiment'].polarization_avg_vector)


    print('\nbefore error - data_and_configs', data_and_configs)
    results = fac.precalculate_unique_coeff_parts(need_to_precalc=need_to_precalc,
                                                  data_and_configs=data_and_configs)
    return results

def test_evaluate_feature_on_grid():
    print()
    from .test_domains import get_features_from_terms, get_data_evaluators_tests
    datadict = get_data_evaluators_tests()
    
    features = get_features_from_terms()
    # print(features)

    spec_window = datadict['spec_eval_setup'].ev_info.spectral_window

    print('\nspec_window', spec_window)
    from ...amplitudes.spectrum_composition import SpectralFeature, Box, RectangularDomain
    spec_window_with_features = SpectralFeature.filter_to_spec_window(features, spec_window)
    # print('\nspec_window_with_features', spec_window_with_features)

    feat_all = spec_window_with_features.full_features + spec_window_with_features.contrib_features
    domains = domfuncs.features_to_clusters(features=feat_all)
    # print('\ndomains', domains)
    domains_in_window = [RectangularDomain(box=Box.union([f.feat_box for f in domains[d]]), full_features=domains[d]) for d in domains]
    
    # print('\ndomains_in_window', domains_in_window)
    
    domain3 = domains_in_window[3]
    d3_all_feats = domain3.full_features + domain3.contrib_features
    # print(d3_all_feats)

    spec_grid = spec_window_with_features.sample_grid({'A': 10, 'B': 10})
    subgrids = domfuncs.cut_grid_with_indices_dict_nd(spec_grid, domains_in_window)
    
    print('\nsubgrids', subgrids[domain3])

    # necessary_data = get_necessary_data(list(terms_hashes.values()))
    print('\nlist(datadict.keys())', list(datadict.keys()))
    from ...amplitudes.term_parts import EvaluationDataAndConfigs, VibStatesData
    from ...amplitudes.vibene_differences import VibDiffCache
    from wilson_suite.wilson_main.abstractions import VibAnaSetup
    vas: VibAnaSetup = datadict['vib_ana_setup']
    dd = EvaluationDataAndConfigs(vibstates_data=VibStatesData(allstates=vas.states, harmonic_osc_states_labels=(0,1)),
                                  vibdiff_cache=VibDiffCache())
    
    final = evaluators.evaluate_feature_on_grid(feature=d3_all_feats[0], 
                                                meshgrids=subgrids[domain3]['grid'], 
                                                necessary_data=dd)
    print(final)


def test_evaluate_feature_on_grid_simple():
    print()
    from ...amplitudes.spectrum_composition import SpectralFeature, SpectralWindow, ResLocGeoObject, Box
    from ...amplitudes.term_parts import EvaluationDataAndConfigs, VibStatesData, VibState, TermParametersChoice, ParameterSet, ResonanceCondition, VibDiffTerm, HarmOscStateSymbolic
    from fractions import Fraction
    
    res_for_dummyterm = [ResonanceCondition(diff=VibDiffTerm(sl=HarmOscStateSymbolic(q=['a']), sr=HarmOscStateSymbolic(q=['a', 'b'])),
                                            pf=['A'])]
    dummy_term = VibPerturbedTerm(coeff=Fraction(1,1), props=[], freqterms=[], res=res_for_dummyterm)

    print(dummy_term)

    feature1 = SpectralFeature(location=ResLocGeoObject({'A': 1100., 'B': 2300.}), 
                               term_contributions=(TermParametersChoice(terms=(dummy_term,),
                                                                        states_parameters=(ParameterSet({'a': 0, 'b': 0}),)),),
                               lineshape_parameter={'A': 3., 'B': 3.}, 
                               lineshape_parameter_single=3.,
                               amplitude_coeff=0.5)

    subgrid_feat1 = SpectralWindow(box=Box({'A': (950., 1200.), 'B': (2000., 2450.)})).sample_grid({'A': 5, 'B': 5})

    from ...amplitudes.vibene_differences import VibDiffCache
    dd = EvaluationDataAndConfigs(vibstates_data=VibStatesData(allstates=(VibState(harm_quanta_coeffs={(0,): 1.}, 
                                                                                   energy=1000., state_label='0'),
                                                                          VibState(harm_quanta_coeffs={(1,): 1.}, 
                                                                                   energy=2000., state_label='1'),
                                                                          VibState(harm_quanta_coeffs={(0,0,): 1.}, 
                                                                                   energy=2100., state_label='0,0'),), 
                                                               harmonic_osc_states_labels=(0,1)),
                                  vibdiff_cache=VibDiffCache())

    print('\nsubgrid_feat1\n', subgrid_feat1)
    final = evaluators.evaluate_feature_on_grid(feature=feature1, 
                                                meshgrids=subgrid_feat1, 
                                                necessary_data=dd)

    print(final)
    print(final.shape)

from ...amplitudes.term_parts import EvalTerm, EvalFeature, make_resonance_function, ResonanceMotif, ResonanceCondition

def test_evaluate_simple_term():
    # lightweight directly
    # res_for_dummyterm = [ResonanceCondition(diff=VibDiffTerm(sl=HarmOscStateSymbolic(q=['a']), sr=HarmOscStateSymbolic(q=['a', 'b'])),
    #                                         pf=['A'])]
    resonance_function = make_resonance_function(res_motif=ResonanceMotif())
    term = EvalTerm(
        prefactor=1.0,
        resonance_function=lambda grid: grid['A']*0 + grid['B']*0 + 1,
        parameters={}
    )

    feature = EvalFeature(
        location={'A':1100, 'B':2300},
        terms=[term],
        amplitude=0.5,
        lineshape_param=3.
    )
    A_mesh, B_mesh = np.meshgrid(np.linspace(950,1200,5), np.linspace(2000,2450,5), indexing='ij')
    grid = {'A': A_mesh,
            'B': B_mesh}
    
    result = evaluators.evaluate_eval_feature_on_grid(feature, grid)
    print('\n', result)
    # assert result.shape == (5,5)

from ...amplitudes.term_parts import make_resonance_function, ResonanceMotif, ResonanceCondition

def test_evaluate_resonance_simple():
    # 1. Make a compiled motif with one term
    motif = NumericalResonanceMotif(res_conds=[
        NumericalResonanceCondition(
            pf_dict={"A": 1, "B": -1},
            vib_energy_diff=10.0,
        )
    ])

    # 2. Make a simple 1D grid
    A = np.array([0.0, 1.0, 2.0])
    B = np.array([3.0, 1.0, 2.0])
    mesh = {"A": A, "B": B}

    # 3. Evaluate with some gamma
    gamma = 0.5
    result = evaluators.evaluate_resonance_on_grid(motif, mesh, gamma)

    # 4. Expected output (do by hand)
    # z_i = E_vib - (A[i] * pfactor) - i * gamma
    expected = 1 / (10 - A + B - 1j*gamma)

    # 5. Assert correctness
    np.testing.assert_allclose(result, expected)