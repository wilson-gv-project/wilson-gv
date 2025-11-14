import wilson_suite.wilson_intensities.amplitudes.evaluators as evaluators #terms_evaluator_general, terms_evaluator_general_compilation
from wilson_suite.wilson_intensities.amplitudes import domains as domfuncs
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

def test_evaluate_feature_on_grid():
    print()
    from .test_domains import get_features_from_terms, get_data_evaluators_tests
    datadict = get_data_evaluators_tests()
    
    features = get_features_from_terms()
    print(features)

    spec_window = datadict['spec_eval_setup'].ev_info.spectral_window

    print('\nspec_window', spec_window)
    from ...amplitudes.spectrum_composition import SpectralFeature, Box, RectangularDomain
    spec_window_with_features = SpectralFeature.filter_to_spec_window(features, spec_window)
    print('\nspec_window_with_features', spec_window_with_features)

    feat_all = spec_window_with_features.full_features + spec_window_with_features.contrib_features
    domains = domfuncs.features_to_clusters(features=feat_all)
    # print('\ndomains', domains)
    domains_in_window = [RectangularDomain(box=Box.union([f.feat_box for f in domains[d]]), full_features=domains[d]) for d in domains]
    
    # print('\ndomains_in_window', domains_in_window)
    
    domain3 = domains_in_window[3]
    d3_all_feats = domain3.full_features + domain3.contrib_features
    print(d3_all_feats)

    spec_grid = spec_window_with_features.sample_grid({'A': 10, 'B': 10})
    subgrids = domfuncs.cut_grid_with_indices_dict_nd(spec_grid, domains_in_window)
    
    print('\nsubgrids', subgrids[domain3])

    terms_hashes = {t.h(): t for t in datadict['derived_terms']}
    evaluators.evaluate_feature_on_grid(feature=d3_all_feats[0], meshgrids=subgrids[domain3]['grid'], terms_hashes=terms_hashes)

