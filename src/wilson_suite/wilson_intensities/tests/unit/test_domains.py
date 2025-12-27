from wilson_suite.wilson_intensities.amplitudes import domains
from ...amplitudes.spectrum_composition import RectangularDomain, SpectralWindow
# from ...amplitudes.evaluators import get_features_from_terms_for_eval
from ....wilson_main import abstractions as wm_abst
import numpy as np

def test_find_domain_groups_by_distance():

    points = [(1., 3.), (5., 11.), (4., 2.), (12., 6.), (8., 2.), (11., 4.)]

    groups1 = domains.find_points_clusters_by_distance(points, distance_thresholds={'A': 10., 'B': 10.}, linkage='single')
    groups2 = domains.find_points_clusters_by_distance(points, distance_thresholds={'A': 10., 'B': 10.}, linkage='ward')

    assert len(groups1) == 1
    assert len(groups2) == 3

    groups = domains.find_points_clusters_by_distance(points, distance_thresholds={'A': 12., 'B': 12.}, linkage='ward')
    assert len(groups) == 2

    groups1 = domains.find_points_clusters_by_distance(points, distance_thresholds={'A': 4., 'B': 4.}, linkage='single')
    groups2 = domains.find_points_clusters_by_distance(points, distance_thresholds={'A': 4., 'B': 4.}, linkage='ward')
    assert len(groups1) == 3
    assert len(groups2) == 4


def test_find_domain_distance_threshold():
    print()
    dists = domains.get_distance_threshold(16, {'A': 4., 'B': 4.})
    print(dists)

def test_terms():
    print()
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()


def generate_props_data_Nmodes(N_modes):
    return {'dipgrad': np.ones((N_modes, 3)), 
            'diphess': np.zeros((N_modes, N_modes, 3)),
            'polgrad': np.zeros((N_modes, 3, 3)), 
            'polhess': np.zeros((N_modes, N_modes, 3, 3)),
            'cff':     np.ones((N_modes, N_modes, N_modes)),
            'qff':     np.zeros((N_modes, N_modes, N_modes, N_modes)),
            }

def get_data_evaluators_tests() -> dict:
    from ....wilson_derive.response_terms import VibPerturbedTerm
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()

    t_inds = [0, 1,-1, -2, -3]
    terms_select: list['VibPerturbedTerm'] = [terms_fuller_flat[tID] for tID in t_inds]

    system = wm_abst.MolecularSystem(name='mock', natoms=3, linear=False)

    # vib_ana_setup needs to have vibstates
    vibana = wm_abst.VibAnaSetup(system=system)
    vibana.nc_sqrt_eigval
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
    vibana.nc_sqrt_eigval = {('0',): 964, ('1',): 1234., ('2',): 1234.}

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
    
    from ...amplitudes.spectrum_composition import Box
    from wilson_suite.wilson_main.spectrum_abstractions import SpecEvalSetup, EvaluationInfo
    from ....fixtures import evv_experiment

    spec_eval_setup = SpecEvalSetup(ev_info=EvaluationInfo(spectral_window=SpectralWindow(box=Box({'A': (1550., 2359.), 
                                                                                               'B': (800., 1300.)})),
                                                           Gamma={'A': 9.5, 'B': 16.5}))
    experiment = evv_experiment()
    return dict(system=system,
                vib_ana_setup=vibana, 
                derived_terms=terms_select, 
                props=props,
                experiment=experiment,
                spec_eval_setup=spec_eval_setup,
                domain_distance_thresholds={'A': 12., 'B': 12.})

# def get_features_from_terms():

#     datadict = get_data_evaluators_tests()

#     features = get_features_from_terms_for_eval(**datadict)
#     return features


def test_terms_features():
    print()
    
    features = get_features_from_terms()
    features_locs = [loc_geo_obj.values for loc_geo_obj in features]
    print('\nfeatures', list(features.values()))
    print('\nfeatures_locs', features_locs)
    spec_features = [i[1] for i in list(features.values())]
    print('\nspec_features', set(spec_features))
    # from ...amplitudes.term_parts import SpectralFeature, GeometricObject, TermParametersChoice, ParameterSet

    assert set(features_locs) == set([(1864.0, 900.0), (2255.0, 1291.0), (2274.0, 1310.0), 
                                      (2255.0, 1021.0), (2368.0, 1134.0), (2360.0, 1126.0), 
                                      (2274.0, 1040.0), (2362.0, 1128.0), (964.0, 0.0), 
                                      (1234.0, 270.0), (964.0, -270.0), (1234.0, 0.0)])


def test_find_clusters_by_distance():
    print()
    features = get_features_from_terms()

    features_locs = [loc_geo_obj.values for loc_geo_obj in features]
    print('\nfeatures', features)
    print('\nfeatures_locs', features_locs)
    # exit()

    clusters = domains.find_points_clusters_by_distance(res_locations=features_locs, 
                                                 distance_thresholds={'A': 10., 'B': 10.}, 
                                                 linkage='single')
    print('\nclusters', clusters)
    assert sorted(clusters.values()) == sorted({5: [(1864.0, 900.0)], 8: [(2255.0, 1291.0)], 
                                                7: [(2274.0, 1310.0)], 9: [(2255.0, 1021.0)], 
                                                0: [(2368.0, 1134.0), (2360.0, 1126.0), (2362.0, 1128.0)], 
                                                6: [(2274.0, 1040.0)], 4: [(964.0, 0.0)], 
                                                2: [(1234.0, 270.0)], 3: [(964.0, -270.0)], 1: [(1234.0, 0.0)]}.values())

    clusters = domains.find_points_clusters_by_distance(res_locations=features_locs, 
                                                 distance_thresholds={'A': 35., 'B': 35.}, 
                                                 linkage='single')
    assert sorted(clusters.values()) == sorted({5: [(964.0, 0.0)], 7: [(1234.0, 270.0)], 
                        4: [(964.0, -270.0)], 3: [(1234.0, 0.0)], 
                        6: [(1864.0, 900.0)], 0: [(2255.0, 1291.0), (2274.0, 1310.0)], 
                        1: [(2255.0, 1021.0), (2274.0, 1040.0)], 
                        2: [(2368.0, 1134.0), (2360.0, 1126.0), (2362.0, 1128.0)]}.values())


def test_find_feature_clusters_by_distance():
    print()
    features = get_features_from_terms()

    # print('\nfeatures', features)

    rec_windows_dict = domains.find_feature_clusters_by_distance(features=features, 
                                                         distance_thresholds={'A': 10., 'B': 10.}, 
                                                         linkage='single')
    # print('\nrec_windows_dict', rec_windows_dict)
    for window in rec_windows_dict:
        # print(rec_windows_dict[window].bounds)
        print('\n', rec_windows_dict[window])


def test_compute_box_adjacency():
    print()
    features = get_features_from_terms()
    # print('features', features)
    for f in features:
        print(f.lineshape_parameter)
    points_from_features = [feat.location._coord_dict for feat in features]
    halfwidths_list_from_features = [feat.lineshape_parameter for feat in features]
    
    rectangular_boxes = domains.points_to_bounds(points_from_features, halfwidths_list_from_features)
    res = domains.compute_box_adjacency(rectangular_boxes)
    print(res)


def test_features_to_clusters():
    print()
    features = get_features_from_terms()
    rr = domains.features_to_clusters(features=features)

    for k in rr:
        print(k, '----\n', rr[k], '\n')

def test_feat_clusters_to_domains():
    features = get_features_from_terms()
    feat_clusters = domains.features_to_clusters(features=features)
    # print('\nfeat_clusters', feat_clusters)

    # from ...amplitudes.spectrum_composition import SpectralFeature

    clusters = []
    for fc in feat_clusters:
        # print('\nfeat_clusters[fc]', feat_clusters[fc])
        clusters.append(RectangularDomain.from_features(feat_clusters[fc]))
        # clusters.append(RectangularDomain())
    
    print('\nclusters', len(clusters))
    print('\nclusters', clusters)
    return
