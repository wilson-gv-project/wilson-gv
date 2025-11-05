from wilson_suite.wilson_intensities.amplitudes import domains


def test_find_domain_groups_by_distance():
    print()

    points = [(1., 3.), (5., 11.), (4., 2.), (12., 6.), (8., 2.), (11., 4.)]

    groups1 = domains.find_clusters_by_distance(points, distance_threshold=[10., 10.], linkage='single')
    groups2 = domains.find_clusters_by_distance(points, distance_threshold=[10., 10.], linkage='ward')

    assert len(groups1) == 1
    assert len(groups2) == 3

    groups = domains.find_clusters_by_distance(points, distance_threshold=[12., 12.], linkage='ward')
    assert len(groups) == 2

    groups1 = domains.find_clusters_by_distance(points, distance_threshold=[4., 4.], linkage='single')
    groups2 = domains.find_clusters_by_distance(points, distance_threshold=[4., 4.], linkage='ward')
    assert len(groups1) == 3
    assert len(groups2) == 4
    print(groups2)


def test_find_domain_distance_threshold():
    print()
    domains.find_distance_threshold(16, {'A': 4., 'B': 4.})

def test_terms():
    print()
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()


from ...amplitudes.evaluators import terms_evaluator_general_compilation
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

def test_terms_evaluator_general_compilation():
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

    features = terms_evaluator_general_compilation(system=system, 
                                        vib_ana_setup=vibana, 
                                        derived_terms=terms_select, 
                                        props=props)
    features_locs = [tuple(ax[1] for ax in loc_tuple) for loc_tuple in features]
    print('\features', features)
    # print('\nfeatures_locs', features_locs)
    # exit()

    clusters = domains.find_clusters_by_distance(res_locations=features_locs, 
                                                 distance_threshold=[10., 10.], 
                                                 linkage='single')
    assert sorted(clusters.values()) == sorted({5: [(1864.0, 900.0)], 8: [(2255.0, 1291.0)], 
                                                7: [(2274.0, 1310.0)], 9: [(2255.0, 1021.0)], 
                                                0: [(2368.0, 1134.0), (2360.0, 1126.0), (2362.0, 1128.0)], 
                                                6: [(2274.0, 1040.0)], 4: [(964.0, 0.0)], 
                                                2: [(1234.0, 270.0)], 3: [(964.0, -270.0)], 1: [(1234.0, 0.0)]}.values())

    clusters = domains.find_clusters_by_distance(res_locations=features_locs, 
                                                 distance_threshold=[35., 35.], 
                                                 linkage='single')
    assert sorted(clusters.values()) == sorted({5: [(964.0, 0.0)], 7: [(1234.0, 270.0)], 
                        4: [(964.0, -270.0)], 3: [(1234.0, 0.0)], 
                        6: [(1864.0, 900.0)], 0: [(2255.0, 1291.0), (2274.0, 1310.0)], 
                        1: [(2255.0, 1021.0), (2274.0, 1040.0)], 
                        2: [(2368.0, 1134.0), (2360.0, 1126.0), (2362.0, 1128.0)]}.values())
    
    locs_by_domains = domains.determine_domains_and_features(features_to_draw=features, 
                                                             dynamic_range=100, 
                                                             Gamma_axes={'A': 10., 'B': 15.})

    assert sorted(locs_by_domains) == sorted([[(964.0, 0.0)], [(1234.0, 270.0)], 
                                              [(964.0, -270.0)], [(1234.0, 0.0)], 
                                              [(1864.0, 900.0)], [(2255.0, 1291.0)], 
                                              [(2274.0, 1310.0)], [(2255.0, 1021.0)], 
                                              [(2368.0, 1134.0), (2360.0, 1126.0), (2362.0, 1128.0)], 
                                              [(2274.0, 1040.0)]])
    
    locs_by_domains = domains.determine_domains_and_features(features_to_draw=features, 
                                                             dynamic_range=500, 
                                                             Gamma_axes={'A': 50., 'B': 75.})
    assert sorted(locs_by_domains) == sorted([[(1864.0, 900.0)], [(2255.0, 1291.0), (2274.0, 1310.0)], 
                                              [(2255.0, 1021.0), (2274.0, 1040.0)], 
                                              [(2368.0, 1134.0), (2360.0, 1126.0), (2362.0, 1128.0)], 
                                              [(964.0, 0.0)], [(1234.0, 270.0)], 
                                              [(964.0, -270.0)], [(1234.0, 0.0)]])
    
    dimension_groups = domains.unzip_tuples([(2368.0, 1134.0), (2360.0, 1126.0), (2362.0, 1128.0)])
    print('\ndimension_groups', dimension_groups)

    print([tuple([min(g), max(g)]) for g in dimension_groups])

