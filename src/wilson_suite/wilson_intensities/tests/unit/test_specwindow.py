from ...amplitudes.spectrum_composition import Box, ResLocGeoObject, SpectralWindow, RectangularDomain, Grid
from ...amplitudes.spectrum_composition import SpectralFeature
from dataclasses import dataclass
import numpy as np
import pytest

def test_spectral_window():
    print()

    sw1d = SpectralWindow(bounds=((5., 10.),))

    res_loc1d_a = ResLocGeoObject({'A': 12.})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter={'A': 3.5})
    res_loc1d_b = ResLocGeoObject({'A': 7.5})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter={'A': 3.5})
    res_loc1d_c = ResLocGeoObject({'A': 9.5})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter={'A': 3.5})



def test_filter_to_spec_window():
    print()

    sw1d = SpectralWindow(box=Box({'A': (5., 10.)}))

    res_loc1d_a = ResLocGeoObject({'A': 12.})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter={'A': 1.5})
    res_loc1d_b = ResLocGeoObject({'A': 7.5})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter={'A': 1.5})
    res_loc1d_c = ResLocGeoObject({'A': 3.5})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter={'A': 1.5})

    sp = SpectralFeature.filter_to_spec_window([sf1, sf2, sf3], sw1d)
    print(sp)

    print('\nfull_features', sp.full_features)
    print('\ncontrib_features', sp.contrib_features)

    q = SpectralFeature.find_clusters_by_distance(spec_features=sp.full_features, 
                                                  distance_thresholds={'A': 6.},
                                                  linkage='ward')    
    print('\n', q)
    print(len(q))


def test_filter_to_spec_window_2():
    print()
    sw1d_a = SpectralWindow(box=Box({'A': (5., 30.)}))

    res_loc1d_d = ResLocGeoObject({'A': 15.})
    sf1 = SpectralFeature(location=res_loc1d_d, lineshape_parameter={'A': 2.5}, term_contributions=())
    res_loc1d_e = ResLocGeoObject({'A': 27.5})
    sf2 = SpectralFeature(location=res_loc1d_e, lineshape_parameter={'A': 2.5}, term_contributions=())
    res_loc1d_f = ResLocGeoObject({'A': 5.5})
    sf3 = SpectralFeature(location=res_loc1d_f, lineshape_parameter={'A': 2.5}, term_contributions=())
    res_loc1d_g = ResLocGeoObject({'A': 4.})
    sf4 = SpectralFeature(location=res_loc1d_g, lineshape_parameter={'A': 2.5}, term_contributions=())
    res_loc1d_h = ResLocGeoObject({'A': 36.})
    sf5 = SpectralFeature(location=res_loc1d_h, lineshape_parameter={'A': 2.5}, term_contributions=())

    spec_window1 = SpectralFeature.filter_to_spec_window([sf1, sf2, sf3, sf4, sf5], sw1d_a)
    print('\n', spec_window1)

    print('\nfull_features', len(spec_window1.full_features), spec_window1.full_features)
    print('\ncontrib_features', len(spec_window1.contrib_features), spec_window1.contrib_features)
    
    from wilson_suite.wilson_intensities.amplitudes import domains
    feat_all = spec_window1.full_features + spec_window1.contrib_features
    doms = domains.features_to_clusters(features=feat_all)
    print('\n\ndoms\n', doms)
    for d in doms:
        print('\n', d)
        print(doms[d])
    
    formal_doms = [RectangularDomain(box=Box.union([f.feat_box for f in doms[d]]), full_features=doms[d]) for d in doms]

    # print('\n')
    # for d in formal_doms:
    #     print(d, '\n')
    print('\n(spec_window1.box)', spec_window1.box)

    for rd in formal_doms:
        rd.box = rd.box.intersect(spec_window1.box)

    print('\n')
    for d in formal_doms:
        print(d, '\n')

    coords_vectors, gr = spec_window1.sample_grid({'A': 100})
    # print(gr.T)
    # gr = spec_window1.sample_grid({'A': 40})
    # print(gr.T)
    # gr = spec_window1.sample_grid({'A': 20})
    # print(gr.T)
    # Grid()
    print('\n(spec_window1)', spec_window1)
    


def cut_grid_with_indices_dict(grid: dict[str, np.ndarray], domains: list):
    """
    Cut subgrids from a structured 2D meshgrid so that both bounds are covered.
    Returns a dict mapping each domain.box -> {'A', 'B', 'indices'}.
    """
    A = grid['A']
    B = grid['B']
    a_coords = np.unique(A[:, 0])
    b_coords = np.unique(B[0, :])

    subgrids = {}

    for domain in domains:
        box = domain.box
        bounds = box.bounds
        a_min, a_max = bounds['A']
        b_min, b_max = bounds['B']

        # Find index ranges covering the bounds
        i_min = np.searchsorted(a_coords, a_min, side="right") - 1
        i_max = np.searchsorted(a_coords, a_max, side="left")
        j_min = np.searchsorted(b_coords, b_min, side="right") - 1
        j_max = np.searchsorted(b_coords, b_max, side="left")

        # Clamp to grid limits
        i_min = max(i_min, 0)
        j_min = max(j_min, 0)
        i_max = min(i_max, len(a_coords) - 1)
        j_max = min(j_max, len(b_coords) - 1)

        # Extract subgrids
        sub_A = A[i_min:i_max + 1, j_min:j_max + 1]
        sub_B = B[i_min:i_max + 1, j_min:j_max + 1]

        # Store in dict
        subgrids[box] = {
            "A": sub_A,
            "B": sub_B,
            "indices": (slice(i_min, i_max + 1), slice(j_min, j_max + 1))
        }

    return subgrids


from wilson_suite.wilson_intensities.amplitudes import domains

def test_filter_to_spec_window_2_2d():
    print()
    sw1d_a = SpectralWindow(box=Box({'A': (5., 30.), 'B': (45., 60.)}))

    res_loc1d_d = ResLocGeoObject({'A': 15., 'B': 55.})
    sf1 = SpectralFeature(location=res_loc1d_d, lineshape_parameter={'A': 2.5, 'B': 1.5}, term_contributions=(), amplitude_coeff=10.)
    res_loc1d_e = ResLocGeoObject({'A': 27.5, 'B': 58.8})
    sf2 = SpectralFeature(location=res_loc1d_e, lineshape_parameter={'A': 2.5, 'B': 1.5}, term_contributions=(), amplitude_coeff=20.)
    res_loc1d_f = ResLocGeoObject({'A': 5.5, 'B': 47.2})
    sf3 = SpectralFeature(location=res_loc1d_f, lineshape_parameter={'A': 2.5, 'B': 1.5}, term_contributions=(), amplitude_coeff=30.)
    res_loc1d_g = ResLocGeoObject({'A': 4., 'B': 43.8})
    sf4 = SpectralFeature(location=res_loc1d_g, lineshape_parameter={'A': 2.5, 'B': 1.5}, term_contributions=(), amplitude_coeff=40.)
    res_loc1d_h = ResLocGeoObject({'A': 36., 'B': 46.2})
    sf5 = SpectralFeature(location=res_loc1d_h, lineshape_parameter={'A': 2.5, 'B': 1.5}, term_contributions=(), amplitude_coeff=50.)

    spec_window1 = SpectralFeature.filter_to_spec_window([sf1, sf2, sf3, sf4, sf5], sw1d_a)
    print('\nspec_window1', spec_window1)
    exit()

    # print('\nfull_features', len(spec_window1.full_features), spec_window1.full_features)
    # print('\ncontrib_features', len(spec_window1.contrib_features), spec_window1.contrib_features)
    
    feat_all = spec_window1.full_features + spec_window1.contrib_features
    doms = domains.features_to_clusters(features=feat_all)

    print('\nfeat_all', len(feat_all), feat_all)
    print('\n')
    print('Number of domains formed:', len(doms))
    print('\n')

    # print('\n\ndoms\n', doms)
    # for d in doms:
    #     print('\n', d)
    #     print(doms[d])
    
    formal_doms = [RectangularDomain(box=Box.union([f.feat_box for f in doms[d]]), full_features=doms[d]) for d in doms]
    print('\nformal_doms', len(formal_doms), formal_doms)

    # print('\n(spec_window1.box)', spec_window1.box)

    for rd in formal_doms:
        rd.box = rd.box.intersect(spec_window1.box)

    # print('\n')
    # for d in formal_doms:
    #     print(d, '\n')


    

def test_evaluate_all_on_grids():
    spec_window1 = SpectralWindow(box=Box({'A': (5., 30.), 'B': (45., 60.)}))

    res_loc1d_d = ResLocGeoObject({'A': 15., 'B': 55.})
    sf1 = SpectralFeature(location=res_loc1d_d, lineshape_parameter={'A': 2.5, 'B': 1.5}, term_contributions=(), amplitude_coeff=10.)
    res_loc1d_g = ResLocGeoObject({'A': 4., 'B': 43.8})
    sf4 = SpectralFeature(location=res_loc1d_g, lineshape_parameter={'A': 2.5, 'B': 1.5}, term_contributions=(), amplitude_coeff=40.)
    spec_window1 = SpectralFeature.filter_to_spec_window([sf1, sf4], spec_window1)
    
    feat_all = spec_window1.full_features + spec_window1.contrib_features
    doms = domains.features_to_clusters(features=feat_all)
    formal_doms = [RectangularDomain(box=Box.union([f.feat_box for f in doms[d]]), full_features=doms[d]) for d in doms]

    coords_vectors, spec_grid = spec_window1.sample_grid({'A': 10, 'B': 10})
    subgrids = domains.cut_grid_with_coords_nd(spec_grid, coords_vectors, formal_doms)

    from wilson_suite.wilson_intensities.amplitudes.term_parts import VibStatesData
    from wilson_suite.wilson_intensities.amplitudes import evaluators
    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    from .test_evaluators import prep_vibanasetup_with_degen_states
    vib_ana_setup = prep_vibanasetup_with_degen_states()
    vib_data = VibStatesData(vib_ana_setup.states)
    vibdiff_cache = VibDiffCache()

    subgrids_with_results = evaluators.evaluate_all_on_grids(subgrids, vib_data=vib_data, vibdiff_cache=vibdiff_cache, gamma=2.0)
    exit()

    # print('\n')
    # for d in subgrids_with_results:
    #     print(d)
    #     print(subgrids_with_results[d], '\n')

    domains.insert_results_to_grid_nd(spec_grid, subgrids_with_results, result_key='result')
    
    print('\n\n', spec_grid['result'])


def test_SpectralWindow_intersect():
    print()
    domain_a = SpectralWindow(bounds=((5., 30.),))
    
    sw1d_b = SpectralWindow(bounds=((1., 33.),))
    print(domain_a.intersect(sw1d_b))

    sw1d_c = SpectralWindow(bounds=((15., 20.),))
    print(domain_a.intersect(sw1d_c))
    
    sw1d_d = SpectralWindow(bounds=((18., 20.),))
    print(domain_a.intersect(sw1d_d))


def test_SpectralWindow_intersect_2d():
    print()
    domain_a = SpectralWindow(bounds=((5., 30.), (45., 50.)))
    
    sw1d_b = SpectralWindow(bounds=((1., 33.),(55., 57.)))
    assert domain_a.intersect(sw1d_b) is None 

    sw1d_c = SpectralWindow(bounds=((15., 20.), (41., 57.)))
    assert domain_a.intersect(sw1d_c).bounds == ((15.0, 20.0), (45.0, 50.0))
    
    sw1d_d = SpectralWindow(bounds=((33., 35.), (41., 47.)))
    assert domain_a.intersect(sw1d_d) is None
