from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import Box, SpectralFeature, ResLocGeoObject, SpectralWindow
import pytest
import copy

def test_union():
    res_loc1d_a = ResLocGeoObject({'A': 12.})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter=3.5)
    res_loc1d_b = ResLocGeoObject({'A': 9.5})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter=4.5, amplitude_coeff=2.5)
    res_loc1d_c = ResLocGeoObject({'A': 9.5})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter=4.5, amplitude_coeff=1.5)

    # equal because both have empty term_contributions
    assert sf2 == sf3
    sf2_U_sf3 = sf3.union(sf2)
    assert sf2_U_sf3.amplitude_coeff == 4.0

    # because locations are different
    assert sf1 != sf3

    with pytest.raises(ValueError, match="both location and lineshape_parameter are the same"):
        sf1.union(sf3)

def test_union_p2():
    res_loc1d_b = ResLocGeoObject({'A': 9.5})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter=4.5, amplitude_coeff=2.5, term_contributions=('smth',))
    res_loc1d_c = ResLocGeoObject({'A': 9.5})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter=4.5, amplitude_coeff=1.5, term_contributions=('smth else',))
    sf2_U_sf3 = sf3.union(sf2)
    assert sf2_U_sf3.term_contributions == ('smth else', 'smth')

def test_equality():
    print()
    res_loc1d_a = ResLocGeoObject({'A': 9.5})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter=4.5, amplitude_coeff=2.5, term_contributions=('smth',))

    res_loc1d_b = ResLocGeoObject({'A': 9.5})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter=4.5, amplitude_coeff=2.5)
    res_loc1d_c = ResLocGeoObject({'A': 9.5})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter=4.5, amplitude_coeff=1.5)
    res_loc1d_d = ResLocGeoObject({'A': 1.5})
    sf4 = SpectralFeature(location=res_loc1d_d, lineshape_parameter=4.5, amplitude_coeff=1.5)

    # equal because both have empty term_contributions
    assert sf2 == sf3

    assert sf1 != sf2
    assert SpectralFeature.share_location([sf1, sf2, sf3])
    assert not SpectralFeature.share_location([sf1, sf2, sf3, sf4])
    assert not SpectralFeature.share_location([sf1, sf4])

def test_filter_to_spec_window():
    sw1d = SpectralWindow(box=Box({'A': (5., 10.)}))

    res_loc1d_a = ResLocGeoObject({'A': 12.})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter=2.5)
    res_loc1d_b = ResLocGeoObject({'A': 7.5})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter=2.5)
    res_loc1d_c = ResLocGeoObject({'A': 3.5})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter=2.5)

    sp = SpectralFeature.filter_to_spec_window([sf1, sf2, sf3], sw1d)

    assert sf1 in sp.contrib_features
    assert sf3 in sp.contrib_features
    assert sf2 in sp.full_features

def test_feature_get_res_motifs():
    print('\n')

    from .test_domains import get_features_from_terms
    features = get_features_from_terms()

    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()
    t_inds = [0, 1,-1, -2, -3]
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    terms_hashes = {t.h(): t for t in terms_select}
    # print(terms_hashes)
    
    res_motifs_feat0 = features[0].get_res_motifs()
    
    print(res_motifs_feat0)
    for refmotif in res_motifs_feat0:
        for recond in refmotif:
            print(recond)

def test_dress_these_with_boxes_1d():
    print()

    res_loc1d_a = ResLocGeoObject({'A': 12.})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter=2.5, amplitude_coeff=120.)
    res_loc1d_b = ResLocGeoObject({'A': 7.5})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter=2.5, amplitude_coeff=170.)
    res_loc1d_c = ResLocGeoObject({'A': 3.5})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter=2.5, amplitude_coeff=30.)

    print([sf1, sf2, sf3])
    print(f"{sf1.get_intensity():.3e}", f"{sf2.get_intensity():.3e}", f"{sf3.get_intensity():.3e}")
    print()
    max_intensity=2.3e14
    min_intensity=1.2e13
    print(sf2.get_intensity()<max_intensity)
    print(f"{sf2.get_intensity():.3e}", f"{max_intensity:.3e}")
    feats = SpectralFeature.dress_these_with_boxes(features=[sf1, sf2, sf3], 
                                                   max_intensity=max_intensity, 
                                                   min_intensity=min_intensity)
    
    SpectralFeature.print_list_features(feats)
    
    assert len(feats) == 2
    f1, f2 = feats

    import numpy as np
    np.testing.assert_allclose(
        f1.feat_box.bounds["A"],
        (4.8199626341930495, 19.18003736580695),
        rtol=1e-12,
        atol=0.0,
    )


    np.testing.assert_allclose(
        f2.feat_box.bounds["A"],
        (-3.270667687307265, 18.270667687307267),
        rtol=1e-12,
        atol=0.0,
    )


def test_dress_these_with_boxes_2d():
    print()

    # --- 2D locations ---
    res_loc2d_a = ResLocGeoObject({'A': 12., 'B': 5.0})
    sf1 = SpectralFeature(
        location=res_loc2d_a,
        lineshape_parameter=2.5,
        amplitude_coeff=120.,
    )

    res_loc2d_b = ResLocGeoObject({'A': 7.5, 'B': 2.0})
    sf2 = SpectralFeature(
        location=res_loc2d_b,
        lineshape_parameter=2.5,
        amplitude_coeff=170.,
    )

    res_loc2d_c = ResLocGeoObject({'A': 3.5, 'B': 8.0})
    sf3 = SpectralFeature(
        location=res_loc2d_c,
        lineshape_parameter=2.5,
        amplitude_coeff=30.,
    )

    print([sf1, sf2, sf3])
    print(f"{sf1.get_intensity():.3e}", f"{sf2.get_intensity():.3e}", f"{sf3.get_intensity():.3e}")
    print()

    max_intensity = 1.8e24
    min_intensity = 1.9e23

    feats = SpectralFeature.dress_these_with_boxes(
        features=[sf1, sf2, sf3],
        max_intensity=max_intensity,
        min_intensity=min_intensity,
    )

    SpectralFeature.print_list_features(feats)

    assert len(feats) == 2
    f1, f2 = feats

    assert set(f1.feat_box.bounds.keys()) == {'A', 'B'}
    assert set(f2.feat_box.bounds.keys()) == {'A', 'B'}

    import numpy as np
    np.testing.assert_allclose(
        f1.feat_box.bounds["A"],
        (9.352188995776729, 14.647811004223271),
    )
    np.testing.assert_allclose(
        f1.feat_box.bounds["B"],
        (2.3521889957767295, 7.6478110042232705),
    )

    np.testing.assert_allclose(
        f2.feat_box.bounds["A"],
        (3.1656858199373845, 11.834314180062616),
    )
    np.testing.assert_allclose(
        f2.feat_box.bounds["B"],
        (-2.3343141800626155, 6.3343141800626155),
    )

def test_dress_these_with_boxes_2d_b():
    print()

    import numpy as np

    # --- 2D locations ---
    res_loc2d_a = ResLocGeoObject({'A': 1000., 'B': 500.})
    sf1 = SpectralFeature(
        location=res_loc2d_a,
        lineshape_parameter=4.5,
        amplitude_coeff=120.,
    )

    res_loc2d_b = ResLocGeoObject({'A': 1015., 'B': 500.})
    sf2 = SpectralFeature(
        location=res_loc2d_b,
        lineshape_parameter=10.5,
        amplitude_coeff=170.,
    )

    res_loc2d_c = ResLocGeoObject({'A': 1250., 'B': 600.})
    sf3 = SpectralFeature(
        location=res_loc2d_c,
        lineshape_parameter=4.5,
        amplitude_coeff=30.,
    )

    res_loc2d_d = ResLocGeoObject({'A': 350., 'B': 1000.})
    sf4 = SpectralFeature(
        location=res_loc2d_d,
        lineshape_parameter=5.,
        amplitude_coeff=80.,
    )

    res_loc2d_e = ResLocGeoObject({'A': 350., 'B': 1000.})
    sf5 = SpectralFeature(
        location=res_loc2d_e,
        lineshape_parameter=2.5,
        amplitude_coeff=50.,
    )

    res_loc2d_f = ResLocGeoObject({'A': 350., 'B': 1400.})
    sf6 = SpectralFeature(
        location=res_loc2d_f,
        lineshape_parameter=12.5,
        amplitude_coeff=55.,
    )

    res_loc2d_g = ResLocGeoObject({'A': 550., 'B': 550.})
    sf7 = SpectralFeature(
        location=res_loc2d_g,
        lineshape_parameter=3.5,
        amplitude_coeff=230.,
    )

    res_loc2d_h = ResLocGeoObject({'A': 800., 'B': 800.})
    sf8 = SpectralFeature(
        location=res_loc2d_h,
        lineshape_parameter=2.5,
        amplitude_coeff=30.,
    )

    sfs = [sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8]

    max_int = np.max([i.get_intensity() for i in sfs])
    min_int = np.min([i.get_intensity() for i in sfs])

    print('max min', max_int, min_int)

    for i in sfs:
        print(i)
        print(i.get_intensity())

    #print([sf1, sf2, sf3])
    #print(f"{sf1.get_intensity():.3e}", f"{sf2.get_intensity():.3e}", f"{sf3.get_intensity():.3e}")
    print()

    max_intensity = max_int
    min_intensity = min_int * 2.0

    feats = SpectralFeature.dress_these_with_boxes(
        features=sfs,
        max_intensity=max_intensity,
        min_intensity=max_intensity/200,
        scale_wrt_max_intensity=True
    )

    for i in feats:
        print('Feature with location', i.location, 'lineshape parameter', i.lineshape_parameter, 'and point intensity', i.get_intensity())
        print('A bounds:', i.feat_box.bounds["A"], 'B bounds', i.feat_box.bounds["B"])
        print('\n')

    #SpectralFeature.print_list_features(feats)

    #print(feats)

    #assert len(feats) == 2
    #f1, f2 = feats

    #assert set(f1.feat_box.bounds.keys()) == {'A', 'B'}
    #assert set(f2.feat_box.bounds.keys()) == {'A', 'B'}

    import numpy as np
    #np.testing.assert_allclose(
    #    f1.feat_box.bounds["A"],
    #    (9.352188995776729, 14.647811004223271),
    #)
    #np.testing.assert_allclose(
    #    f1.feat_box.bounds["B"],
    #    (2.3521889957767295, 7.6478110042232705),
    #)
    #
    #np.testing.assert_allclose(
    #    f2.feat_box.bounds["A"],
    #    (3.1656858199373845, 11.834314180062616),
    #)
    #np.testing.assert_allclose(
    #    f2.feat_box.bounds["B"],
    #    (-2.3343141800626155, 6.3343141800626155),
    #)


def test_get_intensity_SpecFeature():
    print()
    res_loc1d_a = ResLocGeoObject({'A': 12.})
    sf1_1d = SpectralFeature(location=res_loc1d_a, lineshape_parameter=2.5, amplitude_coeff=120.)
    res_loc1d_b = ResLocGeoObject({'A': 7.5})
    sf2_1d = SpectralFeature(location=res_loc1d_b, lineshape_parameter=2.5, amplitude_coeff=170.)

    res_loc2d_a = ResLocGeoObject({'A': 12., 'B': 5.0})
    sf1_2d = SpectralFeature(
        location=res_loc2d_a,
        lineshape_parameter=2.5,
        amplitude_coeff=120.,
    )

    res_loc2d_b = ResLocGeoObject({'A': 7.5, 'B': 2.0})
    sf2_2d = SpectralFeature(
        location=res_loc2d_b,
        lineshape_parameter=2.5,
        amplitude_coeff=170.,
    )
    
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    assert sf1_1d.get_intensity() == abs(sf1_1d.amplitude_coeff/(-1j*convNu2Ene(sf1_1d.lineshape_parameter)))**2
    assert sf2_1d.get_intensity() == abs(sf2_1d.amplitude_coeff/(-1j*convNu2Ene(sf2_1d.lineshape_parameter)))**2

    assert sf1_2d.get_intensity() == abs(sf1_2d.amplitude_coeff/(-1j*convNu2Ene(sf1_2d.lineshape_parameter))/(-1j*convNu2Ene(sf1_2d.lineshape_parameter)))**2
    assert sf2_2d.get_intensity() == abs(sf2_2d.amplitude_coeff/(-1j*convNu2Ene(sf2_2d.lineshape_parameter))/(-1j*convNu2Ene(sf2_2d.lineshape_parameter)))**2


def test_get_max_intensity_feat_SpecFeature():
    print()
    res_loc1d_a = ResLocGeoObject({'A': 12.})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter=2.5, amplitude_coeff=120.)
    res_loc1d_b = ResLocGeoObject({'A': 7.5})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter=2.5, amplitude_coeff=170.)
    res_loc1d_c = ResLocGeoObject({'A': 3.5})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter=2.5, amplitude_coeff=30.)

    assert SpectralFeature.get_max_intensity_feat([sf1, sf2, sf3]) == sf2

    res_loc2d_a = ResLocGeoObject({'A': 12., 'B': 5.0})
    sf1_2d = SpectralFeature(
        location=res_loc2d_a,
        lineshape_parameter=2.5,
        amplitude_coeff=120.,
    )
    res_loc2d_b = ResLocGeoObject({'A': 7.5, 'B': 2.0})
    sf2_2d = SpectralFeature(
        location=res_loc2d_b,
        lineshape_parameter=2.5,
        amplitude_coeff=170.,
    )
    res_loc2d_c = ResLocGeoObject({'A': 3.5, 'B': 8.0})
    sf3_2d = SpectralFeature(
        location=res_loc2d_c,
        lineshape_parameter=2.5,
        amplitude_coeff=30.,
    )
    assert SpectralFeature.get_max_intensity_feat([sf1_2d, sf2_2d, sf3_2d]) == sf2_2d


def test_dress_with_featboxes_SpecWindow():
    print()

    sw1d = SpectralWindow(box=Box({'A': (5., 10.)}))

    res_loc1d_a = ResLocGeoObject({'A': 12.})
    sf1 = SpectralFeature(location=res_loc1d_a, 
                          lineshape_parameter=2.5, amplitude_coeff=120.)
    res_loc1d_b = ResLocGeoObject({'A': 7.5})
    sf2 = SpectralFeature(location=res_loc1d_b, 
                          lineshape_parameter=2.5, amplitude_coeff=170.)
    res_loc1d_c = ResLocGeoObject({'A': 3.5})
    sf3 = SpectralFeature(location=res_loc1d_c, 
                          lineshape_parameter=2.5, amplitude_coeff=30.)

    sp = SpectralFeature.filter_to_spec_window([sf1, sf2, sf3], sw1d)

    assert len(sp.contrib_features) == 2
    assert len(sp.full_features) == 1

    new_specwindow = sp.dress_with_featboxes(10.)

    assert sp.bounds == new_specwindow.bounds
    assert len(new_specwindow.contrib_features) == 1
    assert len(new_specwindow.full_features) == 1