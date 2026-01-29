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
    print()
    max_intensity=abs(110.)**2 
    min_intensity=abs(18.)**2
    print('min_intensity', min_intensity)

    feats = SpectralFeature.dress_these_with_boxes(features=[sf1, sf2, sf3], 
                                                   max_intensity=max_intensity, 
                                                   min_intensity=min_intensity)
    
    print(feats)
    print(len(feats))
    assert len(feats) == 2
    f1, f2 = feats

    import numpy as np
    np.testing.assert_allclose(
        f1.feat_box.bounds["A"],
        (11.354502775632097, 12.645497224367903),
        rtol=1e-12,
        atol=0.0,
    )


    np.testing.assert_allclose(
        f2.feat_box.bounds["A"],
        (5.712699117539398, 9.2873008824606),
        rtol=1e-12,
        atol=0.0,
    )

def test_dress_these_with_boxes_2d():
    print()

    # --- define 2D locations ---
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
    print()

    max_intensity = abs(110.)**2
    min_intensity = abs(18.)**2
    print('min_intensity', min_intensity)

    feats = SpectralFeature.dress_these_with_boxes(
        features=[sf1, sf2, sf3],
        max_intensity=max_intensity,
        min_intensity=min_intensity,
    )

    print(feats)
    print(len(feats))

    assert len(feats) == 2
    f1, f2 = feats

    assert set(f1.feat_box.bounds.keys()) == {'A', 'B'}
    assert set(f2.feat_box.bounds.keys()) == {'A', 'B'}

    import numpy as np
    np.testing.assert_allclose(
        f1.feat_box.bounds["A"],
        (11.354502775632097, 12.645497224367903),
    )
    np.testing.assert_allclose(
        f1.feat_box.bounds["B"],
        (4.354502775632097, 5.645497224367903),
    )
    
    np.testing.assert_allclose(
        f2.feat_box.bounds["A"],
        (5.712699117539398, 9.2873008824606),
    )
    np.testing.assert_allclose(
        f2.feat_box.bounds["B"],
        (0.21269911753939863, 3.7873008824606016),
    )
