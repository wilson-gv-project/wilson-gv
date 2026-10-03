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

def test_dress_these_with_boxes_1d():
    print()

    res_loc1d_a = ResLocGeoObject({'A': 1200.})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter=3.5, amplitude_coeff=120.)
    res_loc1d_b = ResLocGeoObject({'A': 750})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter=4.5, amplitude_coeff=170.)
    res_loc1d_c = ResLocGeoObject({'A': 350})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter=7.5, amplitude_coeff=10.)

    sfs = [sf1, sf2, sf3]

    import numpy as np

    max_int = np.max([i.get_intensity() for i in sfs])


    feats = SpectralFeature.dress_these_with_boxes(features=sfs,
                                                   max_intensity=max_int,
                                                   min_intensity=max_int/200)
    
    SpectralFeature.print_list_features(feats)
    


    for i in feats:
        print('Feature with location', i.location, 'lineshape parameter', i.lineshape_parameter, 'and point intensity', i.get_intensity())
        print('A bounds:', i.feat_box.bounds["A"])
        print('\n')

    assert len(feats) == 2
    f1, f2 = feats

    import numpy as np
    np.testing.assert_allclose(
        f1.feat_box.bounds["A"],
        (1145.6890664782863, 1254.3109335217137),
        rtol=1e-12,
        atol=0.0,
    )


    np.testing.assert_allclose(
        f2.feat_box.bounds["A"],
        (680.1716569006538, 819.8283430993462),
        rtol=1e-12,
        atol=0.0,
    )

def test_dress_these_with_boxes_2d():

    import numpy as np

    res_loc2d_a = ResLocGeoObject({'A': 1000., 'B': 500.})
    sf1 = SpectralFeature(
        location=res_loc2d_a,
        lineshape_parameter=4.5,
        amplitude_coeff=120.,
    )

    res_loc2d_b = ResLocGeoObject({'A': 1250., 'B': 600.})
    sf2 = SpectralFeature(
        location=res_loc2d_b,
        lineshape_parameter=4.5,
        amplitude_coeff=30.,
    )

    res_loc2d_c = ResLocGeoObject({'A': 350., 'B': 1000.})
    sf3 = SpectralFeature(
        location=res_loc2d_c,
        lineshape_parameter=5.,
        amplitude_coeff=80.,
    )

    res_loc2d_d = ResLocGeoObject({'A': 350., 'B': 1000.})
    sf4 = SpectralFeature(
        location=res_loc2d_d,
        lineshape_parameter=2.5,
        amplitude_coeff=10.,
    )

    res_loc2d_e = ResLocGeoObject({'A': 550., 'B': 550.})
    sf5 = SpectralFeature(
        location=res_loc2d_e,
        lineshape_parameter=3.5,
        amplitude_coeff=230.,
    )

    sfs = [sf1, sf2, sf3, sf4, sf5]

    max_int = np.max([i.get_intensity() for i in sfs])
    min_int = np.min([i.get_intensity() for i in sfs])


    # This setup should keep all five features and result in somewhat large boxes

    feats = SpectralFeature.dress_these_with_boxes(
        features=sfs,
        max_intensity=max_int,
        min_intensity=min_int,
        box_range_safety_margin=0.5
    )

    print('Number of features:', len(feats))

    for i in feats:
        print('Feature with location', i.location, 'lineshape parameter', i.lineshape_parameter, 'and point intensity', i.get_intensity())
        print('A bounds:', i.feat_box.bounds["A"], 'B bounds', i.feat_box.bounds["B"])
        print('\n')

    assert len(feats) == 5

    np.testing.assert_allclose(
        feats[0].feat_box.bounds["A"],
        (914.7208017784361, 1085.279198221564),
    )
    np.testing.assert_allclose(
        feats[0].feat_box.bounds["B"],
        (414.7208017784361, 585.2791982215639),
    )

    np.testing.assert_allclose(
        feats[1].feat_box.bounds["A"],
        (1164.720801778436, 1335.279198221564),
    )
    np.testing.assert_allclose(
        feats[1].feat_box.bounds["B"],
        (514.7208017784361, 685.2791982215639),
    )

    np.testing.assert_allclose(
        feats[2].feat_box.bounds["A"],
        (255.24533530937344, 444.75466469062656),
    )
    np.testing.assert_allclose(
        feats[2].feat_box.bounds["B"],
        (905.2453353093734, 1094.7546646906267),
    )

    np.testing.assert_allclose(
        feats[3].feat_box.bounds["A"],
        (302.6226676546867, 397.3773323453133),
    )
    np.testing.assert_allclose(
        feats[3].feat_box.bounds["B"],
        (952.6226676546867, 1047.3773323453133),
    )

    np.testing.assert_allclose(
        feats[4].feat_box.bounds["A"],
        (483.6717347165614, 616.3282652834386),
    )
    np.testing.assert_allclose(
        feats[4].feat_box.bounds["B"],
        (483.6717347165614, 616.3282652834386),
    )

    # This setup (dynamic range 100) should keep three features (discard 2)

    feats = SpectralFeature.dress_these_with_boxes(
        features=sfs,
        max_intensity=max_int,
        min_intensity=max_int/100,
        box_range_safety_margin=0.0
    )

    print('Number of features:', len(feats))

    for i in feats:
        print('Feature with location', i.location, 'lineshape parameter', i.lineshape_parameter, 'and point intensity', i.get_intensity())
        print('A bounds:', i.feat_box.bounds["A"], 'B bounds', i.feat_box.bounds["B"])
        print('\n')

    assert len(feats) == 3

    np.testing.assert_allclose(
        feats[0].feat_box.bounds["A"],
        (955.2255653302021, 1044.774434669798),
    )
    np.testing.assert_allclose(
        feats[0].feat_box.bounds["B"],
        (455.22556533020213, 544.7744346697979),
    )

    np.testing.assert_allclose(
        feats[1].feat_box.bounds["A"],
        (300.250628144669, 399.749371855331),
    )
    np.testing.assert_allclose(
        feats[1].feat_box.bounds["B"],
        (950.250628144669, 1049.749371855331),
    )

    np.testing.assert_allclose(
        feats[2].feat_box.bounds["A"],
        (515.1754397012683, 584.8245602987317),
    )
    np.testing.assert_allclose(
        feats[2].feat_box.bounds["B"],
        (515.1754397012683, 584.8245602987317),
    )

    # This setup represents what could be considered a suggested usage for now (exact parameters may need adjustment):
    # Scale box sizes according to the max intensity over the collection of features and apply minimum box padding.
    # Results in keeping all 5 features of which 3 get minimum box sizes

    feats = SpectralFeature.dress_these_with_boxes(
        features=sfs,
        max_intensity=max_int,
        min_intensity=max_int/100,
        scale_wrt_max_intensity=True,
        minimum_box_padding=12.0
    )

    print('Number of features:', len(feats))

    for i in feats:
        print('Feature with location', i.location, 'lineshape parameter', i.lineshape_parameter, 'and point intensity', i.get_intensity())
        print('A bounds:', i.feat_box.bounds["A"], 'B bounds', i.feat_box.bounds["B"])
        print('\n')

    assert len(feats) == 5

    np.testing.assert_allclose(
        feats[0].feat_box.bounds["A"],
        (985.1817168348343, 1014.8182831651657),
    )
    np.testing.assert_allclose(
        feats[0].feat_box.bounds["B"],
        (485.18171683483433, 514.8182831651657),
    )

    # This feature has central intensity below the dynamic range cutoff but gets a minimum box for safety
    np.testing.assert_allclose(
        feats[1].feat_box.bounds["A"],
        (1238.0, 1262.0),
    )
    np.testing.assert_allclose(
        feats[1].feat_box.bounds["B"],
        (588.0, 612.0),
    )

    # This feature has central intensity above the dynamic range cutoff, but the box size would be smaller than the
    # minimum box, and so gets "upgraded" to a minimum box
    np.testing.assert_allclose(
        feats[2].feat_box.bounds["A"],
        (338.0, 362.0),
    )
    np.testing.assert_allclose(
        feats[2].feat_box.bounds["B"],
        (988.0, 1012.0),
    )

    # This feature has central intensity below the dynamic range cutoff but gets a minimum box for safety
    np.testing.assert_allclose(
        feats[3].feat_box.bounds["A"],
        (338.0, 362.0),
    )
    np.testing.assert_allclose(
        feats[3].feat_box.bounds["B"],
        (988.0, 1012.0),
    )

    np.testing.assert_allclose(
        feats[4].feat_box.bounds["A"],
        (511.69298367139515, 588.3070163286048),
    )
    np.testing.assert_allclose(
        feats[4].feat_box.bounds["B"],
        (511.69298367139515, 588.3070163286048),
    )

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