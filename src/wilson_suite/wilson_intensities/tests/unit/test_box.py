from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import Box, SpectralFeature, ResLocGeoObject
import pytest
import copy

# ---------- VALID CASES ----------
def test_box_init_from_dict():
    b = Box({'A': (0.0, 1.0), 'B': (5.0, 10.0)})
    assert b.bounds == ({'A': (0.0, 1.0), 'B': (5.0, 10.0)})
    assert b.axes == ('A', 'B')
    assert b.grid is None

def test_box_init_from_tuple():
    b = Box(((0.0, 1.0), (5.0, 10.0)))
    assert b.bounds == {'0': (0.0, 1.0), '1': (5.0, 10.0)}
    assert b.axes == ('0', '1')

# ---------- ERROR CASES ----------
def test_invalid_dict_bound_length():
    with pytest.raises(ValueError, match="Invalid bound for 'B'"):
        Box({'A': (0.0, 1.0), 'B': (5.0, 10.0, 15.0)})

def test_invalid_tuple_bound_length():
    with pytest.raises(ValueError, match="tuple of"):
        Box(((0.0, 1.0, 2.0), (5.0, 10.0)))

def test_invalid_type_for_bounds():
    with pytest.raises(TypeError, match="must be either dict"):
        Box([("A", (0.0, 1.0))])

def test_non_numeric_bounds():
    with pytest.raises(TypeError, match="expected numeric"):
        Box({'A': ('low', 'high')})

def test_min_greater_than_max():
    with pytest.raises(ValueError, match="> max"):
        Box({'A': (10.0, 5.0)})

def test_box_expand():
    b = Box({'A': (7.0, 12.0), 'B': (25.0, 35.0)})
    b1 = b.expand({'A': 2., 'B': 4.})
    assert b1.bounds == {'A': (5.0, 14.0), 'B': (21.0, 39.0)}

    b.expand({'A': 4., 'B': 8.}, inplace=True)
    assert b.bounds == {'A': (3.0, 16.0), 'B': (17.0, 43.0)}

def test_box_expand_shrink():
    b = Box({'A': (7.0, 12.0), 'B': (25.0, 35.0)})
    b1 = b.expand({'A': -2., 'B': -4.})
    assert b1.bounds == {'A': (9.0, 10.0), 'B': (29.0, 31.0)}

def test_box_intersect():
    b = Box({'A': (7.0, 12.0), 'B': (25.0, 35.0)})
    b1 = Box({'A': (1.0, 8.0), 'B': (32.0, 38.0)})
    b2 = Box({'A': (13.0, 28.0), 'B': (5.0, 8.0)})
    b_b1 = b.intersect(b1)
    assert b_b1.bounds == {'A': (7.0, 8.0), 'B': (32.0, 35.0)}
    assert b.intersect(b1) == b1.intersect(b)

    b_b2 = b.intersect(b2)
    assert b_b2 is None

    b2 = Box({'A': (1.0, 8.0), 'B': (37.0, 48.0)})
    b_b2 = b.intersect(b2)
    assert b_b2 is None

def test_union_of_boxes():
    b = Box({'A': (7.0, 12.0), 'B': (25.0, 35.0)})
    b1 = Box({'A': (1.0, 8.0), 'B': (32.0, 38.0)})
    b_b1 = b.intersect(b1)

    b_b1_U_b1 = Box.union([b_b1, b1])

    assert b_b1_U_b1 == b1
    assert b1.contains_box(b_b1)

    b2 = Box({'A': (1.0, 8.0), 'B': (37.0, 48.0)})
    b_U_b2 = Box.union([b, b2])
    assert b_U_b2.bounds == {'A': (1.0, 12.0), 'B': (25.0, 48.0)}

def test_union_of_boxes_no_overlap():
    b = Box({'A': (7.0, 12.0), 'B': (25.0, 35.0)})
    b3 = Box({'A': (13.0, 28.0), 'B': (5.0, 8.0)})

    b_U_b3 = Box.union([b, b3])
    print()
    print(b)
    print(b3)
    print(b_U_b3)
    # still returns a union of the boxes areas
    assert b_U_b3.bounds == {'A': (7.0, 28.0), 'B': (5.0, 35.0)}

    b4 = Box({'B': (5.0, 8.0), 'C': (13.0, 28.0)})
    b_U_b4 = Box.union([b, b4])
    print('b_U_b4', b_U_b4)


def test_box_contains_box():
    b = Box({'A': (7.0, 12.0), 'B': (25.0, 35.0)})
    b1 = Box({'A': (1.0, 8.0), 'B': (32.0, 38.0)})
    b2 = Box({'A': (1.0, 8.0), 'B': (37.0, 48.0)})
    b3 = Box({'A': (7.0, 8.0), 'B': (32.0, 34.0)})
    assert b.contains_box(b3)
    assert b1.contains_box(b3)

    assert not any([b.contains_box(b1), b.contains_box(b2)])
    assert not any([b1.contains_box(b), b1.contains_box(b2)])
    assert not any([b2.contains_box(b), b2.contains_box(b1), b2.contains_box(b3)])


def test_box_overlaps():
    b = Box({'A': (7.0, 12.0), 'B': (25.0, 35.0)})
    b1 = Box({'A': (1.0, 8.0), 'B': (32.0, 38.0)})
    b2 = Box({'A': (1.0, 8.0), 'B': (37.0, 48.0)})
    b3 = Box({'A': (7.0, 8.0), 'B': (32.0, 34.0)})

    assert b.overlaps(b1)
    assert b.overlaps(b1) == b.overlaps(b1)
    assert not b.overlaps(b2)
    assert not b2.overlaps(b)
    assert b.overlaps(b3)
    assert b1.overlaps(b3)
    assert b1.overlaps(b3) == b3.overlaps(b1)
    assert not b2.overlaps(b3)


def test_box_contains_feature():
    print()
    b = Box({'A': (7.0, 15.0), 'B': (25.0, 35.0)})

    res_loc1d_a = ResLocGeoObject({'A': 10., 'B': 27.7})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter=2.5)
    assert b.contains_feature(sf1, mode='box') # the whole box of this feature lies within Box b
    assert b.contains_feature(sf1, mode='loc')

    sf1a = copy.deepcopy(sf1)
    sf1a.feat_box = None
    with pytest.raises(ValueError, match="Need to add a box for this feature"):
        b.contains_feature(sf1a, mode='box')

    res_loc1d_b = ResLocGeoObject({'A': 6.5, 'B': 26.})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter=1.5)
    assert b.contains_feature(sf2, mode='box')
    assert not b.contains_feature(sf2, mode='loc')
    
    res_loc1d_c = ResLocGeoObject({'A': 17.5, 'B': 37.})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter=3.5)
    assert b.contains_feature(sf3, mode='box')
    assert not b.contains_feature(sf3, mode='loc')


def test_box_contributing_feature():
    print()
    b = Box({'A': (7.0, 12.0), 'B': (25.0, 35.0)})

    res_loc1d_a = ResLocGeoObject({'A': 10., 'B': 27.})
    sf1 = SpectralFeature(location=res_loc1d_a, lineshape_parameter=3.5)
    assert not b.contributing_feature(sf1)

    sf1a = copy.deepcopy(sf1)
    sf1a.feat_box = None
    sf1a.lineshape_parameter = None
    with pytest.raises(ValueError, match="Expected SpectralFeature with `lineshape_parameter` attribute"):
        b.contributing_feature(sf1a)

    res_loc1d_b = ResLocGeoObject({'A': 7.5, 'B': 15.})
    sf2 = SpectralFeature(location=res_loc1d_b, lineshape_parameter=3.5)
    assert not b.contributing_feature(sf2)
    
    res_loc1d_c = ResLocGeoObject({'A': 19.5, 'B': 26.})
    sf3 = SpectralFeature(location=res_loc1d_c, lineshape_parameter=3.5)
    assert not b.contributing_feature(sf3)

    res_loc1d_d = ResLocGeoObject({'A': 15.5, 'B': 23.})
    sf4 = SpectralFeature(location=res_loc1d_d, lineshape_parameter=3.5)
    assert b.contributing_feature(sf4)



