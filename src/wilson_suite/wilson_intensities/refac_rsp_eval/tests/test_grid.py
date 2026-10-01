"""
grid.py — bounds, boxes and box clustering. Runs with zero molecular data.

The last tests cover features_to_clusters (features.py), the feature-level wrapper of the box clustering.
"""

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    features_to_clusters,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
    compute_box_adjacency,
    connected_components_from_adjacency,
    points_to_bounds,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    AXES,
    NDIMS,
    coords,
    cube,
    feat,
)

## points_to_bounds ---------------------------------------------------------

def test_points_to_bounds_makes_one_bounds_dict_per_point():
    assert points_to_bounds([{'A': 0.}, {'A': 5.}], halfwidth=1.) == [{'A': (-1., 1.)}, {'A': (4., 6.)}]
    assert points_to_bounds([], halfwidth=1.) == []


## Box: construction --------------------------------------------------------

def test_box_sorts_axes_by_name():
    box = Box({'B': (0., 1.), 'A': (2., 3.)})

    assert box.axes == ('A', 'B')
    assert list(box.bounds) == ['A', 'B']
    assert box.ndim == 2


@pytest.mark.parametrize('bad', [[0., 1.], (0., 1., 2.), (0.,)])
def test_box_rejects_bounds_that_are_not_a_pair(bad):
    with pytest.raises(ValueError, match=r'expected \(min, max\)'):
        Box({'A': bad})


def test_box_rejects_non_numeric_bounds():
    with pytest.raises(TypeError):
        Box({'A': ('0', 1.)}) # type: ignore


def test_box_rejects_min_above_max():
    with pytest.raises(ValueError, match='min'):
        Box({'A': (2., 1.)})


def test_box_equality_and_hash_ignore_input_order():
    b1 = Box({'A': (0., 1.), 'B': (2., 3.)})
    b2 = Box({'B': (2., 3.), 'A': (0., 1.)})

    assert b1 == b2
    assert hash(b1) == hash(b2)


def test_box_hash_tells_different_bounds_apart():
    assert len({Box({'A': (0., 1.)}), Box({'A': (0., 1.)}), Box({'A': (0., 2.)})}) == 2


def test_box_accepts_int_bounds_and_zero_width():
    box = Box({'A': (0, 1), 'B': (2., 2.)})

    assert box.bounds == {'A': (0, 1), 'B': (2., 2.)}


## Box: operations ----------------------------------------------------------

def test_box_expand_returns_new_box_and_pads_only_given_axes():
    box = Box({'A': (0., 1.), 'B': (0., 1.)})

    bigger = box.expand({'A': 0.5})

    assert bigger.bounds == {'A': (-0.5, 1.5), 'B': (0., 1.)}
    assert box.bounds == {'A': (0., 1.), 'B': (0., 1.)}


def test_box_expand_inplace_changes_the_box():
    box = Box({'A': (0., 1.)})

    assert box.expand({'A': 1.}, inplace=True) is box
    assert box.bounds == {'A': (-1., 2.)}


def test_box_intersect_returns_overlap_on_shared_axes():
    b1 = Box({'A': (0., 2.), 'B': (0., 2.)})

    assert b1.intersect(Box({'A': (1., 3.), 'B': (-1., 1.)})) == Box({'A': (1., 2.), 'B': (0., 1.)})
    assert b1.intersect(Box({'A': (1., 3.)})) == Box({'A': (1., 2.)})


@pytest.mark.parametrize('other', [
    Box({'A': (5., 6.)}),  # apart
    Box({'A': (1., 2.)}),  # touching
    Box({'B': (0., 1.)}),  # no shared axis
])
def test_box_intersect_returns_none_without_overlap(other):
    assert Box({'A': (0., 1.)}).intersect(other) is None


def test_box_union_spans_all_boxes_and_all_axes():
    union = Box.union([Box({'A': (0., 1.)}), Box({'A': (5., 6.), 'B': (2., 3.)})])

    assert union == Box({'A': (0., 6.), 'B': (2., 3.)})


def test_box_union_of_one_box_is_that_box():
    box = Box({'A': (0., 1.), 'B': (2., 3.)})

    assert Box.union([box]) == box


def test_box_contains_box():
    outer = Box({'A': (0., 10.)})

    assert outer.contains_box(Box({'A': (2., 3.)}))
    assert outer.contains_box(outer)
    assert not outer.contains_box(Box({'A': (9., 11.)}))


def test_box_overlaps_excludes_touching_edges():
    box = Box({'A': (0., 1.)})

    assert box.overlaps(Box({'A': (0.5, 2.)}))
    assert not box.overlaps(Box({'A': (1., 2.)}))
    assert not box.overlaps(Box({'A': (3., 4.)}))


def test_box_contains_points_needs_one_coord_per_axis():
    with pytest.raises(ValueError, match='coords'):
        Box({'A': (0., 1.)}).contains(np.array([[0.5, 0.5]]))


## Box: N dimensions (1 to 4 axes) ------------------------------------------

@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_points_to_bounds_pads_every_axis_by_halfwidth(ndim):
    # a different value on each axis: A=10, B=20, ...
    point = {ax: 10. * (i + 1) for i, ax in enumerate(AXES[:ndim])}

    [bounds] = points_to_bounds([point], halfwidth=2.)

    assert bounds == {ax: (v - 2., v + 2.) for ax, v in point.items()}
    assert Box(bounds).ndim == ndim


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_box_intersect_union_and_contains_box(ndim):
    b1, b2 = cube(ndim, 0., 2.), cube(ndim, 1., 3.)

    assert b1.intersect(b2) == cube(ndim, 1., 2.)
    assert Box.union([b1, b2]) == cube(ndim, 0., 3.)
    assert Box.union([b1, b2]).contains_box(b1)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_box_overlap_needs_every_axis(ndim):
    box = cube(ndim, 0., 2.)
    apart_on_last_axis = cube(ndim, 1., 3., **{AXES[ndim - 1]: (5., 6.)})

    assert box.overlaps(cube(ndim, 1., 3.))
    assert not box.overlaps(apart_on_last_axis)
    assert box.intersect(apart_on_last_axis) is None


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_box_contains_points(ndim):
    points = np.array([[0.5] * ndim,
                       [0.5] * (ndim - 1) + [2.]])  # outside on the last axis only

    np.testing.assert_array_equal(cube(ndim, 0., 1.).contains(points), [True, False])


## Box clustering -----------------------------------------------------------

def test_adjacency_is_symmetric_with_false_diagonal():
    boxes = [Box({'A': (0., 2.)}), Box({'A': (1., 3.)}), Box({'A': (10., 11.)})]

    np.testing.assert_array_equal(compute_box_adjacency(boxes), [[False, True, False],
                                                                 [True, False, False],
                                                                 [False, False, False]])


def test_adjacency_counts_touching_boxes_only_when_inclusive():
    boxes = [Box({'A': (0., 1.)}), Box({'A': (1., 2.)})]

    assert compute_box_adjacency(boxes, touch_inclusive=True)[0, 1]
    assert not compute_box_adjacency(boxes, touch_inclusive=False)[0, 1]


def test_adjacency_needs_overlap_on_every_checked_axis():
    boxes = [Box({'A': (0., 2.), 'B': (0., 1.)}), Box({'A': (1., 3.), 'B': (5., 6.)})]

    assert not compute_box_adjacency(boxes)[0, 1]
    assert compute_box_adjacency(boxes, axis_order=('A',))[0, 1]


def test_adjacency_counts_touching_corners():
    boxes = [Box({'A': (0., 1.), 'B': (0., 1.)}), Box({'A': (1., 2.), 'B': (1., 2.)})]

    assert compute_box_adjacency(boxes)[0, 1]


def test_adjacency_of_no_boxes_is_empty():
    assert compute_box_adjacency([]).shape == (0, 0)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_adjacency_needs_overlap_on_every_axis(ndim):
    boxes = [cube(ndim, 0., 2.), cube(ndim, 1., 3.), cube(ndim, 1., 3., **{AXES[ndim - 1]: (5., 6.)})]

    adjacency = compute_box_adjacency(boxes)

    assert adjacency[0, 1]
    assert not adjacency[0, 2]


def test_connected_components_follow_chains():
    # a-b and b-c are linked, d is alone
    adjacency = np.zeros((4, 4), dtype=bool)
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[1, 2] = adjacency[2, 1] = True

    clusters = connected_components_from_adjacency(adjacency, ['a', 'b', 'c', 'd'])

    assert {k: sorted(v) for k, v in clusters.items()} == {0: ['a', 'b', 'c'], 1: ['d']}


def test_connected_components_of_unlinked_objects_are_singletons():
    clusters = connected_components_from_adjacency(np.zeros((3, 3), dtype=bool), ['a', 'b', 'c'])

    assert clusters == {0: ['a'], 1: ['b'], 2: ['c']}


def test_connected_components_of_nothing_is_empty():
    assert connected_components_from_adjacency(np.zeros((0, 0), dtype=bool), []) == {}


def test_features_to_clusters_joins_touching_boxes():
    # boxes: (-1, 1) and (1, 3) touch at 1
    f0, f1 = feat(A=0.), feat(A=2.)

    assert list(features_to_clusters([f0, f1]).values()) == [[f0, f1]]


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_features_to_clusters_needs_overlap_on_every_axis(ndim):
    # boxes: f0 (-1, 1) and f1 (0.5, 2.5) on every axis; f2 sits on f0, except (9, 11) on the last axis
    f0, f1 = feat(coords(ndim, 0.)), feat(coords(ndim, 1.5))
    f2 = feat(coords(ndim, 0., **{AXES[ndim - 1]: 10.}))

    assert list(features_to_clusters([f0, f1, f2]).values()) == [[f0, f1], [f2]]


def test_features_to_clusters_needs_feature_boxes():
    with pytest.raises(ValueError, match='feature box'):
        features_to_clusters([feat(A=0.), feat(A=1., gamma=None)])


def test_features_to_clusters_of_no_features_is_empty():
    assert features_to_clusters([]) == {}

