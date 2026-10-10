"""
grid.py — bounds, boxes, box slices and box clustering. Runs with zero molecular data.

The last tests cover features_to_clusters (features.py), the feature-level wrapper of the box clustering.
"""

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    features_to_clusters,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
    box_slices,
    compute_box_adjacency,
    connected_components_from_adjacency,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    AXES,
    NDIMS,
    coords,
    cube,
    feat,
)

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


def test_box_stores_numpy_numbers_as_plain_floats():
    box = Box({'A': (np.int64(0), np.int64(1)), 'B': (np.float32(2.), np.float32(3.))}) # type: ignore

    assert box.bounds == {'A': (0., 1.), 'B': (2., 3.)}
    assert all(type(v) is float for bounds in box.bounds.values() for v in bounds)


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


## Box: grid ----------------------------------------------------------------

def test_make_grid_shapes_and_ij_indexing():
    box = Box({'A': (0., 10.), 'B': (100., 200.)})

    axes, grid = box.make_grid({'A': 5, 'B': 3})

    assert (len(axes['A']), len(axes['B'])) == (5, 3)
    assert grid['A'].shape == grid['B'].shape == (5, 3)
    np.testing.assert_array_equal(grid['A'][:, 0], axes['A'])
    np.testing.assert_array_equal(grid['B'][0, :], axes['B'])


def test_make_grid_runs_from_min_to_max_with_both_edges():
    box = Box({'A': (0., 10.), 'B': (100., 200.)})

    axes, _ = box.make_grid({'A': 11, 'B': 5})

    np.testing.assert_allclose(axes['A'], [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    np.testing.assert_allclose(axes['B'], [100., 125., 150., 175., 200.])


@pytest.mark.parametrize('dim_sizes', [
    {'A': 5, 'B': 5},  # one axis too many
    {'B': 5},          # same number of axes, other name
])
def test_make_grid_needs_one_size_per_box_axis(dim_sizes):
    with pytest.raises(ValueError, match='one grid size per box axis'):
        Box({'A': (0., 10.)}).make_grid(dim_sizes)


## box_slices ---------------------------------------------------------------

POINTS_0_TO_9 = {'A': np.arange(10.)}


@pytest.mark.parametrize('bounds, expected', [
    ((2.5, 4.5), slice(2, 6)),    # one more point on each side: 2 and 5
    ((3., 4.), slice(3, 5)),      # edges on grid points: nothing extra
    ((3., 3.), slice(3, 4)),      # zero width
    ((-5., 0.5), slice(0, 2)),    # cut at the low grid edge
    ((8.5, 20.), slice(8, 10)),   # cut at the high grid edge
    ((9., 12.), slice(9, 10)),    # touches the grid edge: point 9 lies in the box
    ((-3., 0.), slice(0, 1)),
])
def test_box_slices_smallest_range_that_covers_the_box(bounds, expected):
    assert box_slices(Box({'A': bounds}), POINTS_0_TO_9) == (expected,)


@pytest.mark.parametrize('bounds', [(12., 15.), (9.5, 12.), (-5., -0.5)])
def test_box_slices_of_a_box_outside_the_grid_is_none(bounds):
    assert box_slices(Box({'A': bounds}), POINTS_0_TO_9) is None


def test_box_slices_is_none_when_outside_on_one_axis_only():
    coords = {'A': np.arange(10.), 'B': np.arange(10.)}

    assert box_slices(Box({'A': (2., 4.), 'B': (20., 30.)}), coords) is None


def test_box_slices_follow_the_axis_order_of_coords():
    """One slice per array dimension of the grid, so the order of coords counts, not the box's sorted axes."""
    coords = {'B': np.arange(10.), 'A': np.arange(100., 110.)}

    assert box_slices(Box({'A': (101., 102.), 'B': (5., 7.)}), coords) == (slice(5, 8), slice(1, 3))


@pytest.mark.parametrize('coords', [{'A': np.arange(10.), 'B': np.arange(10.)}, {'B': np.arange(10.)}])
def test_box_slices_needs_the_grid_axes_of_the_box(coords):
    with pytest.raises(ValueError, match='do not match grid axes'):
        box_slices(Box({'A': (2., 4.)}), coords)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_box_slices_one_slice_per_axis(ndim):
    """
    Grid points 0, 1, ..., 9 on every axis. Box (2.5, 4.5) on every axis, except (6., 6.5) on the last one:
        (2.5, 4.5)  ->  points 2, 3, 4, 5  ->  slice(2, 6)
        (6., 6.5)   ->  points 6, 7        ->  slice(6, 8)
    e.g. ndim=2: box {'A': (2.5, 4.5), 'B': (6., 6.5)}  ->  (slice(2, 6), slice(6, 8))
    """
    coords = {ax: np.arange(10.) for ax in AXES[:ndim]}
    box = cube(ndim, 2.5, 4.5, **{AXES[ndim - 1]: (6., 6.5)})

    expected = (slice(2, 6),) * (ndim - 1) + (slice(6, 8),)
    assert box_slices(box, coords) == expected


## Box: N dimensions (1 to 4 axes) ------------------------------------------

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
def test_nd_box_contains_points_with_edges_included(ndim):
    points = np.array([[0.5] * ndim,
                       [1.] * ndim,                  # max corner: edges count as inside
                       [0.5] * (ndim - 1) + [2.]])  # outside on the last axis only

    np.testing.assert_array_equal(cube(ndim, 0., 1.).contains(points), [True, True, False])


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_make_grid_has_both_edges_and_ij_indexing(ndim):
    # same bounds (0, 10) on every axis, e.g. ndim=2 -> Box({'A': (0., 10.), 'B': (0., 10.)})
    box = cube(ndim, 0., 10.)

    # 3 points per axis: step = (10 - 0) / (3 - 1) = 5
    axes, grid = box.make_grid({ax: 3 for ax in AXES[:ndim]})

    # axes come back in sorted order A, B, C, D
    assert list(axes) == list(AXES[:ndim])
    for i, ax in enumerate(AXES[:ndim]):
        # both box edges are grid points: 0 (min) and 10 (max)
        np.testing.assert_allclose(axes[ax], [0., 5., 10.])
        # one meshgrid per axis, 3 points along every array dimension: ndim=2 -> (3, 3), ndim=3 -> (3, 3, 3)
        assert grid[ax].shape == (3,) * ndim
        # ij indexing: array dimension i belongs to axis i, so spectrum[i, j] sits at A=axes['A'][i], B=axes['B'][j].
        # ndim=2:  grid['A'] = [[ 0,  0,  0],     grid['B'] = [[0, 5, 10],
        #                       [ 5,  5,  5],                  [0, 5, 10],
        #                       [10, 10, 10]]                  [0, 5, 10]]
        # np.diff (difference between neighbours) along dimension j:
        #   5 along the axis's own dimension (j == i), 0 along all others.
        # numpy's default indexing='xy' swaps the first two dimensions, so this check catches it.
        for j in range(ndim):
            np.testing.assert_allclose(np.diff(grid[ax], axis=j), 5. if j == i else 0.)


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
    f0, f1 = feat(A=0., box_extent=1.), feat(A=2., box_extent=1.)

    assert list(features_to_clusters([f0, f1]).values()) == [[f0, f1]]


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_features_to_clusters_needs_overlap_on_every_axis(ndim):
    # boxes: f0 (-1, 1) and f1 (0.5, 2.5) on every axis; f2 sits on f0, except (9, 11) on the last axis
    f0, f1 = feat(coords(ndim, 0.), box_extent=1.), feat(coords(ndim, 1.5), box_extent=1.)
    f2 = feat(coords(ndim, 0., **{AXES[ndim - 1]: 10.}), box_extent=1.)

    assert list(features_to_clusters([f0, f1, f2]).values()) == [[f0, f1], [f2]]


def test_features_to_clusters_needs_feature_boxes():
    """A feature that was never dressed has no box: clustering it raises."""
    with pytest.raises(ValueError, match='feature box'):
        features_to_clusters([feat(A=0., box_extent=1.), feat(A=1.)])


def test_features_to_clusters_of_no_features_is_empty():
    assert features_to_clusters([]) == {}

