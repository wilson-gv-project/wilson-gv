"""
grid.py — bounds, boxes, grids and box slices. Runs with zero molecular data.
"""

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
    box_slices,
    make_mesh,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    AXES,
    NDIMS,
    cube,
)

## Box: construction --------------------------------------------------------

def test_box_sorts_axes_by_name():
    box = Box({'B': (0., 1.), 'A': (2., 3.)})

    assert box.axes == ('A', 'B')
    assert list(box.bounds) == ['A', 'B']


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


@pytest.mark.parametrize('bounds', [(np.nan, 1.), (0., np.nan)])
def test_box_rejects_nan_bounds(bounds):
    with pytest.raises(ValueError, match='NaN'):
        Box({'A': bounds})


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

def test_box_contains_point_includes_the_edges():
    box = Box({'A': (0., 1.), 'B': (5., 6.)})

    assert box.contains_point({'A': 0.5, 'B': 5.5})
    assert box.contains_point({'A': 0., 'B': 6.})
    assert not box.contains_point({'A': 1.5, 'B': 5.5})
    assert not box.contains_point({'A': 0.5, 'B': 4.})


@pytest.mark.parametrize('point', [{'B': 0.5}, {'A': 0.5, 'B': 0.5}])
def test_box_contains_point_needs_the_same_axes(point):
    with pytest.raises(ValueError, match='do not match'):
        Box({'A': (0., 1.)}).contains_point(point)


def test_box_contains_box():
    outer = Box({'A': (0., 10.)})

    assert outer.contains_box(Box({'A': (2., 3.)}))
    assert outer.contains_box(outer)
    assert not outer.contains_box(Box({'A': (9., 11.)}))


def test_box_overlaps_includes_touching_edges():
    box = Box({'A': (0., 1.)})

    assert box.overlaps(Box({'A': (0.5, 2.)}))
    assert box.overlaps(Box({'A': (1., 2.)}))  # one point in common
    assert box.overlaps(Box({'A': (1., 1.)}))  # a zero-width box on the edge
    assert Box({'A': (1., 1.)}).overlaps(Box({'A': (1., 1.)}))
    assert not box.overlaps(Box({'A': (1.5, 2.)}))
    assert not box.overlaps(Box({'A': (3., 4.)}))


@pytest.mark.parametrize('method', ['overlaps', 'contains_box'])
@pytest.mark.parametrize('other', [Box({'B': (0., 1.)}), Box({'A': (0., 1.), 'B': (0., 1.)})])
def test_box_comparison_needs_the_same_axes(method, other):
    with pytest.raises(ValueError, match='do not match'):
        getattr(Box({'A': (0., 1.)}), method)(other)


## Box: grid points, make_mesh ----------------------------------------------

def test_grid_points_gives_the_points_along_each_axis_only():
    box = Box({'A': (0., 10.), 'B': (100., 200.)})

    axis_points = box.grid_points({'A': 5, 'B': 3})

    assert list(axis_points) == ['A', 'B']
    assert (axis_points['A'].shape, axis_points['B'].shape) == ((5,), (3,))


def test_grid_points_runs_from_min_to_max_with_both_edges():
    box = Box({'A': (0., 10.), 'B': (100., 200.)})

    axis_points = box.grid_points({'A': 11, 'B': 5})

    np.testing.assert_allclose(axis_points['A'], [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    np.testing.assert_allclose(axis_points['B'], [100., 125., 150., 175., 200.])


@pytest.mark.parametrize('n_points_per_axis', [
    {'A': 5, 'B': 5},  # one axis too many
    {'B': 5},          # same number of axes, other name
])
def test_grid_points_needs_one_entry_per_box_axis(n_points_per_axis):
    with pytest.raises(ValueError, match='one entry per box axis'):
        Box({'A': (0., 10.)}).grid_points(n_points_per_axis)


def test_grid_points_gives_one_point_on_a_zero_width_axis():
    axis_points = Box({'A': (0., 10.), 'B': (5., 5.)}).grid_points({'A': 3, 'B': 1})

    np.testing.assert_array_equal(axis_points['B'], [5.])
    assert make_mesh(axis_points)['A'].shape == (3, 1)


@pytest.mark.parametrize('n_points_per_axis', [
    {'A': 1, 'B': 1},  # 1 point on an axis with min < max: only min
    {'A': 3, 'B': 0},  # no points
])
def test_grid_points_rejects_too_few_points(n_points_per_axis):
    with pytest.raises(ValueError, match='needs 2 or more points'):
        Box({'A': (0., 10.), 'B': (5., 5.)}).grid_points(n_points_per_axis)


def test_make_mesh_shapes_and_ij_indexing():
    axis_points = {'A': np.linspace(0., 10., 5), 'B': np.linspace(100., 200., 3)}

    mesh = make_mesh(axis_points)

    assert list(mesh) == ['A', 'B']
    assert mesh['A'].shape == mesh['B'].shape == (5, 3)
    np.testing.assert_array_equal(mesh['A'][:, 0], axis_points['A'])
    np.testing.assert_array_equal(mesh['B'][0, :], axis_points['B'])


## box_slices (1 to 4 axes) -------------------------------------------------
# Grid points 0, 1, ..., 9 on every axis, unless a test says otherwise.

# (box bounds on one axis, the cut on that axis)
CUT_CASES = [
    ((2.5, 4.5), slice(2, 6)),    # one more point on each side: 2 and 5
    ((3., 4.), slice(3, 5)),      # edges on grid points: nothing extra
    ((3., 3.), slice(3, 4)),      # zero width
    ((-5., 0.5), slice(0, 2)),    # cut at the low grid edge
    ((8.5, 20.), slice(8, 10)),   # cut at the high grid edge
    ((9., 12.), slice(9, 10)),    # touches the high grid edge: point 9 lies in the box
    ((-3., 0.), slice(0, 1)),     # touches the low grid edge: point 0 lies in the box
    ((-5., 15.), slice(0, 10)),   # covers the whole axis
]
# box bounds on one axis that miss the grid 0, 1, ..., 9
OUTSIDE = [(12., 15.), (9.5, 12.), (-5., -0.5)]
# a different grid on each axis, steps 1, 5, 0.5 and uneven: (points, box bounds, the cut)
OWN_POINTS = {
    'A': (np.arange(10.), (2.5, 4.5), slice(2, 6)),                 # 0, 1, ..., 9: points 2 to 5
    'B': (100. + 5. * np.arange(20), (112., 128.), slice(2, 7)),   # 100, 105, ..., 195: 110 to 130
    'C': (np.linspace(-1., 1., 5), (-0.2, 0.2), slice(1, 4)),      # -1, -0.5, 0, 0.5, 1: -0.5 to 0.5
    'D': (np.array([0., 1., 3., 7., 15.]), (2., 4.), slice(1, 4)),  # uneven: 1 to 7
}


@pytest.mark.parametrize('ndim', NDIMS)
@pytest.mark.parametrize('bounds, expected', CUT_CASES)
def test_nd_box_slices_smallest_range_that_covers_the_box(ndim, bounds, expected):
    """
    Each case on one axis at a time; the box is (2.5, 4.5) -> slice(2, 6) on the other axes.
    e.g. ndim=2, the case (-5., 0.5) on B: box A (2.5, 4.5), B (-5., 0.5) -> (slice(2, 6), slice(0, 2))
    """
    axis_points = {ax: np.arange(10.) for ax in AXES[:ndim]}

    for k, ax in enumerate(AXES[:ndim]):
        cut = [slice(2, 6)] * ndim
        cut[k] = expected
        assert box_slices(cube(ndim, 2.5, 4.5, **{ax: bounds}), axis_points) == tuple(cut), f'the case on {ax}'


@pytest.mark.parametrize('ndim', NDIMS)
@pytest.mark.parametrize('bounds', OUTSIDE)
def test_nd_box_slices_is_none_when_outside_on_any_one_axis(ndim, bounds):
    """Outside the grid on one axis, inside on all the others: no cut."""
    axis_points = {ax: np.arange(10.) for ax in AXES[:ndim]}

    for ax in AXES[:ndim]:
        assert box_slices(cube(ndim, 2.5, 4.5, **{ax: bounds}), axis_points) is None, f'outside on {ax}'


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_box_slices_use_each_axis_own_points_in_the_order_of_axis_points(ndim):
    """
    Each axis has its own grid (OWN_POINTS). One slice per array dimension of the grid, so the cut follows the
    order of axis_points, not the box's sorted axes: checked in sorted and in reversed order.
    """
    axes = AXES[:ndim]
    box = Box({ax: OWN_POINTS[ax][1] for ax in axes})

    for order in (axes, axes[::-1]):
        axis_points = {ax: OWN_POINTS[ax][0] for ax in order}
        assert box_slices(box, axis_points) == tuple(OWN_POINTS[ax][2] for ax in order), f'order {order}'


@pytest.mark.parametrize('ndim', NDIMS)
@pytest.mark.parametrize('bounds, expected', [
    ((4., 6.), slice(0, 1)),      # around the one point
    ((5., 5.), slice(0, 1)),      # on it, zero width
    ((4., 5.), slice(0, 1)),      # touches it
    ((5.5, 6.), None),            # misses it
])
def test_nd_box_slices_on_a_grid_with_one_point_on_the_last_axis(ndim, bounds, expected):
    """
    A slice of a spectrum: the window is zero-wide on the last axis, so the grid has one point
    there, 5. The other axes: 0, 1, ..., 9, box (2.5, 4.5).
    """
    last = AXES[ndim - 1]
    axis_points = {ax: np.arange(10.) for ax in AXES[:ndim - 1]} | {last: np.array([5.])}

    cut = box_slices(cube(ndim, 2.5, 4.5, **{last: bounds}), axis_points)

    assert cut == (None if expected is None else (slice(2, 6),) * (ndim - 1) + (expected,))


def cut_by_hand(points: np.ndarray, mn: float, mx: float) -> slice | None:
    """The cut on one axis, found by looking at every point; None if the box misses the axis."""
    if mx < points[0] or mn > points[-1]:
        return None
    lo, hi = max(mn, points[0]), min(mx, points[-1])   # the box, cut at the grid edges
    start = max(i for i, p in enumerate(points) if p <= lo)
    stop = min(i for i, p in enumerate(points) if p >= hi) + 1
    return slice(start, stop)


def random_bounds(rng: np.random.Generator, points: np.ndarray) -> tuple[float, float]:
    """
    (min, max) on one axis. 1 in 10: outside the grid, below or above it. Else drawn from the grid points, the
    midpoints between them and one value outside each end, so edges on grid points and cuts at the grid edges
    come up often.
    """
    if rng.random() < 0.1:
        offsets = rng.uniform(0.1, 3., size=2)
        return tuple(sorted(points[0] - offsets if rng.random() < 0.5 else points[-1] + offsets))
    values = np.concatenate([points, (points[:-1] + points[1:]) / 2., [points[0] - 1., points[-1] + 1.]])
    return tuple(sorted(rng.choice(values, size=2)))


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_box_slices_agree_with_a_point_by_point_search(ndim):
    """
    300 random boxes (random_bounds on each axis) on the grids of OWN_POINTS.
    Expected: cut_by_hand on each axis; None if any axis gives None. Both outcomes must come up.
    """
    rng = np.random.default_rng(0)
    axis_points = {ax: OWN_POINTS[ax][0] for ax in AXES[:ndim]}
    outcomes = set()

    for _ in range(300):
        bounds = {ax: random_bounds(rng, points) for ax, points in axis_points.items()}
        by_hand = [cut_by_hand(axis_points[ax], *bounds[ax]) for ax in axis_points]
        expected = None if None in by_hand else tuple(by_hand)
        assert box_slices(Box(bounds), axis_points) == expected, bounds
        outcomes.add(expected is None)

    assert outcomes == {True, False}


def test_box_slices_needs_points_on_every_grid_axis():
    with pytest.raises(ValueError, match='have no points'):
        box_slices(Box({'A': (2., 4.)}), {'A': np.array([])})


@pytest.mark.parametrize('axis_points', [{'A': np.arange(10.), 'B': np.arange(10.)}, {'B': np.arange(10.)}])
def test_box_slices_needs_the_grid_axes_of_the_box(axis_points):
    with pytest.raises(ValueError, match='do not match grid axes'):
        box_slices(Box({'A': (2., 4.)}), axis_points)


## Box: N dimensions (1 to 4 axes) ------------------------------------------

@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_box_contains_box_needs_every_axis(ndim):
    outer = cube(ndim, 0., 3.)

    assert outer.contains_box(cube(ndim, 1., 2.))
    assert not outer.contains_box(cube(ndim, 1., 2., **{AXES[ndim - 1]: (2., 4.)}))


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_box_overlap_needs_every_axis(ndim):
    box = cube(ndim, 0., 2.)
    apart_on_last_axis = cube(ndim, 1., 3., **{AXES[ndim - 1]: (5., 6.)})

    assert box.overlaps(cube(ndim, 1., 3.))
    assert not box.overlaps(apart_on_last_axis)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_boxes_that_touch_on_some_axes_and_overlap_on_the_rest_overlap(ndim):
    """
    box (0, 2) on every axis. `other` touches it at 2 on the first k axes, (2, 4), and is (1, 3) on the rest:
        k = 1      a shared face (2D: an edge), e.g. ndim=2: A (2, 4), B (1, 3)
        k = ndim   a shared corner, the one point (2, 2, ...)
    They have points in common for every k, so they overlap, both ways round.
    """
    box = cube(ndim, 0., 2.)

    for k in range(1, ndim + 1):
        other = cube(ndim, 1., 3., **{ax: (2., 4.) for ax in AXES[:k]})
        assert box.overlaps(other) and other.overlaps(box), f'touching on {AXES[:k]}'


@pytest.mark.parametrize('ndim', NDIMS[1:])
def test_nd_boxes_that_touch_on_one_axis_but_are_apart_on_another_do_not_overlap(ndim):
    """`other` touches box (0, 2) on A, at 2, but the last axis (5, 6) is apart: no point in common."""
    box = cube(ndim, 0., 2.)
    other = cube(ndim, 1., 3., A=(2., 4.), **{AXES[ndim - 1]: (5., 6.)})

    assert not box.overlaps(other)
    assert not other.overlaps(box)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_zero_width_box_on_the_boundary_overlaps(ndim):
    """A zero-width box is a point (a feature with half-width 0): on a face or the corner it overlaps, just outside not."""
    box = cube(ndim, 0., 2.)

    assert box.overlaps(cube(ndim, 1., 1., A=(2., 2.)))        # on the face A = 2
    assert box.overlaps(cube(ndim, 2., 2.))                    # on the corner (2, 2, ...)
    assert not box.overlaps(cube(ndim, 1., 1., A=(2.5, 2.5)))  # 0.5 outside on A


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_grid_points_has_both_edges_and_make_mesh_ij_indexing(ndim):
    # same bounds (0, 10) on every axis, e.g. ndim=2 -> Box({'A': (0., 10.), 'B': (0., 10.)})
    box = cube(ndim, 0., 10.)

    # 3 points per axis: step = (10 - 0) / (3 - 1) = 5
    axis_points = box.grid_points({ax: 3 for ax in AXES[:ndim]})
    mesh = make_mesh(axis_points)

    # axes come back in sorted order A, B, C, D
    assert list(axis_points) == list(mesh) == list(AXES[:ndim])
    for i, ax in enumerate(AXES[:ndim]):
        # both box edges are grid points: 0 (min) and 10 (max)
        np.testing.assert_allclose(axis_points[ax], [0., 5., 10.])
        # one full array per axis, 3 points along every array dimension: ndim=2 -> (3, 3), ndim=3 -> (3, 3, 3)
        assert mesh[ax].shape == (3,) * ndim
        # ij indexing: array dimension i belongs to axis i, so spectrum[i, j] sits at
        # A = axis_points['A'][i], B = axis_points['B'][j].
        # ndim=2:  mesh['A'] = [[ 0,  0,  0],     mesh['B'] = [[0, 5, 10],
        #                       [ 5,  5,  5],                  [0, 5, 10],
        #                       [10, 10, 10]]                  [0, 5, 10]]
        # np.diff (difference between neighbours) along dimension j:
        #   5 along the axis's own dimension (j == i), 0 along all others.
        # numpy's default indexing='xy' swaps the first two dimensions, so this check catches it.
        for j in range(ndim):
            np.testing.assert_allclose(np.diff(mesh[ax], axis=j), 5. if j == i else 0.)
