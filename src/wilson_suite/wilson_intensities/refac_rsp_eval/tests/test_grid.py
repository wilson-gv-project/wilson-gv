"""
grid.py — bounds, boxes, grids and box slices. Runs with zero molecular data.
"""

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
    box_slices,
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


@pytest.mark.parametrize('method', ['overlaps', 'contains_box'])
@pytest.mark.parametrize('other', [Box({'B': (0., 1.)}), Box({'A': (0., 1.), 'B': (0., 1.)})])
def test_box_comparison_needs_the_same_axes(method, other):
    with pytest.raises(ValueError, match='do not match'):
        getattr(Box({'A': (0., 1.)}), method)(other)


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


def test_make_grid_gives_one_point_on_a_zero_width_axis():
    axes, grid = Box({'A': (0., 10.), 'B': (5., 5.)}).make_grid({'A': 3, 'B': 1})

    np.testing.assert_array_equal(axes['B'], [5.])
    assert grid['A'].shape == (3, 1)


@pytest.mark.parametrize('dim_sizes', [
    {'A': 1, 'B': 1},  # 1 point on an axis with min < max: only min
    {'A': 3, 'B': 0},  # no points
])
def test_make_grid_rejects_too_few_points(dim_sizes):
    with pytest.raises(ValueError, match='needs 2 or more points'):
        Box({'A': (0., 10.), 'B': (5., 5.)}).make_grid(dim_sizes)


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


def test_box_slices_needs_points_on_every_grid_axis():
    with pytest.raises(ValueError, match='have no points'):
        box_slices(Box({'A': (2., 4.)}), {'A': np.array([])})


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
