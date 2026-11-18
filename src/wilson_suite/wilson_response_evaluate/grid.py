"""
Pure geometry: bounds, boxes and grids. No physics, no SpectralFeature logic.

  in:  (min, max) bounds per named axis
  out: Box, the points of a regular grid over a Box (AxisPoints), every grid point (Mesh),
       where a Box sits in a grid, as index ranges (box_slices)

  holds: Box, make_mesh, box_slices

The three forms of a grid's extent, e.g. A from 0 to 10 with 3 points, B from -1 to 1 with 2:
    Box.bounds     the range of each axis, 2 numbers   {'A': (0., 10.), 'B': (-1., 1.)}
    AxisPoints     the points along each axis          {'A': [0., 5., 10.], 'B': [-1., 1.]}
    Mesh           every grid point, 3 x 2 of them     {'A': [[0., 0.], [5., 5.], [10., 10.]],
                                                        'B': [[-1., 1.], [-1., 1.], [-1., 1.]]}

Imports nothing from the rest of the package.
"""
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# type aliases
MinBound = float
MaxBound = float
DimBounds = tuple[MinBound, MaxBound]
# the points along each axis, ascending: the grid is every combination of them (Box.grid_points makes them)
AxisPoints = dict[str, np.ndarray]
# every grid point, one full array per axis, ij indexing: mesh['A'][i, j] = axis_points['A'][i] (make_mesh)
Mesh = dict[str, np.ndarray]


@dataclass
class Box:
    """
    N-dimensional rectangle with named axes: {axis: (min, max)}. Axes are sorted by name.
    """
    bounds: dict[str, DimBounds]

    def __post_init__(self):
        """
        Check and sort the bounds, given as dict[str, (min, max)],
        e.g. {'B': (5.0, 10.0), 'A': (0.0, 1.0)}.
        Sets self.axes (sorted axis names).
        """
        # --- Check the shape, sort by axis ---
        # Each value must be a (min, max) pair
        for key, val in self.bounds.items():
            if not (isinstance(val, tuple) and len(val) == 2):
                raise ValueError(
                    f"Invalid bound for '{key}': expected (min, max), got {val!r}"
                )
        # Sort axes and bounds by key
        sorted_items = sorted(self.bounds.items(), key=lambda kv: kv[0])
        self.axes = tuple(k for k, _ in sorted_items)
        self.bounds = dict(sorted_items)

        # --- Check the numbers, store floats ---
        for ax, (mn, mx) in self.bounds.items():
            # numbers.Real also accepts numpy numbers like np.int64 and np.float32
            if not all(isinstance(v, numbers.Real) for v in (mn, mx)):
                raise TypeError(
                    f"Invalid values for bound {ax}: expected numeric (min, max), "
                    f"got ({mn!r}, {mx!r})"
                )
            if math.isnan(mn) or math.isnan(mx):
                raise ValueError(f"Invalid bound {ax}: NaN in ({mn}, {mx})")
            if mn > mx:
                raise ValueError(
                    f"Invalid bound {ax}: min ({mn}) > max ({mx})"
                )
        # store plain floats, also for int and numpy inputs
        self.bounds = {ax: (float(mn), float(mx)) for ax, (mn, mx) in self.bounds.items()}

    def __hash__(self):
        return hash(tuple(self.bounds.items()))

    def _check_same_axes(self, other: "Box"):
        if self.axes != other.axes:
            raise ValueError(f"box axes {self.axes} do not match {other.axes}")

    def contains_point(self, point: Mapping[str, float]) -> bool:
        """
        True if `point` lies inside this box, edges included.
        point: one coordinate per box axis, e.g. {'A': 0.5, 'B': 2.}.
        e.g. A (0, 1) contains A = 0, 0.5 and 1, not 1.5.
        """
        if tuple(sorted(point)) != self.axes:
            raise ValueError(f"point axes {tuple(sorted(point))} do not match box axes {self.axes}")
        return all(mn <= point[ax] <= mx for ax, (mn, mx) in self.bounds.items())

    def contains_box(self, other: "Box") -> bool:
        """
        True if `other` lies inside this box, edges included.
        Both boxes need the same axes. e.g. A (0, 10) contains A (2, 3) and A (0, 10),
        not A (9, 11).
        """
        self._check_same_axes(other)
        return all(
            self.bounds[ax][0] <= other.bounds[ax][0]
            and self.bounds[ax][1] >= other.bounds[ax][1]
            for ax in self.axes
        )

    def overlaps(self, other: "Box") -> bool:
        """
        True if the boxes have at least one point in common: on every axis their ranges meet, edges included.
        So boxes that only touch overlap, on one axis (a shared face) or on more (a shared edge or corner).
        Touching on one axis but apart on another: no point in common, no overlap.
        Both boxes need the same axes. e.g. A (0, 1) overlaps A (0.5, 2), A (1, 2) (one point in common) and
        A (1, 1), not A (1.5, 2). 2D, A (0, 1) B (0, 1) overlaps A (1, 2) B (0.5, 2) (a shared edge) and
        A (1, 2) B (1, 2) (the corner (1, 1)), not A (1, 2) B (1.5, 2) (touches on A, apart on B).
        """
        self._check_same_axes(other)
        return all(
            self.bounds[ax][0] <= other.bounds[ax][1] and other.bounds[ax][0] <= self.bounds[ax][1]
            for ax in self.axes
        )

    def grid_points(self, n_points_per_axis: dict[str, int]) -> AxisPoints:
        """
        The points of a regular grid over the box, along each axis (AxisPoints).
        Each axis gets n = n_points_per_axis[ax] points from min to max, both edges included:
        step = (max - min) / (n - 1). So n >= 2, except on a zero-width axis
        (min = max): n = 1.
        e.g. A (0, 10) with 3 points, B (5, 5) with 1: {'A': [0., 5., 10.], 'B': [5.]}
        """
        if set(n_points_per_axis) != set(self.axes):
            raise ValueError(f"n_points_per_axis needs one entry per box axis {self.axes}, "
                             f"got entries for {tuple(sorted(n_points_per_axis))}")
        axis_points = {}
        for ax, (mn, mx) in self.bounds.items():
            n = n_points_per_axis[ax]
            if n < 1 or (n == 1 and mn < mx):
                raise ValueError(f"{n} points for {ax} on ({mn}, {mx}): "
                                 f"needs 2 or more points, or 1 with min = max")
            axis_points[ax] = np.linspace(mn, mx, n)
        return axis_points


def make_mesh(axis_points: AxisPoints) -> Mesh:
    """
    Every grid point, one full array per axis (Mesh), ij indexing: array dimension k is the
    k-th axis of axis_points, e.g. A with 3 points, B with 2: mesh['A'] and mesh['B'] of
    shape (3, 2), mesh['A'][i, j] = axis_points['A'][i], mesh['B'][i, j] = axis_points['B'][j].
    Takes one float per grid point per axis, 
    e.g. 8 bytes (float64) per grid point per axis, e.g. 201^3 points x 8 B = 65 MB per axis.
    """
    return dict(zip(axis_points, np.meshgrid(*axis_points.values(), indexing="ij")))


def box_slices(box: Box, axis_points: AxisPoints) -> tuple[slice, ...] | None:
    """
    Where a box sits in the grid, as index ranges: one slice per grid axis.
    It only finds the grid points: no array is made or copied here. The caller indexes with
    the slices, e.g. boxed_features_on_grid (evaluate.py): amplitude[slices] += the feature
    on those points.

    The slices come in the order of axis_points, the order of the grid array's dimensions (not
    the box's sorted axes), so they index the grid directly:
        array[slices]                  the values at those grid points, e.g. amplitude[slices] += ...
        axis_points['A'][slices[0]]    those points along the grid's first axis, here A

    The rule, on each axis on its own: from the last grid point at or below the box's min to
    the first at or above its max.
      - A box edge between two grid points: the range takes the point beyond it, so the
        points cover the whole box.
      - A box edge on a grid point: nothing more.
      - A box that sticks out of the grid: the range stops at the grid's end. A box that only
        touches the grid's end still gets that one point.
    None if the box misses the grid on any one axis.

    axis_points - the grid (AxisPoints), e.g. window_box.grid_points(n_points_per_axis): one
                  ascending, non-empty 1D array per axis; the same axes as the box.

    e.g. one axis, grid points A = 0, 1, ..., 9:
        box (2.5, 4.5)  ->  slice(2, 6)   points 2, 3, 4, 5: one more on each side
        box (3., 4.)    ->  slice(3, 5)   points 3, 4: edges on grid points
        box (-5., 0.5)  ->  slice(0, 2)   points 0, 1: stops at the grid's end
        box (9., 12.)   ->  slice(9, 10)  point 9: the box touches the grid's end there
        box (12., 15.)  ->  None          misses the grid
    Two axes, each by the same rule: grid A = 0, 1, ..., 9 and B = 0, 5, ..., 45:
        box A (2.5, 4.5), B (12., 18.)  ->  (slice(2, 6), slice(2, 5))   B points 10, 15, 20
        box A (2.5, 4.5), B (50., 60.)  ->  None                         misses on B: enough
    """
    if set(axis_points) != set(box.axes):
        raise ValueError(f'box axes {box.axes} do not match grid axes {tuple(axis_points)}')
    empty = [ax for ax, points in axis_points.items() if len(points) == 0]
    if empty:
        raise ValueError(f'grid axes {empty} have no points')

    slices = []
    for ax, points in axis_points.items():
        mn, mx = box.bounds[ax]
        if mx < points[0] or mn > points[-1]:
            return None
        i_min = max(int(np.searchsorted(points, mn, side='right')) - 1, 0)
        i_max = min(int(np.searchsorted(points, mx, side='left')), len(points) - 1)
        slices.append(slice(i_min, i_max + 1))
    return tuple(slices)
