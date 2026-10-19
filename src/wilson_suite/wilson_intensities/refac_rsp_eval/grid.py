"""
Pure geometry: bounds, boxes and grids. No physics, no SpectralFeature logic.

  in:  (min, max) bounds per named axis
  out: Box, regular grid over a Box, the part of a grid that covers a Box (slices)

  holds: Box, box_slices

Imports nothing from the rest of the package.
"""
import math
import numbers
from dataclasses import dataclass

import numpy as np

# type aliases
MinBound = float
MaxBound = float
DimBounds = tuple[MinBound, MaxBound]


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
        Sets self.axes (sorted axis names) and self.ndim.
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
        self.ndim = len(self.bounds)

    def __hash__(self):
        return hash(tuple(self.bounds.items()))

    # -------------------------------------------------
    # Box operations
    # -------------------------------------------------
    def _check_same_axes(self, other: "Box"):
        if self.axes != other.axes:
            raise ValueError(f"box axes {self.axes} do not match {other.axes}")

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
        True if the boxes overlap on every axis; boxes that only touch count as apart.
        Both boxes need the same axes. e.g. A (0, 1) overlaps A (0.5, 2), not A (1, 2).
        """
        self._check_same_axes(other)
        return all(
            not (self.bounds[ax][1] <= other.bounds[ax][0]
                 or other.bounds[ax][1] <= self.bounds[ax][0])
            for ax in self.axes
        )

    def make_grid(
        self, dim_sizes: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Generate a regular grid of points spanning the box.
        Each axis gets n = dim_sizes[ax] points from min to max, both edges included:
        step = (max - min) / (n - 1). So n >= 2, except on a zero-width axis
        (min = max): n = 1.
        Returns (1D coords per axis, meshgrid per axis); meshgrids use ij indexing,
        array dimension i = self.axes[i].
        """
        if set(dim_sizes) != set(self.axes):
            raise ValueError(f"Expected one grid size per box axis {self.axes}, "
                             f"got sizes for {tuple(sorted(dim_sizes))}")
        coords = {}
        for ax, (mn, mx) in self.bounds.items():
            n = dim_sizes[ax]
            if n < 1 or (n == 1 and mn < mx):
                raise ValueError(f"Grid size {n} for {ax} on ({mn}, {mx}): "
                                 f"needs 2 or more points, or 1 with min = max")
            coords[ax] = np.linspace(mn, mx, n)

        mesh = np.meshgrid(*coords.values(), indexing="ij")
        return coords, dict(zip(coords, mesh))


def box_slices(box: Box, coords: dict[str, np.ndarray]) -> tuple[slice, ...] | None:
    """
    Where `box` sits in a grid: one slice per axis, in the order of `coords`
    (the grid's array dimensions). The smallest index range that covers the box:
    the last grid point at or below min to the first at or above max, so the cut-out
    grid holds the whole box. Cut at the grid edges.
    None if the box lies outside the grid.

    coords - one ascending, non-empty 1D array per axis, e.g. the first output of
             Box.make_grid; same axes as the box.

    e.g. grid points A = 0, 1, ..., 9:
        box (2.5, 4.5)  ->  slice(2, 6)   points 2, 3, 4, 5
        box (3., 4.)    ->  slice(3, 5)   points 3, 4: edges on grid points
        box (-5., 0.5)  ->  slice(0, 2)   cut at the grid edge
        box (9., 12.)   ->  slice(9, 10)  touches the grid edge: point 9 lies in the box
        box (12., 15.)  ->  None
    """
    if set(coords) != set(box.axes):
        raise ValueError(f'box axes {box.axes} do not match grid axes {tuple(coords)}')
    empty = [ax for ax, points in coords.items() if len(points) == 0]
    if empty:
        raise ValueError(f'grid axes {empty} have no points')

    slices = []
    for ax, points in coords.items():
        mn, mx = box.bounds[ax]
        if mx < points[0] or mn > points[-1]:
            return None
        i_min = max(int(np.searchsorted(points, mn, side='right')) - 1, 0)
        i_max = min(int(np.searchsorted(points, mx, side='left')), len(points) - 1)
        slices.append(slice(i_min, i_max + 1))
    return tuple(slices)
