"""
Pure geometry: bounds, boxes and box clustering. No physics, no SpectralFeature logic.

  in:  (min, max) bounds per named axis
  out: Box, regular grid over a Box, the part of a grid that covers a Box (slices), box adjacency matrix, clusters of boxes

  holds: Box, compute_box_adjacency, connected_components_from_adjacency, box_slices

features.py imports from this module, never the other way around.
"""
import numbers
from dataclasses import dataclass

import numpy as np

# type aliases
Min_bound = float
Max_bound = float
Dim_bounds = tuple[Min_bound, Max_bound]

@dataclass
class Box:
    """
    N-dimensional rectangle with named axes: {axis: (min, max)}. Axes are sorted by name.

    not sure about the grid yet

    could have different grids for same box?
    is a box a property of grid? 
    box would be slightly smaller than grid, so the grid includes the whole box for sure
    """
    bounds: dict[str, Dim_bounds]

    def __post_init__(self):
        """
        Check and sort the bounds, given as dict[str, (min, max)], e.g. {'B': (5.0, 10.0), 'A': (0.0, 1.0)}.
        Sets self.axes (sorted axis names) and self.ndim.
        """
        # --- Normalize input ---
        # Ensure all values are 2-tuples of numbers
        for key, val in self.bounds.items():
            if not (isinstance(val, tuple) and len(val) == 2):
                raise ValueError(
                    f"Invalid bound for '{key}': expected (min, max), got {val!r}"
                )
        # Sort axes and bounds by key
        sorted_items = sorted(self.bounds.items(), key=lambda kv: kv[0])
        self.axes = tuple(k for k, _ in sorted_items)
        self.bounds = dict(sorted_items)


        # --- Validate numeric consistency ---
        for ax, (mn, mx) in self.bounds.items():

            # numbers.Real also accepts numpy numbers like np.int64 and np.float32
            if not all(isinstance(v, numbers.Real) for v in (mn, mx)):
                raise TypeError(
                    f"Invalid values for bound {ax}: expected numeric (min, max), got ({mn!r}, {mx!r})"
                )
            if mn > mx:
                raise ValueError(
                    f"Invalid bound {ax}: min ({mn}) > max ({mx})"
                )
        # store plain floats, also for int and numpy inputs
        self.bounds = {ax: (float(mn), float(mx)) for ax, (mn, mx) in self.bounds.items()}
        self.ndim = len(self.bounds)


    def __hash__(self):
        return hash(tuple(zip(self.bounds.items()))) # type: ignore

    # -------------------------------------------------
    # Box modifications
    # -------------------------------------------------
    # UNUSED - useful for analysis or for future?
    def expand(self, padding: dict[str, float], inplace: bool = False):
        new_bounds = {}
        for axis in self.axes:
            mn, mx = self.bounds[axis]
            pad = padding.get(axis, 0.0)
            new_bounds[axis] = (mn - pad, mx + pad)

        if inplace:
            self.bounds = new_bounds
            return self
        return Box(new_bounds)

    # -------------------------------------------------
    # Box operations
    # -------------------------------------------------
    # UNUSED - useful for analysis or for future?
    def intersect(self, other: "Box") -> "Box | None":
        common_axes = set(self.axes) & set(other.axes)
        if not common_axes:
            return None  # No shared dimensions

        overlap_bounds = {}
        for ax in common_axes:
            mn1, mx1 = self.bounds[ax]
            mn2, mx2 = other.bounds[ax]
            lower, upper = max(mn1, mn2), min(mx1, mx2)
            if lower >= upper:
                return None  # no overlap
            overlap_bounds[ax] = (lower, upper)
        return Box(overlap_bounds)

    @classmethod
    def union(cls, boxes: list["Box"]) -> "Box":
        """
        makes a union over all provided boxes, 
            even if they aren't connected and don't have same axes
        """
        all_axes = set().union(*(b.axes for b in boxes))
        union_bounds = {}
        for ax in all_axes:
            mins, maxs = [], []
            for b in boxes:
                if ax in b.bounds:
                    mn, mx = b.bounds[ax]
                    mins.append(mn)
                    maxs.append(mx)
            union_bounds[ax] = (min(mins), max(maxs))
        return cls(union_bounds)


    def contains_box(self, other: "Box") -> bool:
        shared_axes = set(self.axes) & set(other.axes)
        return all(
            self.bounds[ax][0] <= other.bounds[ax][0]
            and self.bounds[ax][1] >= other.bounds[ax][1]
            for ax in shared_axes
        )

    # UNUSED - useful for analysis or for future?
    def overlaps(self, other: "Box") -> bool:
        shared_axes = set(self.axes) & set(other.axes)
        return all(
            not (self.bounds[ax][1] <= other.bounds[ax][0] or other.bounds[ax][1] <= self.bounds[ax][0])
            for ax in shared_axes
        )

    # UNUSED
    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Return boolean mask of which points lie inside the box, edges included (same rule as SpectralFeature.is_inside).
        Point coordinates follow the sorted axis order self.axes.
        """
        if points.shape[-1] != self.ndim:
            raise ValueError(f"Expected points with {self.ndim} coords, got {points.shape[-1]}")
        inside = np.ones(points.shape[:-1], dtype=bool)
        for i, (mn, mx) in enumerate(self.bounds.values()):
            inside &= (points[..., i] >= mn) & (points[..., i] <= mx)
        return inside

    def make_grid(self, dim_sizes: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Generate a regular grid of points spanning the box.
        Each axis gets dim_sizes[ax] points from min to max, both edges included: step = (max - min) / (n - 1).
        Returns (1D coords per axis, meshgrid per axis); meshgrids use ij indexing, array dimension i = self.axes[i].
        """

        if set(dim_sizes) != set(self.axes):
            raise ValueError(f"Expected one grid size per box axis {self.axes}, got sizes for {tuple(sorted(dim_sizes))}")
        axes = {}
        for ax in self.bounds:
            mn, mx = self.bounds[ax]
            axes[ax] = np.linspace(mn, mx, dim_sizes[ax])

        coords_vectors = list(axes.values())
        grid = np.meshgrid(*coords_vectors, indexing="ij")
        grid_d = {ax: grid[i] for i, ax in enumerate(axes)}
        
        return axes, grid_d


def compute_box_adjacency(
                            boxes: list["Box"],
                            *,
                            touch_inclusive: bool = True,
                            axis_order: tuple[str, ...] | None = None,
                        ) -> np.ndarray:
    """
    Minimal adjacency for axis-labeled Boxes.
    Assumes all boxes are comparable on the given axis_order.
    No axis compatibility checks are performed here.
    Parameters:
      boxes: list[Box]
      touch_inclusive: True => touching counts as adjacent (>=); False => strict overlap only (>)
      axis_order: optional explicit axis order to use. If None, uses boxes[0].axes.
    Returns:
      n x n boolean adjacency matrix
    """
    n = len(boxes)
    adj = np.zeros((n, n), dtype=bool)
    if n == 0:
        return adj
    
    axes = axis_order if axis_order is not None else boxes[0].axes
    
    for i in range(n):
        bi = boxes[i]
        for j in range(i + 1, n):
            bj = boxes[j]
            overlap = True
            for ax in axes:
                mn_i, mx_i = bi.bounds[ax]
                mn_j, mx_j = bj.bounds[ax]
                if touch_inclusive:
                    if not (mx_i >= mn_j and mx_j >= mn_i):
                        overlap = False
                        break
                else:
                    if not (mx_i > mn_j and mx_j > mn_i):
                        overlap = False
                        break
            if overlap:
                adj[i, j] = adj[j, i] = True
    return adj

def connected_components_from_adjacency(adjacency: np.ndarray, box_objects: list) -> dict[int, list]:
    """
    Generic DFS-based connected component finder.
    adjacency: n x n boolean matrix
    objects: list of objects corresponding to rows of adjacency
    Returns: dict[label] -> list of objects
    """
    n = len(box_objects)
    visited = np.zeros(n, dtype=bool)
    clusters = {}
    label_counter = 0

    for i in range(n):
        if not visited[i]:
            stack = [i]
            members = []
            while stack:
                k = stack.pop()
                if visited[k]:
                    continue
                visited[k] = True
                members.append(box_objects[k])
                neighbors = np.where(adjacency[k])[0]
                stack.extend(neighbors)
            clusters[label_counter] = members
            label_counter += 1
    return clusters



def box_slices(box: Box, coords: dict[str, np.ndarray]) -> tuple[slice, ...] | None:
    """
    Where `box` sits in a grid: one slice per axis, in the order of `coords` (the grid's array dimensions).
    The smallest index range that covers the box: the last grid point at or below min to the first at or above
    max, so the cut-out grid holds the whole box. Cut at the grid edges. None if the box lies outside the grid.

    coords - one ascending 1D array per axis, e.g. the first output of Box.make_grid; same axes as the box.

    e.g. grid points A = 0, 1, ..., 9:
        box (2.5, 4.5)  ->  slice(2, 6)   points 2, 3, 4, 5
        box (3., 4.)    ->  slice(3, 5)   points 3, 4: edges on grid points, nothing extra
        box (-5., 0.5)  ->  slice(0, 2)   cut at the grid edge
        box (9., 12.)   ->  slice(9, 10)  touches the grid edge: point 9 lies in the box
        box (12., 15.)  ->  None
    """
    if set(coords) != set(box.axes):
        raise ValueError(f'box axes {box.axes} do not match grid axes {tuple(coords)}')

    slices = []
    for ax, points in coords.items():
        mn, mx = box.bounds[ax]
        if mx < points[0] or mn > points[-1]:
            return None
        i_min = max(int(np.searchsorted(points, mn, side='right')) - 1, 0)
        i_max = min(int(np.searchsorted(points, mx, side='left')), len(points) - 1)
        slices.append(slice(i_min, i_max + 1))
    return tuple(slices)

