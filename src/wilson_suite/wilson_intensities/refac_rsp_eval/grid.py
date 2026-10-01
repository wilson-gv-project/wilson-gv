"""

    in: features, window, resolution 
    out: spectrum array 
    notes:full-grid Lorentzians
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.refac_rsp_eval.features import SpectralFeature

def points_to_bounds(points: list[dict[str,float]], 
                     halfwidth: float) -> list[dict[str,tuple[float,float]]]:
    # FIXME should know combination of states to get the Gamma; now it's using a single value everywhere
    # halfwidth - doesn't have to be simply Gamma, shouldn't...

    return [
        {axis: (p[axis]-halfwidth, p[axis]+halfwidth) for axis in p}
        for p in points
    ]


# type aliases
Min_bound = float
Max_bound = float
Dim_bounds = tuple[Min_bound, Max_bound]

@dataclass
class Box:
    """
    not sure about the grid yet

    could have different grids for same box?
    is a box a property of grid? 
    box would be slightly smaller than grid, so the grid includes the whole box for sure
    """
    bounds: dict[str, Dim_bounds]

    def __post_init__(self):
        """
        Create a Box either from:
            - dict[str, (min, max)], e.g. {'A': (0.0, 1.0), 'B': (5.0, 10.0)}
            - tuple of (min, max) pairs, e.g. ((0.0, 1.0), (5.0, 10.0))
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

            if not all(isinstance(v, (int, float)) for v in (mn, mx)):
                raise TypeError(
                    f"Invalid values for bound {ax}: expected numeric (min, max), got ({mn!r}, {mx!r})"
                )
            if mn > mx:
                raise ValueError(
                    f"Invalid bound {ax}: min ({mn}) > max ({mx})"
                )
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
        """Return boolean mask of which points lie inside the window."""
        if points.shape[-1] != self.ndim:
            raise ValueError(f"Expected points with {self.ndim} coords, got {points.shape[-1]}")
        inside = np.ones(points.shape[:-1], dtype=bool)
        for i, (mn, mx) in enumerate(self.bounds.values()):
            inside &= (points[..., i] >= mn) & (points[..., i] < mx)
        return inside

    # ----------------------------------------------
    # relations to SpectralFeature
    # ----------------------------------------------
    def contains_feature(self, spec_feature: 'SpectralFeature', mode='loc') -> bool:
        """
        Return boolean for whether SpectralFeature lies inside the window.

        NEW: feature.is_inside(box)
        """
        if mode=='box':
            if spec_feature.feat_box is not None:
                return self.overlaps(spec_feature.feat_box)
                # return self.contains_box(spec_feature.feat_box)
            raise ValueError('Need to add a box for this feature')
        if mode=='loc':
            spec_feature_ndim = len(spec_feature.location.values)
            if spec_feature_ndim != self.ndim:
                raise ValueError(f"Expected SpectralFeature with a location with {self.ndim} coords, got {spec_feature_ndim}")

            inside = True
            for ax, (mn, mx) in self.bounds.items():
                inside &= (spec_feature.location._coord_dict[ax] >= mn) & (spec_feature.location._coord_dict[ax] <= mx)
            return inside
        raise ValueError('Supported modes of check: `loc`, `box`')

    def contributing_feature(self, spec_feature: 'SpectralFeature') -> bool:
        """
        Return boolean for whether SpectralFeature is contributing to this window, 
            based on lineshape_parameter of this SpectralFeature

        NEW: feature.contributes_to(box)
        """
        spec_feature_ndim = len(spec_feature.location.coordinates)
        if spec_feature_ndim != self.ndim:
            raise ValueError(f"Expected SpectralFeature with a location with {self.ndim} coords, got {spec_feature_ndim}")

        if spec_feature.lineshape_parameter is None:
            raise ValueError("Expected SpectralFeature with `lineshape_parameter` attribute")

        contributing = True
        for ax, (mn, mx) in self.bounds.items():
            Gamma = spec_feature.lineshape_parameter
            # FIXME??   2*Gamma ??
            # in place ADDition
            contributing &= (spec_feature.location._coord_dict[ax] >= mn-2*Gamma) & (spec_feature.location._coord_dict[ax] <= mx+2*Gamma)
        return contributing and not self.contains_feature(spec_feature)


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

