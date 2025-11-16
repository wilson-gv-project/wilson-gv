import numpy as np
import copy
from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.domains import points_to_bounds, compute_box_adjacency, connected_components_from_adjacency
from wilson_suite.wilson_intensities.amplitudes.term_parts import ResonanceMotif, TermParametersChoice, is_tuple_of_tuples, safe_arange_inclusive_scaled

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Literal


@dataclass
class Grid:
    """
    """
    meshgrids: Tuple[np.ndarray, ...]
    # ndim: int
    # grid_coords: Tuple

# type aliases
Min_bound = float
Max_bound = float
Dim_bounds = Tuple[Min_bound, Max_bound]

@dataclass
class Box:
    """
    not sure about the grid yet

    could have different grids for same box?
    is a box a property of grid? 
    box would be slightly smaller than grid, so the grid includes the whole box for sure
    """
    bounds: Union[Tuple[Dim_bounds, ...], Dict[str, Dim_bounds]]
    grid: Grid = None
    has_grid: bool = None

    def __init__(self,
                 bounds: Union[Tuple[Dim_bounds, ...], Dict[str, Dim_bounds]],
                 grid: Optional["Grid"] = None):
        """
        Create a Box either from:
            - dict[str, (min, max)], e.g. {'A': (0.0, 1.0), 'B': (5.0, 10.0)}
            - tuple of (min, max) pairs, e.g. ((0.0, 1.0), (5.0, 10.0))
        """
        # --- Normalize input ---
        if isinstance(bounds, dict):
            # Ensure all values are 2-tuples of numbers
            for key, val in bounds.items():
                if not (isinstance(val, tuple) and len(val) == 2):
                    raise ValueError(
                        f"Invalid bound for '{key}': expected (min, max), got {val!r}"
                    )
            # Sort axes and bounds by key
            sorted_items = sorted(bounds.items(), key=lambda kv: kv[0])
            self.axes = tuple(k for k, _ in sorted_items)
            self.bounds = dict(sorted_items)

        elif isinstance(bounds, tuple):
            if not all(isinstance(b, tuple) and len(b) == 2 for b in bounds):
                raise ValueError(
                    "Bounds must be a tuple of (min, max) pairs, "
                    "e.g. ((0.0, 1.0), (5.0, 10.0))."
                )
            # self.bounds_tuple = bounds
            # Give numeric or string axis labels consistently
            self.axes = tuple(str(i) for i in range(len(bounds)))
            self.bounds = {str(i): b for i, b in enumerate(bounds)}
        else:
            raise TypeError(
                f"Bounds must be either dict[str, (min, max)] or tuple of (min, max), got {type(bounds).__name__}"
            )

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
        self.grid = grid
        self.has_grid = grid is not None

    def __hash__(self):
        return hash(tuple(zip(self.bounds.items())))

    # -------------------------------------------------
    # Box modifications
    # -------------------------------------------------
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
    def intersect(self, other: "Box") -> Optional["Box"]:
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

    # @classmethod
    # def union_if_connected(
    #     cls,
    #     boxes: List["Box"],
    #     adjacency_fn=compute_box_adjacency,
    #     on_disconnected="raise",  # options: "raise", "none", "components"
    # ) -> "Box" | None | List["Box"]:
    #     """
    #     Return a single union only if the boxes form one connected component and share axes.
    #     Behavior when disconnected:
    #       - "raise": raises ValueError
    #       - "none": returns None
    #       - "components": returns unions per connected component (same as unions_of_connected_components)
    #     """
    #     sigs = {tuple(sorted(b.axes)) for b in boxes}
    #     if len(sigs) != 1:
    #         msg = "Boxes must share the same axis set to test connectivity jointly."
    #         if on_disconnected == "raise":
    #             raise ValueError(msg)
    #         elif on_disconnected == "none":
    #             return None

    #     # axis_order = next(iter(sigs))
    #     # bounds = boxes_to_bounds(boxes, axis_order)
    #     adjacency = adjacency_fn(bounds)
    #     comps = connected_components_from_adjacency(adjacency, boxes)
    #     if len(comps) == 1:
    #         return cls.union_all(boxes)
    #     if on_disconnected == "raise":
    #         raise ValueError(f"Boxes are not a single connected component; found {len(comps)} components.")
    #     elif on_disconnected == "none":
    #         return None
    #     elif on_disconnected == "components":
    #         # flatten to unions of components
    #         return [cls.union_all(comp) for comp in comps.values()]

    def contains_box(self, other: "Box") -> bool:
        shared_axes = set(self.axes) & set(other.axes)
        return all(
            self.bounds[ax][0] <= other.bounds[ax][0]
            and self.bounds[ax][1] >= other.bounds[ax][1]
            for ax in shared_axes
        )

    def overlaps(self, other: "Box") -> bool:
        shared_axes = set(self.axes) & set(other.axes)
        return all(
            not (self.bounds[ax][1] <= other.bounds[ax][0] or other.bounds[ax][1] <= self.bounds[ax][0])
            for ax in shared_axes
        )
    # ----------------------------------------------
    # relations to SpectralFeature
    # ----------------------------------------------
    def contains_feature(self, spec_feature: 'SpectralFeature', mode='loc') -> bool:
        """Return boolean for whether SpectralFeature lies inside the window."""
        if mode=='box':
            if spec_feature.feat_box is not None:
                return self.contains_box(spec_feature.feat_box)
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
        """
        spec_feature_ndim = len(spec_feature.location.coordinates)
        if spec_feature_ndim != self.ndim:
            raise ValueError(f"Expected SpectralFeature with a location with {self.ndim} coords, got {spec_feature_ndim}")

        if spec_feature.lineshape_parameter is None:
            raise ValueError(f"Expected SpectralFeature with `lineshape_parameter` attribute")

        contributing = True
        for ax, (mn, mx) in self.bounds.items():
            Gamma = spec_feature.lineshape_parameter[ax]
            # FIXME??   2*Gamma ??
            # in place ADDition
            contributing &= (spec_feature.location._coord_dict[ax] >= mn-2*Gamma) & (spec_feature.location._coord_dict[ax] <= mx+2*Gamma)
        return contributing and not self.contains_feature(spec_feature)

# Type aliases
CoordValue = Union[float, Literal['all']]
Coordinates = Tuple[Tuple[str, CoordValue], ...]

class ResLocGeoObject:
    """
    Represents geometric objects in N-dimensional space that are hashable.

    Examples:
        Point:      (('A', 1864.0), ('B', 900.0))
        Line:       (('A', 1864.0), ('B', 'all'))
        Plane:      (('A', 'all'), ('B', 'all'), ('C', 1200.0))
    """
    def __init__(self, coord_dict: dict[str, CoordValue]):
        self._coord_dict = coord_dict
        # Convert dict to sorted tuple of tuples for consistent hashing
        self.coordinates: Coordinates = tuple(sorted(coord_dict.items()))

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.coordinates)

    @property
    def values(self) -> tuple[CoordValue, ...]:
        return tuple(v for _, v in self.coordinates)

    @property
    def dimensionality(self) -> int:
        """Returns dimensionality of the object (0=point, 1=line, 2=plane, etc)"""
        return sum(1 for v in self.values if v == 'all')

    def __getitem__(self, axis: str) -> CoordValue:
        for k, v in self.coordinates:
            if k == axis:
                return v
        raise KeyError(f"Axis {axis} not found")

    def __eq__(self, other) -> bool:
        if not isinstance(other, ResLocGeoObject):
            return NotImplemented
        return self.coordinates == other.coordinates

    def __hash__(self) -> int:
        return hash(self.coordinates)

    def is_point(self) -> bool:
        return self.dimensionality == 0

    def is_line(self) -> bool:
        return self.dimensionality == 1

    def is_plane(self) -> bool:
        return self.dimensionality == 2

    def __repr__(self) -> str:
        """
        Returns a string representation showing type and coordinates.

        Examples:
            Point(A=1864.0, B=900.0)
            Line(A=1864.0, B=all)
            Plane(A=all, B=all, C=1200.0)
        """
        type_name = "Point" if self.is_point() else "Line" if self.is_line() else "Plane" if self.is_plane() else "Object"
        coords = ", ".join(f"{k}={v}" for k, v in self.coordinates)
        return f"{type_name}({coords})"


@dataclass
class SpectralFeature:
    location: 'ResLocGeoObject'
    term_contributions: tuple[TermParametersChoice] # grouped by res_motif
    lineshape_parameter: dict
    lineshape_parameter_single: float = 1.5
    amplitude_coeff: float = None
    feat_type: str = None
    feat_box: Box = None
    
    def __post_init__(self):
        if self.lineshape_parameter is not None:
            bounds = points_to_bounds(points=[self.location._coord_dict],
                                    halfwidths_list=[self.lineshape_parameter])[0]
            self.feat_box = Box(bounds)

    def __hash__(self) -> int:
        # return hash((self.location, self.term_contributions))
        return hash(self.location)

    def __eq__(self, other) -> bool:
        if not isinstance(other, SpectralFeature):
            return False
        return (self.location == other.location 
                and self.lineshape_parameter == other.lineshape_parameter 
                and self.term_contributions == other.term_contributions)

    @classmethod
    def share_location(cls, features: list['SpectralFeature']):
        if len(features)<2:
            raise ValueError("Can't compare 1 feature location")
        
        loc0 = features[0].location
        for f in features[1:]:
            if f.location != loc0:
                return False
        return True

    def union(self, other: 'SpectralFeature'):
        if self.location == other.location and self.lineshape_parameter == other.lineshape_parameter:
            if self.amplitude_coeff is None:
                raise ValueError("This SpectralFeature doesn't have `amplitude_coeff`")
            if other.amplitude_coeff is None:
                raise ValueError("Other SpectralFeature doesn't have `amplitude_coeff`")

            return SpectralFeature(location=self.location,
                                   term_contributions=self.term_contributions+other.term_contributions,
                                   amplitude_coeff=self.amplitude_coeff+other.amplitude_coeff)
        else:
            raise ValueError('Union is possible only when both location and lineshape_parameter are the same')

    @classmethod
    def filter_to_spec_window(cls, spec_features: list['SpectralFeature'],
                              spec_window: 'SpectralWindow'):
        """
        return spectral window with sorted features which are going to be evaluated in it
        """
        full_features = []
        contrib_features = []

        for feature in spec_features:
            # print('\nspec_window.box', spec_window.box)
            if spec_window.box.contains_feature(feature, mode='loc'):
                feature.feat_type = 'full'
                full_features.append(feature)
            if spec_window.box.contributing_feature(feature):
                feature.feat_type = 'contributing'
                contrib_features.append(feature)

        upd_spec_window = copy.deepcopy(spec_window)
        upd_spec_window.full_features = full_features
        upd_spec_window.contrib_features = contrib_features

        return upd_spec_window

    @classmethod
    def find_clusters_by_distance(cls, spec_features: list['SpectralFeature'],
                                  distance_thresholds: dict,
                                  linkage: str = 'single'):

        # features_locs = {loc_geo_obj.values:features[loc_geo_obj][1] for loc_geo_obj in features}
        features_locs = {feature.location.values: feature for feature in spec_features}
        from wilson_suite.wilson_intensities.amplitudes import domains

        clusters = domains.find_points_clusters_by_distance(res_locations=list(features_locs.keys()),
                                                            distance_thresholds=distance_thresholds,
                                                            linkage=linkage)
        rec_windows_dict = {}

        for g in clusters:
            rec_windows_dict[g] = RectangularDomain.from_features([features_locs[i] for i in clusters[g]])

        return rec_windows_dict

    def get_res_motifs(self) -> list[ResonanceMotif]:
        return [i.res_motif for i in self.term_contributions]


@dataclass
class SpectralWindow:
    """
    Represents an N-dimensional rectangular region (bounds only).

    For the full spectrum
    """
    box: 'Box'
    full_features: List['SpectralFeature'] = field(default_factory=list)
    contrib_features: List['SpectralFeature'] = field(default_factory=list)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return self.box.bounds

    # def __post_init__(self):
    #     if not is_tuple_of_tuples(self.bounds):
    #         raise ValueError("Bounds should be provided as a tuple of tuples")

    #     if not all(len(b) == 2 for b in self.bounds):
    #         raise ValueError("Each bound must be a (min, max) pair.")
    #     for i, (mn, mx) in enumerate(self.bounds):
    #         if mn > mx:
    #             raise ValueError(f"Invalid bound {i}: min >= max ({mn} >= {mx})")

    @property
    def ndim(self) -> int:
        return len(self.bounds)

    @property
    def widths(self) -> Tuple[float, ...]:
        return tuple(mx - mn for mn, mx in self.bounds)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return boolean mask of which points lie inside the window."""
        if points.shape[-1] != self.ndim:
            raise ValueError(f"Expected points with {self.ndim} coords, got {points.shape[-1]}")
        inside = np.ones(points.shape[:-1], dtype=bool)
        for i, (mn, mx) in enumerate(self.bounds):
            inside &= (points[..., i] >= mn) & (points[..., i] < mx)
        return inside

    # def contains_feature(self, spec_feature: SpectralFeature) -> bool:
    #     """Return boolean for whether SpectralFeature lies inside the window."""
    #     spec_feature_ndim = len(spec_feature.location.values)
    #     if spec_feature_ndim != self.ndim:
    #         raise ValueError(f"Expected SpectralFeature with a location with {self.ndim} coords, got {spec_feature_ndim}")

    #     inside = True
    #     for i, (mn, mx) in enumerate(self.bounds):
    #         inside &= (spec_feature.location.values[i] >= mn) & (spec_feature.location.values[i] <= mx)
    #     return inside

    # def contributing_feature(self, spec_feature: SpectralFeature) -> bool:
    #     """
    #     Return boolean for whether SpectralFeature is contributing to this window, 
    #         based on lineshape_parameter of this SpectralFeature
    #     """
    #     spec_feature_ndim = len(spec_feature.location.coordinates)
    #     if spec_feature_ndim != self.ndim:
    #         raise ValueError(f"Expected SpectralFeature with a location with {self.ndim} coords, got {spec_feature_ndim}")

    #     lineshape_params = list(spec_feature.lineshape_parameter.values())
    #     contributing = True
    #     for i, (mn, mx) in enumerate(self.bounds):
    #         Gamma = lineshape_params[i]
    #         # FIXME??   2*Gamma ??
    #         contributing &= (spec_feature.location.values[i] >= mn-2*Gamma) & (spec_feature.location.values[i] <= mx+2*Gamma)
    #     return contributing and not self.contains_feature(spec_feature)

    def sample_grid(self, dim_sizes: Dict) -> Dict[str, np.ndarray]:
        """Generate a regular grid of points spanning the window."""
        if len(dim_sizes) != self.ndim:
            raise ValueError("Grid shape must match dimensionality.")

        axes = {}
        for ax in self.bounds:
            mn, mx = self.bounds[ax]
            adjusted_mx = mx + (mx - mn) / dim_sizes[ax]
            axes[ax] = np.linspace(mn, adjusted_mx, dim_sizes[ax], endpoint=False)
            # axes.append(np.linspace(mn, mx, dim_sizes[ax], endpoint=True))
        # grid = np.stack(np.meshgrid(*list(axes.values()), indexing="ij"), axis=-1)
        grid = np.meshgrid(*list(axes.values()), indexing="ij")
        grid_d = {ax: grid[i] for i, ax in enumerate(axes)}
        return grid_d
        # return grid.reshape(-1, self.ndim)

    # def expand(self, padding: dict[str, float]) -> "SpectralWindow":
    #     """Return a new window expanded by `padding` in all directions."""
    #     pad_vals = list(padding.values())
    #     self.bounds = tuple((mn - pad_vals[i], mx + pad_vals[i]) for i, (mn, mx) in enumerate(self.bounds))
    #     # return SpectralWindow(tuple((mn - pad_vals[i], mx + pad_vals[i]) for i, (mn, mx) in enumerate(self.bounds)))

    # def intersect(self, other: "SpectralWindow") -> Optional["SpectralWindow"]:
    #     """Return the overlapping rectangular region, or None if no overlap."""
    #     if self.ndim != other.ndim:
    #         raise ValueError("Windows must have the same dimensionality.")

    #     overlap_bounds = []
    #     for (mn1, mx1), (mn2, mx2) in zip(self.bounds, other.bounds):
    #         lower = max(mn1, mn2)
    #         upper = min(mx1, mx2)
    #         if lower >= upper:
    #             # No overlap in this dimension - no intersection
    #             return None
    #         overlap_bounds.append((lower, upper))

    #     self.bounds = tuple(overlap_bounds)

    #     return self

    # @classmethod
    # def union(cls, windows: list["SpectralWindow"]):
    #     # union_bounds = []
    #     all_mins = []
    #     all_maxes = []
    #     for w in windows:
    #         for dim_bounds in w.bounds:
    #             all_mins.append(dim_bounds[0])
    #             all_maxes.append(dim_bounds[1])
    #     return cls(tuple([min(all_mins), max(all_maxes)]))

    def find_clusters_by_distance(self,
                                  spec_window_full: 'SpectralWindow',
                                  distance_thresholds: dict,
                                  linkage: str = 'single') -> dict[int, 'RectangularDomain']:

        spec_features = self.full_features + self.contrib_features

        # features_locs = {loc_geo_obj.values:features[loc_geo_obj][1] for loc_geo_obj in features}
        features_locs = {feature.location.values: feature for feature in spec_features}
        from wilson_suite.wilson_intensities.amplitudes import domains

        clusters = domains.find_points_clusters_by_distance(res_locations=list(features_locs.keys()),
                                                            distance_thresholds=distance_thresholds,
                                                            # spec_window_full=spec_window_full,
                                                            linkage=linkage)
        rec_windows_dict = {}

        for g in clusters:
            domain = RectangularDomain.from_features([features_locs[i] for i in clusters[g]])
            for feature in domain.full_features:
                paddig_dict = {k:2*v for k,v in feature.lineshape_parameter.items()}
                domain.box.expand(paddig_dict)
            domain.box.intersect(spec_window_full)
            rec_windows_dict[g] = domain

            # rec_windows_dict[g] = RectangularDomain.from_features([features_locs[i] for i in clusters[g]])

        return rec_windows_dict


@dataclass
class RectangularDomain:
    """
    N-dimensional domain with labeled axes and spectral features.


    """
    box: Box
    shape: Tuple[int, ...] = None
    labels: Optional[Tuple[str, ...]] = None
    full_features: List['SpectralFeature'] = field(default_factory=list)
    contrib_features: List['SpectralFeature'] = field(default_factory=list)
    level_clustered: int = None
    union_lvl0_mask: np.ndarray = None # should be of the shape of the grid of lvl1 domain

    def __post_init__(self):
        from wilson_suite.wilson_utils.common_labels import cap_alpha_labels

        # --- bounds ---
        if isinstance(self.box, tuple) or isinstance(self.box, dict):  # allow legacy tuple bounds
            self.box = Box(self.box)
        
        # if self.bounds.ndim != len(self.shape):
        #     raise ValueError("Bounds dimensionality must match shape length.")

        # --- labels ---
        # if self.labels is None:
        #     if len(self.shape) > len(cap_alpha_labels):
        #         raise ValueError(f"Not enough predefined labels for {len(self.shape)} dimensions.")
        #     self.labels = tuple(cap_alpha_labels[:len(self.shape)])
        # elif len(self.labels) != len(self.shape):
        #     raise ValueError("Labels must match shape dimensionality.")

        # --- label-index map ---
        if self.labels is not None:
            self._label_to_index = {label: i for i, label in enumerate(self.labels)}

    def __hash__(self):
        return hash(tuple([self.box, tuple(self.full_features), tuple(self.contrib_features)]))
    
    # -----------------------------------------------------------
    # Feature utilities
    # -----------------------------------------------------------
    def add_full_features(self, features: List['SpectralFeature']):
        self.full_features.extend(features)

    def add_a_full_feature(self, feature: 'SpectralFeature'):
        self.full_features.append(feature)

    def add_contrib_features(self, features: List['SpectralFeature']):
        self.contrib_features.extend(features)

    def add_a_contrib_feature(self, feature: 'SpectralFeature'):
        self.contrib_features.append(feature)

    def features_in_bounds(self) -> List['SpectralFeature']:
        """Return features whose coordinates lie inside the domain's window."""
        if not self.full_features:
            return []
        coords = np.array([f.location.as_array() for f in self.full_features])
        mask = self.box.contains(coords)
        return [f for f, m in zip(self.full_features, mask) if m]

    @classmethod
    def from_features(cls, features: List['SpectralFeature'], padding: float = 0.0):
        """Create a domain whose window tightly bounds given features."""
        if not features:
            raise ValueError("Feature list cannot be empty.")

        # Collect all numeric coords per axis
        axis_vals: dict[str, list[float]] = {}
        for f in features:
            for axis, val in f.location.coordinates:
                if val != "all":
                    axis_vals.setdefault(axis, []).append(float(val))

        # Determine bounds and shape
        bounds = []
        for axis, vals in sorted(axis_vals.items()):
            min_val, max_val = min(vals) - padding, max(vals) + padding
            bounds.append((min_val, max_val))
        shape = tuple(len(axis_vals[a]) for a in sorted(axis_vals))

        window = SpectralWindow(Box(tuple(bounds)))
        domain = cls(bounds=window)
        domain.add_full_features(features)
        return domain

    def generate_meshgrids(self, steps_axes) -> Tuple[np.ndarray, ...]:
        """Generate coordinate grids for each dimension."""

        coords_1d = [self.axis_coords(i, steps_axes) for i in self.labels]

        return np.meshgrid(*coords_1d, indexing='ij')

    def axis_coords(self, key: Union[int, str], steps_axes) -> np.ndarray:
        """Return coordinate values along a labeled axis."""

        i = self.axis_index(key)
        min_val, max_val = self.box.bounds[i]
        print('min_val, max_val', min_val, max_val)
        print(safe_arange_inclusive_scaled(min_val, max_val, steps_axes[key]))
        return safe_arange_inclusive_scaled(min_val, max_val, steps_axes[key])
        # return linspace_with_step(min_val, max_val, steps_axes[key])

    def axis_index(self, key: Union[int, str]) -> int:
            """Resolve axis index from label or integer."""

            if isinstance(key, str):
                return self._label_to_index[key]
            return key


@dataclass
class SpectroscopicAxis:
    """
    label to [list of tuples of pulse IDs] 
            which corresponds to linear combination of the axis variables make up the axes

    {'A': [(-1,)]}, {'B': [(2,)]}
    {'A': [(2,)]},  {'B': [(-1,), (2,)]}
    """
    label: str
    indep_vars: tuple[tuple]

    def __hash__(self):
        return hash((self.label, self.indep_vars))


@dataclass
class SpectroscopicAxes:
    """
    Dict-like holder of axes label to [list of tuples of pulse IDs] 
            which corresponds to linear combination of the axis variables make up the axes

    experiment.canonical_axes   {'A': [(-1,)], 'B': [(2,)]}
    chosen                      {'A': [(2,)], 'B': [(-1,), (2,)]}
    """
    axes: tuple[SpectroscopicAxis]

    @classmethod
    def from_axes_dict(cls, axes_dict: dict[str, list[tuple]]):
        axes = tuple([SpectroscopicAxis(label=k, indep_vars=tuple(v)) for k,v in axes_dict.items()])
        return cls(axes)



