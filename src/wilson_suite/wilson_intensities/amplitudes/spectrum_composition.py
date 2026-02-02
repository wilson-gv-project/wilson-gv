import numpy as np
import copy
from wilson_suite.wilson_intensities.amplitudes.domains import points_to_bounds
from wilson_suite.wilson_intensities.amplitudes.term_parts import ResonanceMotif, TermParametersChoice, safe_arange_inclusive_scaled

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Literal


@dataclass
class Grid:
    """
    """
    meshgrids: Tuple[np.ndarray, ...]

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
    grid: Grid = None # UNUSED
    has_grid: bool = None # UNUSED

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
    # ----------------------------------------------
    # relations to SpectralFeature
    # ----------------------------------------------
    def contains_feature(self, spec_feature: 'SpectralFeature', mode='loc') -> bool:
        """Return boolean for whether SpectralFeature lies inside the window."""
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

    # UNUSED - useful for analysis or for future?
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
    term_contributions: tuple[TermParametersChoice] = None # grouped by res_motif
    lineshape_parameter: float = None # will be by the time of init in the unit of cm-1
    amplitude_coeff: float = None
    feat_type: str = None
    feat_box: Box = None
    
    def __post_init__(self):
        # making boxes around the points for features using the lineshape_parameter
        if self.lineshape_parameter is not None:
            # print('-- self.lineshape_parameter for feature box:', self.lineshape_parameter)
            bounds = points_to_bounds(points=[self.location._coord_dict],
                                    halfwidth=self.lineshape_parameter)[0]
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

    # UNUSED
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

            t1 = tuple(self.term_contributions) if self.term_contributions is not None else ()
            t2 = tuple(other.term_contributions) if other.term_contributions is not None else ()
            term_contributions = t1 + t2

            return SpectralFeature(location=self.location,
                                   term_contributions=term_contributions,
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

    # UNUSED
    @classmethod
    def find_clusters_by_distance(cls, spec_features: list['SpectralFeature'],
                                  distance_thresholds: dict,
                                  linkage: str = 'single'):

        features_locs = {feature.location.values: feature for feature in spec_features}
        from wilson_suite.wilson_intensities.amplitudes import domains

        clusters = domains.find_points_clusters_by_distance(res_locations=list(features_locs.keys()),
                                                            distance_thresholds=distance_thresholds,
                                                            linkage=linkage)
        rec_windows_dict = {}

        for g in clusters:
            rec_windows_dict[g] = RectangularDomain.from_features([features_locs[i] for i in clusters[g]])

        return rec_windows_dict

    # UNUSED
    def get_res_motifs(self) -> list[ResonanceMotif]:
        return [i.res_motif for i in self.term_contributions]

    @classmethod
    def get_max_intensity_feat(cls, features: list['SpectralFeature'],
                          intensity_expr: str = 'abs()**2') -> 'SpectralFeature':
        """
        amplitude of a feature is given by: amplitude_coeff / lineshape_parameter**2
        """
        result = None
        intensity_result = 0

        for feat in features:
            
            if feat.get_intensity(intensity_expr) > intensity_result:
                result = feat
                intensity_result = feat.get_intensity(intensity_expr)
        
        return result

    def get_intensity(self, intensity_expr: str = 'abs()**2') -> float:
        """
        ! Assumption: lineshape_parameter is homogeneous/ universal over spectral dimensions
        intensity will be returned in au
        """
        
        from wilson_suite.wilson_utils.unit_convertor import convNu2Ene, linewidth_cm_or_au
        
        if intensity_expr == 'abs()**2':
           N = len(self.location.dims)

           if linewidth_cm_or_au(self.lineshape_parameter) == 'au':
                return abs(self.amplitude_coeff / (-1j*self.lineshape_parameter)**N)**2
           elif linewidth_cm_or_au(self.lineshape_parameter) == 'cm-1':
                return abs(self.amplitude_coeff / (-1j*convNu2Ene(self.lineshape_parameter))**N)**2
        else:
            raise NotImplementedError("Only standard 'abs()**2' expression is implemented.")
    
    @classmethod
    def dress_these_with_boxes(cls, features: list['SpectralFeature'], 
                               max_intensity, min_intensity,
                               lineshape_parameter=None) -> list['SpectralFeature']:
        """
        but features should have lineshape_parameter before doing this 
            or it will be set to be the same here from input

        In this function there is also a calculation of the box halfwidth, based on the solution of the equation:

        | C / (wa - A - i*G) / (wb - B - i*G) |**2 >= f_min
        where C is amplitude coeff, f_min is the minimum intensity for this dynamic range.
            Gamma is the same for both resonance conditions 
            and the box will have equal sides (halfwidths in all dimensions will be the same)

        wa - A = deltaA - 0 at the resonance, otherwise represent distance from the resonance
        wb - B = deltaB - 0 at the resonance, otherwise represent distance from the resonance

        assuming the same distances in all dimensions:
        | C / (deltaA - i*G)**2 |**2 >= f_min
        | C |**2 / (deltaA - i*G)**4 >= f_min
        | C |**2 / (deltaA**2 + G**2)**2 >= f_min

        | C |**2 / (deltaA - i*G)**4 >= f_min
        (deltaA - i*G)**4 <= | C |**2 / f_min


        [YES, NOW IT ONLY WORKS FOR 2 RESONANCE CONDITIONS]

        generally: 
        | C / (deltaA - i*G)**N |**2 >= f_min
        | C |**2 / (deltaA - i*G)**2*N >= f_min
        (deltaA - i*G)**2*N <= | C |**2 / f_min
        |deltaA - i*G| <= (| C |**2 / f_min) ** 1/(2*N)

        
        for N = 1:
        | C / (deltaA - i*G) |**2 >= f_min
        | C |**2 / (deltaA - i*G)**2 >= f_min
        (deltaA - i*G)**2 <= | C |**2 / f_min
        
        """
        import copy
        features = copy.deepcopy(features)

        # sort out units of lineshape_parameter
        # should be in unit of the grid - cm-1 normally

        for feat in features[:]:  # Iterate over a copy of the list
            
            if feat.get_intensity() < min_intensity:
                features.remove(feat)
            else:

                feat.feat_box = None
                
                if feat.get_intensity() > max_intensity:
                    raise ValueError(f"The feature {feat} will have higher intensity than max_intensity ({max_intensity})")
                
                if lineshape_parameter is not None:
                    feat.lineshape_parameter = lineshape_parameter
                
                # will make a square box, so lineshape_parameter is assumed to be the same for all dimensions
                N = len(feat.location.dims)
                D_gen = (abs(feat.amplitude_coeff)**2 / min_intensity) ** (1/N)
                # lineshape_parameter should be in unit of the grid? - cm-1 normally
                from wilson_suite.wilson_utils.unit_convertor import convNu2Ene, linewidth_cm_or_au
                
                if linewidth_cm_or_au(feat.lineshape_parameter) == 'cm-1':
                    lineshape_parameter = convNu2Ene(feat.lineshape_parameter)
                elif linewidth_cm_or_au(feat.lineshape_parameter) == 'au':
                    lineshape_parameter = feat.lineshape_parameter
                
                if D_gen < lineshape_parameter**2:
                    # not sure if this will be reached if condition feat.get_intensity() < min_intensity is satisfied
                    print(f"Warning: {feat.get_intensity(): .2e}, min {min_intensity:.2e} max {max_intensity: .2e}")
                    features.remove(feat)
                else:
                    delta_a_general = np.sqrt(D_gen - lineshape_parameter**2)
                    # convert au to cm-1 for deltaA for box bounds
                    delta_a_general = convNu2Ene(delta_a_general, reverse=True)
                    feat.feat_box = Box({k: (v-delta_a_general, v+delta_a_general) for k,v in feat.location._coord_dict.items()})
        return features

    @classmethod
    def print_list_features(cls, features: list['SpectralFeature']):
        for feat in features:
            print('\n -- A feature at the location', feat.location, 'with featbox', feat.feat_box)

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

    @property
    def ndim(self) -> int:
        return len(self.bounds)

    # UNUSED
    @property
    def widths(self) -> Tuple[float, ...]:
        return tuple(mx - mn for mn, mx in self.bounds)

    # UNUSED
    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return boolean mask of which points lie inside the window."""
        if points.shape[-1] != self.ndim:
            raise ValueError(f"Expected points with {self.ndim} coords, got {points.shape[-1]}")
        inside = np.ones(points.shape[:-1], dtype=bool)
        for i, (mn, mx) in enumerate(self.bounds):
            inside &= (points[..., i] >= mn) & (points[..., i] < mx)
        return inside


    def sample_grid(self, dim_sizes: Dict) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate a regular grid of points spanning the window.
        """

        if len(dim_sizes) != self.ndim:
            raise ValueError("Grid shape must match dimensionality.")
        axes = {}
        for ax in self.bounds:
            mn, mx = self.bounds[ax]
            adjusted_mx = mx + (mx - mn) / dim_sizes[ax]
            axes[ax] = np.linspace(mn, adjusted_mx, dim_sizes[ax], endpoint=False) # adjust min too???

        coords_vectors = list(axes.values())
        grid = np.meshgrid(*coords_vectors, indexing="ij")
        grid_d = {ax: grid[i] for i, ax in enumerate(axes)}
        
        return axes, grid_d


    def find_clusters_by_featboxes(self) -> tuple['RectangularDomain']:
        all_features = self.full_features + self.contrib_features

        from . import domains
        clusters = domains.features_to_clusters(features=all_features)

        return tuple(
            RectangularDomain.from_features(clusters[c])
            for c in clusters
        )

    def dress_with_featboxes(self, dynrange):
        """

        making SpectralFeature.feat_box attribute value
            as a concequence, can rm features that are outside of the range 
        
        !warning: it is posssibly late to do this for a window when it has identified full_features and contrib_features
        """
        feat = SpectralFeature.get_max_intensity_feat(self.full_features)
        max_intensity_in_window = feat.get_intensity()
        min_intensity_in_window = max_intensity_in_window / dynrange

        new_full_features = SpectralFeature.dress_these_with_boxes(self.full_features, max_intensity_in_window, min_intensity_in_window)
        new_contrib_features = SpectralFeature.dress_these_with_boxes(self.contrib_features, max_intensity_in_window, min_intensity_in_window)

        return SpectralWindow(box=self.box, 
                              full_features=new_full_features, 
                              contrib_features=new_contrib_features)



@dataclass
class RectangularDomain:
    """
    N-dimensional domain with labeled axes and spectral features.


    """
    box: Box
    full_features: List['SpectralFeature'] = field(default_factory=list)
    contrib_features: List['SpectralFeature'] = field(default_factory=list)

    def __post_init__(self):

        # --- bounds ---
        if isinstance(self.box, tuple) or isinstance(self.box, dict):  # allow legacy tuple bounds
            self.box = Box(self.box)
        

    def __hash__(self):
        return hash(tuple([self.box, tuple(self.full_features), tuple(self.contrib_features)]))
    
    # -----------------------------------------------------------
    # Feature utilities
    # -----------------------------------------------------------
    # UNUSED by extention
    def add_full_features(self, features: List['SpectralFeature']):
        self.full_features.extend(features)

    # UNUSED
    def add_a_full_feature(self, feature: 'SpectralFeature'):
        self.full_features.append(feature)

    # UNUSED
    def add_contrib_features(self, features: List['SpectralFeature']):
        self.contrib_features.extend(features)

    # UNUSED
    def add_a_contrib_feature(self, feature: 'SpectralFeature'):
        self.contrib_features.append(feature)

    # UNUSED - so far? useful for analysis?
    def features_in_bounds(self) -> List['SpectralFeature']:
        """Return features whose coordinates lie inside the domain's window."""
        if not self.full_features:
            return []
        coords = np.array([f.location.as_array() for f in self.full_features])
        mask = self.box.contains(coords)
        return [f for f, m in zip(self.full_features, mask) if m]


    @classmethod
    def from_features(cls, features: list[SpectralFeature]):
        return cls(
            box=Box.union([f.feat_box for f in features]),
            full_features=features
        )

# UNUSED by extention
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


# UNUSED
@dataclass
class SpectroscopicAxes:
    """
    Dict-like holder of axes label to [list of tuples of pulse IDs] 
            which corresponds to linear combination of the axis variables make up the axes

    experiment.canonical_axes   {'A': [(-1,)], 'B': [(2,)]}
    chosen                      {'A': [(2,)], 'B': [(-1,), (2,)]}
    """
    axes: tuple[SpectroscopicAxis]

    # FIXME: This method is not up to date with the spectral axes generalization but is in a class marked as unused, don't know whether update is needed.
    @classmethod
    def from_axes_dict(cls, axes_dict: dict[str, list[tuple]]):
        axes = tuple([SpectroscopicAxis(label=k, indep_vars=tuple(v)) for k,v in axes_dict.items()])
        return cls(axes)



