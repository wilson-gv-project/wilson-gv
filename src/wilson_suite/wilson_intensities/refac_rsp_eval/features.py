"""
Spectral features, and the windows and domains that hold them.

  in:  ContributionRow tables (built by evaluate.build_contributions), lineshape parameters
  out: SpectralFeature lists, SpectralWindow, RectangularDomain clusters

  holds: ContributionRow, ContributionTable, SpectralFeature, features_from_rows, features_to_clusters,
         SpectralWindow, RectangularDomain

Takes Box and box clustering from grid.py; grid.py never imports from here.
Note: SpectralWindow.sample_grid is pure geometry and could move to grid.py later.
"""
import copy
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
    compute_box_adjacency,
    connected_components_from_adjacency,
    points_to_bounds,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResonanceMotif,
)
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
        Dim_bounds,
    )
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
        ResLocPoint,
    )

## -------------------------------------------------------------------------------
##          Complete data computation: from terms to contributions to features
## -------------------------------------------------------------------------------

"""
CompiledTerm list
  │ 1. index sets: fix the motif labels (make_idx_sets over term.idx_nonsumm)
  v
(term, params) pairs
  │ 2. coeff: evaluate_term_coeff_sumover, sums the other labels     ← MolSystemData, polarization
  │    skip the pair if coeff == 0
  │ 3. location: solve_LSE_motif, once per (motif, params)            ← VibStatesData
  v
ContributionRow(term_id, motif, params, location, coeff)   "why" layer
  │ 4. features_from_rows: group by (motif, location), add the coeffs
  v
SpectralFeature(location, rows)                            "what" layer
  │ 5. add peak width, filter to the spectral window
  v
grid.py → spectrum array                                   the picture

"""

K = TypeVar('K', bound=Hashable)   # group_by key


@dataclass(frozen=True)
class ContributionRow:
    """One term at one index set. The term itself is terms[term_id]."""
    term_id: int
    motif: ResonanceMotif
    params: ParameterSet
    location: 'ResLocPoint'
    coeff: float


class ContributionTable:
    def __init__(self, rows: Iterable[ContributionRow]):
        self._rows = tuple(rows)

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    # the two general tools
    def where(self, keep: Callable[[ContributionRow], bool]) -> 'ContributionTable':
        return ContributionTable(r for r in self._rows if keep(r))

    def group_by(self, key: Callable[[ContributionRow], K]) -> dict[K, 'ContributionTable']:
        groups = defaultdict(list)
        for r in self._rows:
            groups[key(r)].append(r)
        return {k: ContributionTable(v) for k, v in groups.items()}

    # named shortcuts for frequent questions
    def by_params(self):
        return self.group_by(lambda r: r.params)

    def by_motif(self):
        return self.group_by(lambda r: r.motif)

    def by_location(self, tol_cm: float = 0.01):
        return self.group_by(lambda r: tuple((ax, round(v / tol_cm)) for ax, v in r.location.coordinates))

    def axis_range(self, axis):
        vals = [r.location[axis] for r in self._rows if axis in r.location.axes]
        if not vals:
            raise ValueError(f'no row has axis {axis!r}')
        return min(vals), max(vals)


@dataclass
class SpectralFeature:
    """
    One peak: the rows of one motif at one location (made by features_from_rows).
    Same motif + same point -> same peak shape, so the coefficients of the rows add.
    location and lineshape_parameter are in cm-1.
    """
    location: 'ResLocPoint'
    rows: tuple[ContributionRow, ...] = ()
    lineshape_parameter: float | None = None
    scale: float = 1.0      # normalize_coeffs_to_max changes this, never the rows
    feat_type: str | None = None
    feat_box: Box | None = None

    def __post_init__(self):
        # making boxes around the points for features using the lineshape_parameter
        if self.lineshape_parameter is not None:
            bounds = points_to_bounds(points=[self.location.as_dict()],
                                    halfwidth=self.lineshape_parameter)[0]
            self.feat_box = Box(bounds)

    @property
    def amplitude_coeff(self) -> float | None:
        """scale * sum of the row coefficients; None for a feature without rows."""
        if not self.rows:
            return None
        return self.scale * sum(r.coeff for r in self.rows)

    @property
    def motif(self) -> ResonanceMotif | None:
        motifs = {r.motif for r in self.rows}
        if len(motifs) > 1:
            raise ValueError(f'Expected one motif per feature, got {len(motifs)}')
        return next(iter(motifs), None)

    @property
    def param_sets(self) -> tuple[ParameterSet, ...]:
        """Index sets of the rows, each once, in row order."""
        return tuple(dict.fromkeys(r.params for r in self.rows))

    @property
    def term_ids(self) -> tuple[int, ...]:
        """term_id of the rows, each once, in row order."""
        return tuple(dict.fromkeys(r.term_id for r in self.rows))

    def __hash__(self) -> int:
        return hash((self.location, self.lineshape_parameter, self.rows, self.scale))

    def __eq__(self, other) -> bool:
        if not isinstance(other, SpectralFeature):
            return False
        return (self.location == other.location
                and self.lineshape_parameter == other.lineshape_parameter
                and self.rows == other.rows
                and self.scale == other.scale)

    def __lt__(self, other: 'SpectralFeature') -> bool:
        if not isinstance(other, SpectralFeature):
            return NotImplemented
        if self.amplitude_coeff is not None and other.amplitude_coeff is not None:
            return abs(self.amplitude_coeff) < abs(other.amplitude_coeff)
        else:
            raise ValueError("cannot compare amplitude_coeff of features")

    def anharm_contributions(self, terms) -> dict:
        """
        amplitude_coeff split by terms[term_id].anharmonicity; the parts add up to amplitude_coeff.
        terms - indexable by term_id, e.g. the VibPerturbedTerm list the CompiledTerms were made from
        """
        result = defaultdict(float)
        for r in self.rows:
            result[terms[r.term_id].anharmonicity] += self.scale * r.coeff
        return dict(result)

    def __repr__(self) -> str:
        return f'SpectralFeature(location={self.location}, rows={len(self.rows)}, amplitude_coeff={self.amplitude_coeff})'

    @classmethod
    def sort_by_params(cls, features: list['SpectralFeature']):
        """
        Sort features by their sorted index sets.
        """
        return sorted(features, key=lambda f: tuple(sorted(f.param_sets)))

    @classmethod
    def normalize_coeffs_to_max(cls, features: list['SpectralFeature'], external_max: float | None = None):
        """
        returns a new list; each copy gets scale / |max|, the rows stay as they are

        external_max - can take external input for max , instead of finding max of the given list
        """
        if external_max is None:
            max_feat_coeff = cls.get_max_intensity_feat(features, intensity_expr=None).amplitude_coeff
        else:
            max_feat_coeff = external_max

        return_feats = copy.deepcopy(features)

        for f in return_feats:
            if f.amplitude_coeff is None:
                raise ValueError(f'feature {f} has no amplitude_coeff')
            f.scale = f.scale / abs(max_feat_coeff) # type: ignore
        return return_feats

    @classmethod
    def get_feats_with_params(cls, features: list['SpectralFeature'], params: dict):
        """
        e.g.:
            params = {'a': 0, 'b': 1}
        """
        target = ParameterSet(params)
        return [f for f in features if target in f.param_sets]

    def _check_same_axes(self, box: Box):
        if self.location.axes != box.axes:
            raise ValueError(f"Expected SpectralFeature with axes {box.axes}, got {self.location.axes}")

    def is_inside(self, box: Box) -> bool:
        """
        Return boolean for whether the feature location lies inside `box`, edges included.
        """
        self._check_same_axes(box)

        inside = True
        for ax, (mn, mx) in box.bounds.items():
            inside &= (self.location[ax] >= mn) & (self.location[ax] <= mx)
        return inside

    def feat_box_overlaps(self, box: Box) -> bool:
        """
        Return boolean for whether the feature box (feat_box) overlaps `box`, touching edges excluded.
        """
        self._check_same_axes(box)

        if self.feat_box is None:
            raise ValueError('Need to add a box for this feature')
        return box.overlaps(self.feat_box)

    def contributes_to(self, box: Box) -> bool:
        """
        Return boolean for whether this feature lies outside `box` but still adds intensity inside it:
            within 2*lineshape_parameter of `box` on every axis.
        """
        self._check_same_axes(box)

        if self.lineshape_parameter is None:
            raise ValueError("Expected SpectralFeature with `lineshape_parameter` attribute")

        contributing = True
        for ax, (mn, mx) in box.bounds.items():
            Gamma = self.lineshape_parameter
            # FIXME??   2*Gamma ??
            # in place ADDition
            contributing &= (self.location[ax] >= mn-2*Gamma) & (self.location[ax] <= mx+2*Gamma)
        return contributing and not self.is_inside(box)

    
    @classmethod
    def filter_to_spec_window(cls, spec_features: list['SpectralFeature'],
                              spec_window: 'SpectralWindow'):
        """
        return spectral window with sorted features which are going to be evaluated in it.
        Creates a deep copy of spec_features.
        """
        cp_spec_features = copy.deepcopy(spec_features)
        full_features = []
        contrib_features = []

        for feature in cp_spec_features:
            if feature.is_inside(spec_window.box):
                feature.feat_type = 'full'
                full_features.append(feature)
            if feature.contributes_to(spec_window.box):
                feature.feat_type = 'contributing'
                contrib_features.append(feature)
        upd_spec_window = copy.deepcopy(spec_window)
        upd_spec_window.full_features = full_features
        upd_spec_window.contrib_features = contrib_features

        return upd_spec_window

    @classmethod
    def get_max_intensity_feat(cls, features: list['SpectralFeature'],
                          intensity_expr: str | None = 'abs()**2') -> 'SpectralFeature':
        """
        amplitude of a feature is given by: amplitude_coeff / lineshape_parameter**2
        """
        result = None
        num_result = 0

        for feat in features:
            
            if intensity_expr is not None:

                if feat.get_intensity(intensity_expr) > num_result:
                    result = feat
                    num_result = feat.get_intensity(intensity_expr)
            
            else:

                if feat.amplitude_coeff and abs(feat.amplitude_coeff) > num_result:
                    result = feat
                    num_result = abs(feat.amplitude_coeff)

        if result is not None:
            return result
        if not features:
            raise ValueError('Expected at least one feature, got an empty list')
        measure = 'amplitude_coeff' if intensity_expr is None else 'intensity'
        raise ValueError(f'None of the {len(features)} features has a nonzero {measure}')
    

    def get_intensity(self, intensity_expr: str = 'abs()**2') -> float:
        """
        ! Assumption: lineshape_parameter is homogeneous/ universal over spectral dimensions
        lineshape_parameter is in cm-1; intensity will be returned in au
        """

        if self.amplitude_coeff is None or self.lineshape_parameter is None:
            raise ValueError('this feature has no amplitude_coeff and/or lineshape_parameter')
        if self.lineshape_parameter <= 0:
            raise ValueError(f'lineshape_parameter must be > 0, got {self.lineshape_parameter}')

        if intensity_expr == 'abs()**2':
            N = len(self.location.axes)
            gamma_au = convNu2Ene(self.lineshape_parameter)
            return abs(self.amplitude_coeff / (-1j*gamma_au)**N)**2
        else:
            raise NotImplementedError("Only standard 'abs()**2' expression is implemented.")
    
    @classmethod
    def dress_these_with_boxes(cls, features: list['SpectralFeature'],
                               max_intensity, min_intensity,
                               box_range_safety_margin: float=0.1,
                               scale_wrt_max_intensity: bool=False,
                               minimum_box_padding: float=0.) -> list['SpectralFeature']:
        """
        Assumptions: Feature locations, lineshape parameters (and minimum box padding if used) given in the same units

        Limitations: This method is currently for m == n for m-D features in n-D spectra. May need further work for
                     m != n.

        Note that well-functioning of this routine should be reviewed after usage experience.

        Dressing features with boxes: Uses a helper function describing at which radius from a Lorentzian's centerpoint
        that the intensity is "dynamic range times smaller" than at the centerpoint, and then constructs an n-dimensional
        (currently "n-dimensional cubic") box of that distance from the centerpoint along the given axes:
          - the distance provided by the helper function gives distances along the axes under the "worst case" assumption
            that each axis is parallel to the resonance condition direction
            - the rest of the term is then taken as constant and the distance can be determined in each direction separately
            - otherwise (if not each axis and resonance condition parallel), walking along one axis means gaining
              distance to more than one resonance condition, which means that the intensity will be even smaller than
              dictated by a single Lorentzian
            - if so (not parallel), the box would become somewhat larger than needed but I think at most sqrt(N),
              where N is the spectrum dimensionality, for a given direction
                - for instance in the 2D case, if one resonance condition runs along axis A and the other along A + B,
                then the lineshapes will be at a 45 degree angle to each other on a plot over the (A, B) space. The box
                could have then been sqrt(2) shorter along the B direction.
                - This can likely be accomplished by inspecting each resonance condition more closely for which
                  combination of axes they run along and accounting for this angle in the box scaling via trigonometry

        For m != n (feature--spectrum dimensionality): One typical example is a line feature (e.g. CARS) in a 2D spectrum).
        This feature decreases along one direction in the spectrum and stays constant along the other, and the present
        algorithm will then likely need adjustment to make an appropriate box for this (if sticking to rectangular boxes,
        will probably need to return a box covering the full region of the spectral window spanned by such a feature to
        a given tolerance).

        Each feature's own lineshape_parameter sets its box size; a feature without one raises a ValueError.

        box_range_safety_margin: How much larger should the box dimensions be than what is dictated by
            lorentzian_distance_to_dynrange_weaker_than_max?
            default: 0.1 (ten percent margin). Note: Accounting for the larger boxes that may be warranted by constructive
            interference between features whose lineshapes overlap might be feasible to handle in a somewhat crude manner
            by adjusting this parameter.

        scale_wrt_max_intensity (default False): When determining box sizes for a given feature, should the box size be
        scaled according to the max_intensity parameter? (i.e. "weaker features need smaller boxes so I decrease the
        desired dynamic range for weaker features (compared to their centerpoint value)")

        minimum_box_padding (default 0.0): Apply a minimum size box around each feature, where the value of this parameter
        is the minimum distance to the box surface centerpoints along each positive/negative Cartesian direction, i.e.
        boxes may be larger than this minumum, but not smaller. If applied, then is unchanged by box_range_safety_margin.
        If set to 0.0, features with point intensity < min_intensity will be removed, potentially removing effects
        that should have been included if e.g. these features were close to each other on the shoulder of a stronger feature

        Recommended use unless performance is critical:
            - box_range_safety margin at default or other judiciously chosen nonzero value
            - scale_wrt_max_intensity: True (recommended as long as the collection of features includes the spectrum's strongest feature)
            - minimum_box_padding: Low but positive number (2-10 * lineshape parameter)

        """


        def lorentzian_distance_to_dynrange_weaker_than_max(gamma: float, dynrange) -> float:
            """
            Helper function: For a Lorentzian lineshape with parameter gamma, with the functional form
            L(w) = | A / ((w - w_0) - i*gamma) |^2
            after applying absolute square for intensity,
            at which radius (units same as gamma) must an n-dimensional sphere be drawn around the
            resonance location w_0 (the maximum) of the lineshape so that this lineshape's intensity is < 1/dynrange of the
            maximum's intensity everywhere outside the sphere?

            Found by evaluating L(x - x_0)/L(x_0) = k, with L from the above expression, and solving for the x that
            gives the desired k (where k = 1/dynrange)
            """

            # If called with effective dynamic range, it is possible to request dynrange < 0 but this corresponds
            # to a negative box size (and the default formula will return NaN) - therefore return 0.0 instead
            if dynrange < 1.0:
                return 0.0

            else:
                return ( (gamma ** 2.0 - (1.0 / dynrange) * (gamma ** 2.0)) / (1.0 / dynrange) ) ** 0.5

        import copy

        # Defining here an implied dynamic range based on the max/min specified intensity ratio
        implied_dynrange = max_intensity/min_intensity

        # Return data initialization
        res_features = copy.deepcopy(features)

        for feat in res_features[:]:

            feat_intensity = feat.get_intensity()

            # FIXME Warning: If features removed by this were e.g. close to each other and/or on the shoulder of a stronger feature,
            # this removal may be too strict. Can be mitigated by choosing nonzero minimum_box_padding. 
            if (feat_intensity < min_intensity) and (minimum_box_padding == 0.0):
                res_features.remove(feat)
            else:

                feat.feat_box = None

                if feat_intensity > max_intensity:
                    raise ValueError(f"The feature {feat} will have higher intensity than max_intensity ({max_intensity})")
                
                if scale_wrt_max_intensity:

                    feat_intensity_wrt_max = feat_intensity/max_intensity
                    delta_a_general = lorentzian_distance_to_dynrange_weaker_than_max(feat.lineshape_parameter, implied_dynrange*feat_intensity_wrt_max) # type: ignore

                else:
                    delta_a_general = lorentzian_distance_to_dynrange_weaker_than_max(feat.lineshape_parameter, implied_dynrange) # type: ignore

                # gamma = feat.lineshape_parameter
                # c = feat.amplitude_coeff
                box_extent = max(delta_a_general*(1.0 + box_range_safety_margin), minimum_box_padding)
                # print('box_extent', box_extent, np.abs(c/(box_extent-1j*gamma)/(box_extent-1j*gamma)), np.abs(c/(-1j*gamma)/(-1j*gamma)), feat.amplitude_coeff)
                
                feat.feat_box = Box({k: (v - box_extent,
                                         v + box_extent)
                                     for k,v in feat.location.as_dict().items()})

        return res_features

    
    @classmethod
    def apply_magn_cond_filter(cls, features: list['SpectralFeature'],
                               magn_conditions: tuple,
                               magn_conditions_margin: float):
        """
        magn_conditions:
            (('-A', 'B',),) -- when w1,w2
            (('B',),) -- when w1,w2-w1

        if None - just returns back features from input
        """
        if magn_conditions is None:
            return features
        
        res_features = []

        for feat in features:
            if magn_conditions == (('B',),):
                if feat.location['B'] > (0+magn_conditions_margin):
                    res_features.append(feat)
            elif magn_conditions == (('-A', 'B',),):
                if feat.location['B'] - feat.location['A'] > (0+magn_conditions_margin):
                    res_features.append(feat)
            else:
                raise ValueError("this magn_conditions isn't implemented")
        return res_features

    
    @classmethod
    def print_list_features(cls, features: list['SpectralFeature']):
        for feat in features:
            print('\n -- A feature at the location', feat.location, 'with featbox', feat.feat_box, 'with amplitude_coeff', feat.amplitude_coeff)
            for r in feat.rows:
                print('   ', r)


def features_from_rows(table: ContributionTable, lineshape_parameter: float | None = None) -> list[SpectralFeature]:
    """
    One feature per (motif, location). Same motif + same point -> same peak shape, so the
    coefficients add. Rows of different motifs at one point stay separate features;
    the grid adds them up as complex numbers.
    """
    groups = table.group_by(lambda r: (r.motif, r.location))
    return [SpectralFeature(location=location, rows=tuple(g), lineshape_parameter=lineshape_parameter)
            for (_, location), g in groups.items()]


def features_to_clusters(features: list['SpectralFeature']) -> dict[int, list['SpectralFeature']]:
    """
    takes features and, based on those feature boxes overlapping, returns clusters
    """
    feature_boxes = []
    for f in features:
        if f.feat_box is None:
            raise ValueError(f"No feature box for {f}")
        else:
            feature_boxes.append(f.feat_box)
    adjacency = compute_box_adjacency(feature_boxes)

    return connected_components_from_adjacency(adjacency, features)

@dataclass
class SpectralWindow:
    """
    The full spectrum range (a Box) with the features evaluated in it.
        full_features: location inside the box
        contrib_features: location outside the box, but close enough to add intensity inside it
    """
    box: 'Box'
    full_features: list['SpectralFeature'] = field(default_factory=list)
    contrib_features: list['SpectralFeature'] = field(default_factory=list)

    @property
    def bounds(self) -> dict[str, 'Dim_bounds']:
        return self.box.bounds

    @property
    def ndim(self) -> int:
        return len(self.bounds)

    # UNUSED
    @property
    def widths(self) -> dict[str, float]:
        # return tuple(mx - mn for mn, mx in self.bounds)
        return {k: (v[1]-v[0]) for k, v in self.bounds.items()}


# tuple[list[np.ndarray], dict[str, np.ndarray]]
    def sample_grid(self, dim_sizes: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Generate a regular grid of points spanning the window.
        Each axis gets dim_sizes[ax] points from min to max, both edges included: step = (max - min) / (n - 1).
        """

        if set(dim_sizes) != set(self.box.axes):
            raise ValueError(f"Expected one grid size per window axis {self.box.axes}, got sizes for {tuple(sorted(dim_sizes))}")
        axes = {}
        for ax in self.bounds:
            mn, mx = self.bounds[ax]
            axes[ax] = np.linspace(mn, mx, dim_sizes[ax])

        coords_vectors = list(axes.values())
        grid = np.meshgrid(*coords_vectors, indexing="ij")
        grid_d = {ax: grid[i] for i, ax in enumerate(axes)}
        
        return axes, grid_d


    def find_clusters_by_featboxes(self) -> tuple['RectangularDomain', ...]:
        all_features = self.full_features + self.contrib_features

        clusters = features_to_clusters(features=all_features)

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
        feat = SpectralFeature.get_max_intensity_feat(self.full_features+self.contrib_features)
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
    from_features makes one domain per cluster: its box spans the boxes of the cluster's features.
    """
    box: Box
    full_features: list['SpectralFeature'] = field(default_factory=list)
    contrib_features: list['SpectralFeature'] = field(default_factory=list)
        

    def __hash__(self):
        return hash((self.box, tuple(self.full_features), tuple(self.contrib_features)))
    
    # -----------------------------------------------------------
    # Feature utilities
    # -----------------------------------------------------------
    # UNUSED by extention
    def add_full_features(self, features: list['SpectralFeature']):
        self.full_features.extend(features)

    # UNUSED
    def add_a_full_feature(self, feature: 'SpectralFeature'):
        self.full_features.append(feature)

    # UNUSED
    def add_contrib_features(self, features: list['SpectralFeature']):
        self.contrib_features.extend(features)

    # UNUSED
    def add_a_contrib_feature(self, feature: 'SpectralFeature'):
        self.contrib_features.append(feature)

    # UNUSED - so far? useful for analysis?
    # def features_in_bounds(self) -> list['SpectralFeature']:
    #     """Return features whose coordinates lie inside the domain's window."""
    #     if not self.full_features:
    #         return []
    #     coords = np.array([f.location.as_array() for f in self.full_features])
    #     mask = self.box.contains(coords)
    #     return [f for f, m in zip(self.full_features, mask) if m]


    @classmethod
    def from_features(cls, features: list[SpectralFeature]):
        feats = []
        for f in features:
            if f.feat_box is None:
                raise ValueError(f'feature {f} does not have feat_box')
            else:
                feats.append(f.feat_box)
        return cls(
            box=Box.union(feats),
            full_features=features
        )
