"""
Spectral features, and the windows and domains that hold them.

  in:  ContributionRow tables (built by evaluate.build_contributions), propagator parameters (propagator_param)
  out: SpectralFeature lists, SpectralWindow, RectangularDomain clusters

  holds: ContributionRow, ContributionTable, SpectralFeature, features_from_rows, features_to_clusters,
         SpectralWindow, RectangularDomain

Takes Box, box grids and box clustering from grid.py; grid.py never imports from here.
"""
import copy
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
    compute_box_adjacency,
    connected_components_from_adjacency,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResonanceMotif,
)
from wilson_suite.wilson_system.system_data import VibDiff
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
        Dim_bounds,
    )
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
        ResLocPoint,
    )
    from wilson_suite.wilson_system.system_data import VibStatesData

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

# propagator_param, cm-1: the propagator's parameter (for the Lorentzian: gamma, the half width)
#   a number   -> the same on every resonance condition, everywhere
#   a function -> (vd, freqs) -> cm-1, called once per condition
#       vd     the condition's pair of states (VibDiff), e.g. vd.energy_difference(), vd.left.state_label
#       freqs  {axis: cm-1}: the grid when drawing (arrays of the grid's shape), the feature's location
#              for its box and peak height (numbers, the feature's own axes only)
#       e.g. lambda vd, freqs: 0.01 * abs(vd.energy_difference())    one width per condition
#            lambda vd, freqs: 2. + 0.002 * np.abs(freqs['A'])        changes with the frequency on A
#       abs(): energy_difference is signed (left - right). A negative width conjugates the feature.
#   A function needs the feature's conditions (features_from_rows with the states).
PropagatorParamFunction = Callable[[VibDiff, dict[str, float | np.ndarray]], float | np.ndarray]
PropagatorParam = float | PropagatorParamFunction


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
    location and propagator_param are in cm-1; propagator_param is a number or a function (PropagatorParam).
    feat_box: None until SpectralWindow.from_features gives the feature its box, so drawing or clustering a feature
        without one raises.
    conditions: the pair of states of each resonance condition, in motif order, from rows[0].params (as
        evaluate.eval_feature_on_grid). features_from_rows sets them for a propagator_param function, which
        needs them for the box and the peak height (peak_widths). Not part of == or hash: the rows fix them.
    """
    location: 'ResLocPoint'
    rows: tuple[ContributionRow, ...] = ()
    propagator_param: PropagatorParam | None = None
    scale: float = 1.0      # normalize_coeffs_to_max changes this, never the rows
    feat_box: Box | None = None
    conditions: tuple[VibDiff, ...] | None = None

    @property
    def amplitude_coeff(self) -> float | None:
        """scale * sum of the row coefficients; None for a feature without rows."""
        if not self.rows:
            return None
        return self.scale * sum(r.coeff for r in self.rows)

    @property
    def net_fraction(self) -> float | None:
        """
        How much of the rows is left in their sum: |sum of coeffs| / sum of |coeffs|. scale does not change it.
            1: no cancellation (one row, or rows of one sign);  0: the rows cancel
            e.g. rows 0.5, -0.4 -> 0.1 / 0.9 = 0.11;  rows 0.5, -0.5 -> 0
        None without rows, or if every row is 0 (nothing cancels there).
        """
        total = sum(abs(r.coeff) for r in self.rows)
        if total == 0.:
            return None
        return abs(sum(r.coeff for r in self.rows)) / total

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
        return hash((self.location, self.propagator_param, self.rows, self.scale))

    def __eq__(self, other) -> bool:
        if not isinstance(other, SpectralFeature):
            return False
        return (self.location == other.location
                and self.propagator_param == other.propagator_param
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

    @classmethod
    def get_max_intensity_feat(cls, features: list['SpectralFeature'],
                          intensity_expr: str | None = 'abs()**2') -> 'SpectralFeature':
        """
        amplitude of a feature is given by: amplitude_coeff / propagator_param**2
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
    

    def peak_widths(self) -> tuple[float, ...]:
        """
        propagator_param at the peak, cm-1, one per resonance condition: what get_intensity and the boxes use.
            a number gamma -> (gamma, gamma, ...), one per axis (as many as conditions, B7)
            a function     -> propagator_param(vd, location) for each condition: needs `conditions`
        For a function that changes with the frequency, this is the width at the peak, not farther out.
            e.g. lambda vd, freqs: 0.01 * abs(vd.energy_difference()), conditions with E = 1600 and 500 -> (16., 5.)
        """
        param = self.propagator_param
        if param is None:
            raise ValueError(f'feature {self} has no propagator_param')
        if not callable(param):
            return (param,) * len(self.location.axes)
        if self.conditions is None:
            raise ValueError(f'feature {self} has a propagator_param function but no conditions '
                             f'(features_from_rows gives them when it gets the states)')
        at_peak = self.location.as_dict()
        return tuple(float(param(vd, at_peak)) for vd in self.conditions)

    def get_intensity(self, intensity_expr: str = 'abs()**2') -> float:
        """
        |peak amplitude|^2 in au. At the peak every condition gives 1 / (-i * gamma_i), so
            |amplitude_coeff / ((-i * gamma_1) * (-i * gamma_2) * ...)|^2
        gamma_i: peak_widths, cm-1 (converted here). One number -> amplitude_coeff / (-i * gamma)^N.
        """

        if self.amplitude_coeff is None or self.propagator_param is None:
            raise ValueError('this feature has no amplitude_coeff and/or propagator_param')
        widths = self.peak_widths()
        if any(gamma <= 0 for gamma in widths):
            raise ValueError(f'propagator_param must be > 0, got {widths} at the peak')

        if intensity_expr == 'abs()**2':
            peak = complex(self.amplitude_coeff)
            for gamma in widths:
                peak /= -1j * convNu2Ene(gamma)
            return abs(peak)**2
        else:
            raise NotImplementedError("Only standard 'abs()**2' expression is implemented.")
    
    @classmethod
    def print_list_features(cls, features: list['SpectralFeature']):
        for feat in features:
            print('\n -- A feature at the location', feat.location, 'with featbox', feat.feat_box, 'with amplitude_coeff', feat.amplitude_coeff)
            for r in feat.rows:
                print('   ', r)


def features_from_rows(table: ContributionTable, propagator_param: PropagatorParam | None = None,
                       states: 'VibStatesData | None' = None) -> list[SpectralFeature]:
    """
    One feature per (motif, location). Same motif + same point -> same peak shape, so the
    coefficients add. Rows of different motifs at one point stay separate features;
    the grid adds them up as complex numbers.

    states: for a propagator_param function, each feature gets its conditions (condition_states), which
        the function needs for the box and the peak height. A number needs no states and gets no conditions.
    """
    groups = table.group_by(lambda r: (r.motif, r.location))
    with_conditions = callable(propagator_param) and states is not None
    return [SpectralFeature(location=location, rows=tuple(g), propagator_param=propagator_param,
                            conditions=condition_states(motif, next(iter(g)).params, states) if with_conditions else None) # type: ignore
            for (motif, location), g in groups.items()]


def condition_states(motif: ResonanceMotif, params: ParameterSet, states: 'VibStatesData') -> tuple[VibDiff, ...]:
    """
    The pair of states of each condition of `motif` at `params`, in motif order (as evaluate.eval_resonance_factor).
        e.g. conditions <ab - a> and <b - a> at a=0, b=1: states ('0,1', '0') and ('1', '0')
    """
    return tuple(VibDiff.from_quanta(*key.diff, params.to_dict(), states) for key in motif)


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


def _check_drawable(f: SpectralFeature):
    """raises ValueError for a feature that cannot get a box or be drawn: no propagator_param or no rows"""
    if f.propagator_param is None:
        raise ValueError(f'feature {f} has no propagator_param, so no box size')
    if f.amplitude_coeff is None:
        raise ValueError(f'feature {f} has no rows, so no amplitude')


def amplitude_in_window(f: SpectralFeature, window: Box) -> float:
    """
    The feature's largest amplitude inside `window`, au, without drawing.
        inside:   its peak amplitude, sqrt(get_intensity())
        outside:  its amplitude at the nearest point of the window, the Lorentzian of each axis it lies outside on,
                  e.g. 100 cm-1 outside on A, gamma 5: peak * 5 / hypot(100, 5) = peak * 0.05
    gamma: the largest of its peak_widths, as for its box.
    """
    amplitude, gamma = np.sqrt(f.get_intensity()), max(f.peak_widths())
    for ax, x in f.location.as_dict().items():
        mn, mx = window.bounds[ax]
        amplitude *= gamma / np.hypot(max(mn - x, 0., x - mx), gamma)
    return float(amplitude)


# A box rule: (window, features) -> the box half-width of each feature, cm-1, in the order of `features`
# (the input of SpectralWindow.from_features; draw_spectrum takes the rule). Two rules: amplitude_half_widths and
# intensity_half_widths. functools.partial gives a rule its knobs, e.g. partial(amplitude_half_widths, dynrange=500.)
BoxRule = Callable[[Box, Sequence[SpectralFeature]], Sequence[float]]


def amplitude_half_widths(window: Box, features: Sequence[SpectralFeature], *,
                          dynrange: float,
                          eps: float = 0.1,
                          min_gammas: float = 1.,
                          max_cm: float | None = None,
                          a_max: float | None = None) -> list[float]:
    """
    The box half-width of each feature, cm-1, in the order of `features`, from its amplitude (todo.txt E10):
    strong features get big boxes, weak ones small. A box rule (BoxRule).

    Why: the plot shows intensities down to its floor, a_max**2 / dynrange. Drawing adds amplitudes, then squares,
    so a tail t cut off where it meets an amplitude A changes the intensity by about 2 * A * t, much more than t**2.
    The rule takes the worst case, A = a_max, and keeps that change below eps * floor:
        k          = 2 * dynrange * a / (eps * a_max)     a: the feature's peak amplitude, sqrt(get_intensity())
        half-width = gamma * sqrt(k**2 - 1)               there its Lorentzian tail, a * gamma / sqrt(d**2 + gamma**2),
                                                          is down to eps * a_max / (2 * dynrange)
        then at least min_gammas * gamma, and at most max_cm
    gamma: the largest of the feature's peak_widths, on every axis. With one width for every feature,
    a / a_max = |coeff| / max |coeff|.
        e.g. gamma 4.7, dynrange 500, eps 0.1:  a / a_max = 1 -> 47000 cm-1 (the whole window), 0.1 -> 4700,
             0.01 -> 470, 0.001 -> 47, 1e-4 or less -> 4.7 (1 gamma)

    The knobs, to tune against draw_all (tests/e10_boxes.py):
        dynrange    the plot's dynamic range, max / floor, e.g. 500. Required: it says what the plot shows.
        eps         the allowed error, as a fraction of the floor. Smaller: bigger boxes, more accurate, slower.
                    Measured 2026-10-14 (formaldehyde, ACDM, dynrange 100-1000): eps 0.1 -> at most 0.13 floors,
                    the worst visible point 2.2 % off; eps 0.3 -> 0.47 floors, 4.9 %.
        min_gammas  the smallest half-width, in gammas: how much of a weak peak is drawn. Its tail beyond
                    k gammas is below the error anyway; more only changes how a weak peak looks. Default 1.
        max_cm      the largest half-width, cm-1; None (default): no cap. A cap bounds the cost of the strong
                    features, but then eps no longer bounds the error.
        a_max       the strongest amplitude in the window, au; None (default): estimated without drawing, the
                    largest amplitude_in_window of `features` (0.4-10 % from the true max, todo.txt E10).
                    Pass it e.g. to give two windows the same boxes, or to try the true max of a drawing.
    Every feature needs rows and a propagator_param > 0. A run whose features all have amplitude 0 gets the
    smallest boxes: it draws nothing.
    """
    if not dynrange > 1.:
        raise ValueError(f'dynrange must be > 1, got {dynrange}')
    if not eps > 0.:
        raise ValueError(f'eps must be > 0, got {eps}')
    if not min_gammas >= 0.:
        raise ValueError(f'min_gammas must be >= 0, got {min_gammas}')
    if max_cm is not None and not max_cm > 0.:
        raise ValueError(f'max_cm must be > 0 or None, got {max_cm}')
    if a_max is not None and not a_max > 0.:
        raise ValueError(f'a_max must be > 0 or None, got {a_max}')
    if not features:
        return []

    for f in features:
        _check_drawable(f)
    if a_max is None:
        a_max = max(amplitude_in_window(f, window) for f in features)

    half_widths = []
    for f in features:
        gamma = max(f.peak_widths())
        k = 2. * dynrange * np.sqrt(f.get_intensity()) / (eps * a_max) if a_max > 0. else 0.
        # gamma * sqrt(k**2 - 1), 0 for k <= 1: where the Lorentzian intensity is down to 1/k**2 of its peak
        half_width = max(lorentzian_distance_to_dynrange_weaker_than_max(gamma, k ** 2), min_gammas * gamma)
        half_widths.append(half_width if max_cm is None else min(half_width, max_cm))
    return half_widths


def intensity_half_widths(window: Box, features: Sequence[SpectralFeature], *,
                          dynrange: float,
                          box_range_safety_margin: float = 0.1,
                          scale_wrt_max_intensity: bool = False,
                          minimum_box_padding: float = 0.,
                          max_intensity: float | None = None) -> list[float]:
    """
    The box half-width of each feature, cm-1, in the order of `features`: the old code's rule,
    SpectralFeature.dress_these_with_boxes (until 2026-10-14), as a box rule (BoxRule). The old knobs keep their
    names, meanings and defaults.

    dynrange: the old function took max_intensity and min_intensity, and computed
    implied_dynrange = max_intensity / min_intensity. Both old callers (SpectralWindow.dress_with_featboxes,
    evaluation_wf.dress_features_with_boxes) set min_intensity = max_intensity / dynamic_range, so
    implied_dynrange is the plot's dynamic range: `dynrange` here. min_intensity = max_intensity / dynrange.

    Per feature, as in the old function (gamma: the largest of its peak_widths, on every axis; i: its peak
    intensity, get_intensity()):
        unscaled (default):  d = gamma * sqrt(dynrange - 1)                      down to 1/dynrange of its own peak
        scaled:              d = gamma * sqrt(dynrange * i / max_intensity - 1)   down to min_intensity, the floor;
                                                                                  0 for a feature at or below it
        half-width (the old box_extent) = max(d * (1 + box_range_safety_margin), minimum_box_padding)
        e.g. gamma 5, dynrange 101, margin 0: unscaled 50 cm-1 for every feature; scaled 50 cm-1 for the
             strongest, 0 (or minimum_box_padding) for a feature 101 times weaker or weaker still

    The knobs:
        dynrange                 the plot's dynamic range, max / floor, e.g. 100. Required.
        box_range_safety_margin  the boxes this much larger, e.g. 0.1 (default): 10 % larger.
        scale_wrt_max_intensity  False (default): each box to 1/dynrange of its own peak. True: to the floor.
        minimum_box_padding      the smallest half-width, cm-1. Default 0. (The old setup class passed 10.)
        max_intensity            used only when scaled. None (default): the strongest peak intensity of the
                                 features inside `window`, as the old callers took it.
    The old settings' defaults (wilson_main, spectrum_abstractions.py): dynamic_range 100, margin 0.1, unscaled,
    minimum_box_padding 10.

    Changed from dress_these_with_boxes:
        - No feature is dropped (decided 2026-10-09). The old function dropped every feature below the floor
          (i < min_intensity) when minimum_box_padding was 0, scaled or not. Here such a feature keeps its box as
          above: unscaled the usual one; scaled, half-width minimum_box_padding (with 0: drawn only on the grid
          points around its peak). Dropping is not offered for now; it could come back as a knob.
        - No error for a feature stronger than max_intensity: a stronger feature outside the window is normal
          (ACDM, quick_start window: 4 of them). Scaled, it gets a box larger than the unscaled one.
        - Returns half-widths, not copies of the features with boxes: SpectralWindow.from_features makes those.

    Cuts more than amplitude_half_widths: a box ends by the feature's own intensity, but the cut tail adds to the
    amplitude of its neighbours (see amplitude_half_widths). Measured 2026-10-14 (todo.txt E10), unscaled, each
    feature on its own box: 1.7-12 floors off; amplitude_half_widths with eps 0.1: at most 0.13 floors.
    Every feature needs rows and a propagator_param > 0. Scaled, without max_intensity, at least one feature must
    lie inside `window` (the old callers raised too).
    """
    if not dynrange > 1.:
        raise ValueError(f'dynrange must be > 1, got {dynrange}')
    if not box_range_safety_margin >= 0.:
        raise ValueError(f'box_range_safety_margin must be >= 0, got {box_range_safety_margin}')
    if not minimum_box_padding >= 0.:
        raise ValueError(f'minimum_box_padding must be >= 0, got {minimum_box_padding}')
    if max_intensity is not None and not max_intensity > 0.:
        raise ValueError(f'max_intensity must be > 0 or None, got {max_intensity}')
    if not features:
        return []

    for f in features:
        _check_drawable(f)
    i_max = 0.
    if scale_wrt_max_intensity:
        if max_intensity is not None:
            i_max = max_intensity
        else:
            inside = [f.get_intensity() for f in features if f.is_inside(window)]
            if not inside:
                raise ValueError('scaled boxes need the strongest peak inside the window, and no feature is inside: '
                                 'pass max_intensity')
            i_max = max(inside)

    half_widths = []
    for f in features:
        gamma = max(f.peak_widths())
        if scale_wrt_max_intensity:
            # implied_dynrange * feat_intensity_wrt_max, as in the old function; 0 if every peak inside is 0
            k = dynrange * f.get_intensity() / i_max if i_max > 0. else 0.
        else:
            k = dynrange
        # gamma * sqrt(k - 1), 0 for k <= 1: where the Lorentzian intensity is down to 1/k of its peak
        box_extent = lorentzian_distance_to_dynrange_weaker_than_max(gamma, k) * (1. + box_range_safety_margin)
        half_widths.append(max(box_extent, minimum_box_padding))
    return half_widths


@dataclass
class SpectralWindow:
    """
    The full spectrum range (a Box) with the features evaluated in it (made by from_features).
        full_features: location inside the box
        contrib_features: location outside the box, but the feature's box overlaps it
    """
    box: 'Box'
    full_features: list['SpectralFeature'] = field(default_factory=list)
    contrib_features: list['SpectralFeature'] = field(default_factory=list)

    @classmethod
    def from_features(cls, box: Box, features: Sequence['SpectralFeature'],
                      half_widths: Sequence[float]) -> 'SpectralWindow':
        """
        The window over `box` with the features that add amplitude inside it, each with its box. In this order:
          1. a box around every feature: its location +- its half-width, the same on every axis
             half_widths: cm-1, one per feature, in the order of `features`, e.g. from a box rule (BoxRule):
             amplitude_half_widths or intensity_half_widths. The window does not care how they were chosen.
          2. keep the features whose box overlaps `box` (touching edges excluded):
             location inside `box` -> full_features, outside -> contrib_features
          No feature is dropped for its amplitude (decided 2026-10-09; the old code kept them too, C3):
            - a weak feature stays, however weak
            - a term with coefficient 0 never becomes a row (build_contributions puts it in `zero`)
            - a feature whose rows cancel (amplitude 0) stays as a record of it; it draws nothing

        e.g. half-width 55 cm-1, `box` A (0, 10): a feature at A = 60 is kept (its box starts at 5), a feature at
        A = 70 is not (its box starts at 15).

        Returns copies with boxes; the input features stay as they are. Every feature needs rows and a
        propagator_param: drawing needs both.
        """
        if len(half_widths) != len(features):
            raise ValueError(f'one half-width per feature: {len(half_widths)} half-widths, {len(features)} features')

        dressed = []
        for f, half_width in zip(features, half_widths):
            _check_drawable(f)
            if not half_width >= 0.:
                raise ValueError(f'half-width must be >= 0, got {half_width} for {f}')
            feat_box = Box({k: (v - half_width, v + half_width) for k, v in f.location.as_dict().items()})
            dressed.append(replace(f, feat_box=feat_box))

        kept = [f for f in dressed if f.feat_box_overlaps(box)]

        return cls(box,
                   full_features=[f for f in kept if f.is_inside(box)],
                   contrib_features=[f for f in kept if not f.is_inside(box)])

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

    def sample_grid(self, dim_sizes: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Regular grid over the window box, see Box.make_grid.
        """
        return self.box.make_grid(dim_sizes)


    def find_clusters_by_featboxes(self) -> tuple['RectangularDomain', ...]:
        all_features = self.full_features + self.contrib_features

        clusters = features_to_clusters(features=all_features)

        return tuple(
            RectangularDomain.from_features(clusters[c])
            for c in clusters
        )


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
