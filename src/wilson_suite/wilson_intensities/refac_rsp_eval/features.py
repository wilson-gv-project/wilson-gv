"""
Spectral features, and the windows that hold them.

  in:  ContributionRow tables (evaluate.build_contributions), line-shape parameters
       (PropagatorParam)
  out: SpectralFeature lists, SpectralWindow

  holds:
    ContributionRow, ContributionTable   one row per (term, index set); filter, group rows
    SpectralFeature, features_from_rows  one resonance per (motif, location), from the rows
    amplitude_half_widths,               box rules (BoxRule): the size of each feature's box
      intensity_half_widths
    SpectralWindow                       the spectrum range with the features reaching in

What a box decides depends on cut_tails in pipeline.spectrum_on_grid:
    cut_tails=False (default)  whether the feature is kept: its box must reach the window.
                               A kept feature is evaluated on the whole grid.
    cut_tails=True             also where it is evaluated: only inside its box, so its tail
                               beyond the box is cut off.
Either way a box leaves out the feature's amplitude beyond its edge where that amplitude
lies in the window: all of it for a dropped feature, the tail for a cut one.

The peak heights (get_intensity), the amplitude reaching into a window (amplitude_in_window)
and the box sizes take the line shape from lineshape.py, as the grid does.

The whole flow, from terms to the plot, is drawn in the pipeline.py docstring.
Takes Box from grid.py and the line shape from lineshape.py; neither imports from here.
"""
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval import lineshape
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResonanceMotif,
)
from wilson_suite.wilson_system.system_data import VibDiff
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
        ResLocPoint,
    )
    from wilson_suite.wilson_system.system_data import VibStatesData

# The type of the keys ContributionTable.group_by groups by: whatever its `key` function
# returns, e.g. int for lambda r: r.term_id. Hashable, since the keys become dict keys.
GroupKey = TypeVar('GroupKey', bound=Hashable)

# propagator_param, cm-1: the parameter of each condition's line shape (lineshape.py),
# e.g. for the Lorentzian its half width gamma. Either:
#   a. a number: the same width for every resonance condition, everywhere
#   b. a function width(vd, freqs) -> cm-1, called once per condition. Its inputs:
#      - vd: the condition's pair of states (VibDiff), e.g. vd.left.state_label,
#        vd.energy_difference()
#      - freqs: {axis: cm-1}, the frequencies where the width is needed. On the grid: the
#        grid points (grid.Mesh, one array per axis). For the box and the peak height: the
#        feature's location (one number per axis, its own axes only).
#
# Examples of a function:
#   1. one width per condition, from its energy difference:
#      lambda vd, freqs: 0.01 * abs(vd.energy_difference())
#   2. Gamma_mn by the pair of states (m, n), e.g. gamma_mn = {('zero', '4'): 5., ...}:
#      lambda vd, freqs: gamma_mn[vd.left.state_label, vd.right.state_label]
#   3. a width that changes along axis A:
#      lambda vd, freqs: 2. + 0.002 * np.abs(freqs['A'])
#
# For a function:
#   - use abs() on energy_difference: it is signed (left - right), and a negative width
#     gives the complex conjugate of the feature
#   - pass the states to features_from_rows: the function needs each feature's conditions
#   - a feature takes its pairs of states from its first row (SpectralFeature.conditions)
PropagatorParamFunction = Callable[[VibDiff, dict[str, float | np.ndarray]], float | np.ndarray]
PropagatorParam = float | PropagatorParamFunction


@dataclass(frozen=True)
class ContributionRow:
    """
    One term evaluated at one index set.
        term_id   the term's position in the term list: the term is terms[term_id]
        motif     the term's resonance conditions
        params    the index set: one mode per resonance label, e.g. {'a': 0, 'b': 1}
        location  where the conditions resonate, cm-1 (one coordinate per axis)
        coeff     the term's coefficient at this index set, summed over its other labels
    """
    term_id: int
    motif: ResonanceMotif
    params: ParameterSet
    location: 'ResLocPoint'
    coeff: float


class ContributionTable:
    """
    A read-only list of ContributionRows, with tools to filter rows (where) and group them
    (group_by). Both return new tables; the rows are never changed.
    """
    def __init__(self, rows: Iterable[ContributionRow]):
        self._rows = tuple(rows)

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def where(self, keep: Callable[[ContributionRow], bool]) -> 'ContributionTable':
        """
        The rows for which keep(row) is True, e.g. table.where(lambda r: r.term_id == 3).
        """
        return ContributionTable(r for r in self._rows if keep(r))

    def group_by(self, key: Callable[[ContributionRow], GroupKey]
                 ) -> dict[GroupKey, 'ContributionTable']:
        """
        {key(row): table of the rows with that key}, keys in the order they first appear,
        e.g. table.group_by(lambda r: r.term_id).
        """
        groups = defaultdict(list)
        for r in self._rows:
            groups[key(r)].append(r)
        return {k: ContributionTable(v) for k, v in groups.items()}


@dataclass
class SpectralFeature:
    """
    One resonance: the rows of one motif at one location (made by features_from_rows).
    Same motif + same location -> the same resonance factor on the grid, so the coefficients
    of the rows add up. Not always one visible peak: features of other motifs at the same
    location add to it on the grid, its rows can cancel, and a flat feature has no peak.

        location          where the conditions resonate, cm-1 (one coordinate per axis)
        rows              the ContributionRows of this resonance
        propagator_param  the parameter of each condition's line shape (lineshape.py),
                          cm-1, e.g. the Lorentzian's half width gamma: a number or a
                          function (PropagatorParam)
        feat_box         the feature's box: it keeps the feature if it reaches the window,
                          and with cut_tails=True it is where the feature is evaluated (see
                          the module docstring). None until SpectralWindow.from_features
                          gives the feature its box.
        conditions        the pair of states of each resonance condition, in motif order,
                          at rows[0].params (condition_states). features_from_rows sets
                          them when propagator_param is a function, which needs them for
                          the box and the peak height (peak_widths). The grid uses the same
                          pairs. The rows at one location share their pairs, unless two
                          energy differences are exactly equal (e.g. two modes of the same
                          energy): then every row gets the first row's pairs, and so its
                          width, e.g. its Gamma_mn.

    == compares location, rows and propagator_param. It leaves out conditions (the rows fix
    them) and feat_box (the same feature gets another box in another window).

    Rows without resonance conditions make a flat background (is_flat): no peak, the same
    amplitude on every grid point.
    """
    location: 'ResLocPoint'
    rows: tuple[ContributionRow, ...] = ()
    propagator_param: PropagatorParam | None = None
    feat_box: Box | None = field(default=None, compare=False)
    conditions: tuple[VibDiff, ...] | None = field(default=None, compare=False)

    @property
    def amplitude_coeff(self) -> float | None:
        """The sum of the row coefficients; None without rows."""
        if not self.rows:
            return None
        return sum(r.coeff for r in self.rows)

    @property
    def net_fraction(self) -> float | None:
        """
        How much of the rows survives in their sum, i.e. how much they cancel each other:
        |sum of coeffs| / sum of |coeffs|.
            1  no cancellation (one row, or rows of one sign)
            0  the rows cancel completely
            e.g. rows 0.5, -0.4 -> 0.1 / 0.9 = 0.11;  rows 0.5, -0.5 -> 0
        None without rows, or if every row is 0 (nothing to cancel).
        """
        total = sum(abs(r.coeff) for r in self.rows)
        if total == 0.:
            return None
        return abs(sum(r.coeff for r in self.rows)) / total

    @property
    def motif(self) -> ResonanceMotif | None:
        """
        The motif shared by all rows; None without rows.
        Raises ValueError if the rows have different motifs.
        """
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
        """The term_id of the rows, each once, in row order."""
        return tuple(dict.fromkeys(r.term_id for r in self.rows))

    @property
    def is_flat(self) -> bool:
        """
        True for a feature that is its coefficient only: the rows come from terms
        without resonance conditions, so there is no resonance factor, and the location
        has no axes. On the grid it is a flat background, amplitude_coeff on every grid
        point: no peak, no width, no box of its own (SpectralWindow.from_features gives
        it the whole window).
            e.g. an IR absorption term such as 1/4 * polhess_(a,a) / <a>
        """
        return not self.location.axes

    def __repr__(self) -> str:
        return f'SpectralFeature(location={self.location}, rows={len(self.rows)}, amplitude_coeff={self.amplitude_coeff})'

    def _check_same_axes(self, box: Box):
        if self.location.axes != box.axes:
            raise ValueError(f"Expected SpectralFeature with axes {box.axes}, got {self.location.axes}")

    def is_inside(self, box: Box) -> bool:
        """
        True if the feature's location lies inside `box`, edges included.
        The axes must match.
        """
        self._check_same_axes(box)
        return box.contains_point(self.location.as_dict())

    def feat_box_overlaps(self, box: Box) -> bool:
        """
        True if the feature's own box (feat_box) overlaps `box`, touching included
        (a shared face, edge or corner; see Box.overlaps). The axes must match.
        Raises ValueError if the feature has no box yet.
        """
        self._check_same_axes(box)

        if self.feat_box is None:
            raise ValueError('Need to add a box for this feature')
        return box.overlaps(self.feat_box)

    def peak_widths(self) -> tuple[float, ...]:
        """
        The line-shape parameter of each resonance condition, cm-1, e.g. the Lorentzian's
        half width gamma, taken at the feature's location, where every condition resonates
        (the feature's peak). Each condition has its own line shape, whichever axes it uses
        (e.g. A + B).
            a number gamma  -> (gamma, gamma, ...), one per condition
            a function      -> propagator_param(vd, location) for each condition;
                               needs `conditions`
            a flat feature  -> (): no conditions
        A function that changes with the frequency gives the value at the location,
        not farther out.
            e.g. lambda vd, freqs: 0.01 * abs(vd.energy_difference()),
                 conditions with E = 1600 and 500 -> (16., 5.)
        """
        param = self.propagator_param
        if param is None:
            raise ValueError(f'feature {self} has no propagator_param')
        if self.is_flat:
            return ()
        if not callable(param):
            return (float(param),) * len(self.motif or ())
        if self.conditions is None:
            raise ValueError(f'feature {self} has a propagator_param function but no conditions '
                             f'(features_from_rows gives them when it gets the states)')
        at_peak = self.location.as_dict()
        return tuple(float(param(vd, at_peak)) for vd in self.conditions)

    def get_intensity(self) -> float:
        """
        The intensity at the peak, |peak amplitude|^2, in au (atomic units): amplitude_coeff
        times each condition's line shape at resonance (lineshape.peak), squared.
            intensity = |amplitude_coeff * peak(gamma_1) * peak(gamma_2) * ...|^2
        gamma_i: peak_widths, cm-1 (converted to au here). For the Lorentzian,
        peak(gamma) = 1 / (-i * gamma).
        A flat feature (is_flat) has no conditions, so no resonance factor: its amplitude is
        its coefficient only, and its intensity |amplitude_coeff|^2 on every grid point.
        Raises ValueError if a width is <= 0.
        """

        if self.amplitude_coeff is None or self.propagator_param is None:
            raise ValueError('this feature has no amplitude_coeff and/or propagator_param')
        widths = self.peak_widths()
        if any(gamma <= 0 for gamma in widths):
            raise ValueError(f'propagator_param must be > 0, got {widths} at the peak')

        peak = complex(self.amplitude_coeff)
        for gamma in widths:
            peak *= lineshape.peak(convNu2Ene(gamma))
        return abs(peak)**2


def features_from_rows(table: ContributionTable, propagator_param: PropagatorParam | None = None,
                       states: 'VibStatesData | None' = None) -> list[SpectralFeature]:
    """
    Groups the rows into features, one per (motif, location). Same motif + same location ->
    the same resonance factor, so the coefficients add up. Rows of different motifs at one
    location stay separate features, because their resonance factors differ; on the grid
    their amplitudes add up as complex numbers.

    propagator_param - the line-shape parameter given to every feature (PropagatorParam)
    states           - needed when propagator_param is a function: each feature then gets
                       its conditions (condition_states), which the function needs for the
                       box and the peak height. Without states such a feature has no
                       conditions, and peak_widths raises. A number needs no states and
                       gets no conditions.
    """
    groups = table.group_by(lambda r: (r.motif, r.location))
    with_conditions = callable(propagator_param) and states is not None
    return [SpectralFeature(location=location, rows=tuple(g), propagator_param=propagator_param,
                            conditions=condition_states(motif, next(iter(g)).params, states) if with_conditions else None) # type: ignore
            for (motif, location), g in groups.items()]


def condition_states(motif: ResonanceMotif, params: ParameterSet, states: 'VibStatesData') -> tuple[VibDiff, ...]:
    """
    For each resonance condition of `motif`, its two states as a VibDiff (left, right),
    with the mode labels set to the modes in `params`. In motif order.
        e.g. the conditions <ab - a> (state ab against state a) and <b - a>, at a=0, b=1:
            <ab - a>  ->  left '0,1' (one quantum in mode 0 and one in mode 1), right '0'
            <b - a>   ->  left '1', right '0'
    The grid builds the same pairs (evaluate.eval_resonance_factor).
    """
    return tuple(VibDiff.from_quanta(*key.diff, params.to_dict(), states) for key in motif)


def _check_has_rows_and_width(f: SpectralFeature):
    """
    Raises ValueError for a feature that cannot get a box or go on the grid:
    no propagator_param, or no rows.
    """
    if f.propagator_param is None:
        raise ValueError(f'feature {f} has no propagator_param, so no box size')
    if f.amplitude_coeff is None:
        raise ValueError(f'feature {f} has no rows, so no amplitude')


def amplitude_in_window(f: SpectralFeature, window: Box) -> float:
    """
    How strong the feature is inside `window`: its largest amplitude there, au, estimated
    without a grid. The largest of these over all features estimates the window's strongest
    amplitude, a_max. a_max sets the plot's floor, the faintest intensity the plot shows,
    a_max**2 / dynrange (amplitude_half_widths).
        peak inside   its peak amplitude, sqrt(get_intensity())
        peak outside  its amplitude at the nearest point of the window: the peak
                      amplitude times the line shape's tail, lineshape.tail(d, gamma), for
                      each axis it lies outside on (d: the distance to the window on that
                      axis). For the Lorentzian the tail is gamma / sqrt(d**2 + gamma**2),
                      e.g. 100 cm-1 outside on A, gamma 5:
                      peak * 5 / hypot(100, 5) = peak * 0.05
        flat          |amplitude_coeff|, the same on every grid point (is_flat)
    gamma: the largest of its peak_widths, as for its box.
    The estimate takes each feature alone; on the grid, features add up or cancel.
    """
    amplitude = np.sqrt(f.get_intensity())
    if f.is_flat:
        return float(amplitude)
    gamma = max(f.peak_widths())
    # on each axis of the location: the peak amplitude times the line shape's tail
    # for the distance from the peak to the window on that axis
    for ax, x in f.location.as_dict().items():
        mn, mx = window.bounds[ax]
        # d: mn - x below the window, x - mx above it, 0 inside its range
        d = max(mn - x, 0., x - mx)
        # 1 at d = 0; for the Lorentzian gamma / sqrt(d**2 + gamma**2), about gamma / d far away
        amplitude *= lineshape.tail(d, gamma)
    return float(amplitude)


# A box rule: (window, features) -> the box half-width of each feature, cm-1, in the order
# of `features`. The half-widths are the input of SpectralWindow.from_features.
# Two rules: amplitude_half_widths and intensity_half_widths.
# functools.partial sets a rule's knobs, e.g. partial(amplitude_half_widths, dynrange=500.)
BoxRule = Callable[[Box, Sequence[SpectralFeature]], Sequence[float]]


def amplitude_half_widths(window: Box, features: Sequence[SpectralFeature], *,
                          dynrange: float,
                          eps: float = 0.1,
                          min_gammas: float = 1.,
                          a_max: float | None = None) -> list[float]:
    """
    The box half-width of each feature, cm-1, in the order of `features`, chosen from its
    amplitude: strong features get big boxes, weak ones small. A box rule (BoxRule).

    Why amplitude: the plot shows intensities down to its floor, a_max**2 / dynrange. A box
    leaves out the feature's amplitude beyond its edge: all of it if the feature is dropped,
    its tail if it is cut (cut_tails=True). The grid first adds the amplitudes of all
    features, then squares the sum. So a left-out amplitude a_out, at a point where the
    other features add up to an amplitude A, changes the intensity there by about
    2 * A * a_out. That is much more than a_out**2, the change a feature alone would see.
    The rule takes the worst case, A = a_max, and keeps the change below eps * floor:
        k          = 2 * dynrange * a / (eps * a_max)
                     a: the feature's peak amplitude, sqrt(get_intensity())
        half-width = lineshape.distance(1 / k, gamma), 0 for k <= 1
                     at this distance its tail, a * lineshape.tail(d, gamma), is down to
                     a / k = eps * a_max / (2 * dynrange); for the Lorentzian
                     gamma * sqrt(k**2 - 1)
        then at least min_gammas * gamma
    gamma: the largest of the feature's peak_widths, used on every axis. When all features
    have the same width, a / a_max = |coeff| / max |coeff|.
        e.g. Lorentzian, gamma 4.7, dynrange 500, eps 0.1:  a / a_max = 1 -> 47000 cm-1,
             0.1 -> 4700, 0.01 -> 470, 0.001 -> 47, 1e-4 or less -> 4.7 (1 gamma)

    The knobs (to check a setting, compare the result with evaluate.all_features_on_grid,
    which leaves nothing out):
        dynrange    the plot's dynamic range, max / floor, e.g. 500. Required: it says what
                    the plot shows.
        eps         the allowed error, as a fraction of the floor. Smaller: bigger boxes,
                    more accurate, slower (more features kept; with cut_tails=True also
                    more grid points per box).
        min_gammas  the smallest half-width, in gammas. Default 1. The tail of a weak
                    feature beyond it is below the allowed error anyway, so a larger value
                    only changes, with cut_tails=False, whether a weak feature just outside
                    the window is kept, and with cut_tails=True, how a weak feature looks.
        a_max      the strongest amplitude in the window, au. None (default): estimated
                    without the grid, as the largest amplitude_in_window of `features`. This
                    estimate ignores how features add up on the grid, so it can differ from
                    the true maximum. Pass a_max e.g. to give two windows the same boxes, or
                    to use the true maximum of a grid result.
    Every feature needs rows and a propagator_param > 0. If every feature has amplitude 0,
    each gets the smallest box: they add nothing to the grid.
    A flat feature (is_flat) gets 0: it has no peak, and SpectralWindow.from_features gives
    it the whole window. Its amplitude does count for a_max, since it is on every grid
    point.
    """
    if not dynrange > 1.:
        raise ValueError(f'dynrange must be > 1, got {dynrange}')
    if not eps > 0.:
        raise ValueError(f'eps must be > 0, got {eps}')
    if not min_gammas >= 0.:
        raise ValueError(f'min_gammas must be >= 0, got {min_gammas}')
    if a_max is not None and not a_max > 0.:
        raise ValueError(f'a_max must be > 0 or None, got {a_max}')
    if not features:
        return []

    for f in features:
        _check_has_rows_and_width(f)
    if a_max is None:
        a_max = max(amplitude_in_window(f, window) for f in features)

    half_widths = []
    for f in features:
        if f.is_flat:
            half_widths.append(0.)
            continue
        gamma = max(f.peak_widths())
        k = 2. * dynrange * np.sqrt(f.get_intensity()) / (eps * a_max) if a_max > 0. else 0.
        # where the amplitude is down to 1/k of its peak; 0 for k <= 1
        reach = lineshape.distance(1. / k, gamma) if k > 1. else 0.
        half_widths.append(max(reach, min_gammas * gamma))
    return half_widths


def intensity_half_widths(window: Box, features: Sequence[SpectralFeature], *,
                          dynrange: float,
                          box_range_safety_margin: float = 0.1,
                          scale_wrt_max_intensity: bool = False,
                          minimum_box_padding: float = 10.,
                          max_intensity: float | None = None) -> list[float]:
    """
    The box half-width of each feature, cm-1, in the order of `features`, chosen from its
    intensity. A box rule (BoxRule).

    Per feature (gamma: the largest of its peak_widths, used on every axis;
    i: its peak intensity, get_intensity()), d: where its intensity drops to 1/k of its peak,
    lineshape.distance(1 / sqrt(k), gamma), 0 for k <= 1:
        unscaled (default)  k = dynrange: 1/dynrange of its own peak
                            (Lorentzian: d = gamma * sqrt(dynrange - 1))
        scaled              k = dynrange * i / max_intensity: the plot's floor,
                            max_intensity / dynrange; 0 for a feature at or below the floor
                            (Lorentzian: d = gamma * sqrt(dynrange * i / max_intensity - 1))
        half-width = max(d * (1 + box_range_safety_margin), minimum_box_padding)
        e.g. Lorentzian, gamma 5, dynrange 101, margin 0: unscaled 50 cm-1 for every
             feature; scaled 50 cm-1 for the strongest, 0 (or minimum_box_padding) for a
             feature 101 times weaker or weaker still

    The knobs:
        dynrange                 the plot's dynamic range, max / floor, e.g. 100. Required.
        box_range_safety_margin  makes the boxes this much larger, e.g. 0.1 (default): 10 %.
        scale_wrt_max_intensity  False (default): each box ends at 1/dynrange of the
                                 feature's own peak.
                                 True: each box ends at the plot's floor.
        minimum_box_padding      the smallest half-width, cm-1. Default 10.
        max_intensity            used only when scaled. None (default): the strongest peak
                                 intensity of the features inside `window`.

    The rule drops no feature for its strength: a feature below the floor still gets a box
    (scaled: half-width minimum_box_padding). With 0, the feature is kept only if its peak
    lies in the window, and with cut_tails=True evaluated only on the grid points around
    its peak.
    A feature stronger than max_intensity is allowed, e.g. a strong feature outside the
    window. Scaled, it gets a box larger than the unscaled one.

    This rule leaves out more than amplitude_half_widths: each box ends by the feature's own
    intensity, but on the grid a left-out amplitude (a dropped feature, or with
    cut_tails=True a cut tail) also changes the intensity of its neighbours, because
    amplitudes add before squaring (see amplitude_half_widths). To see the error for a case,
    compare with evaluate.all_features_on_grid.
    Every feature needs rows and a propagator_param > 0. Scaled, without max_intensity, at
    least one feature must lie inside `window`.
    A flat feature (is_flat) gets 0, as in amplitude_half_widths. Scaled, it counts as
    inside the window: its intensity is on every grid point.
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
        _check_has_rows_and_width(f)
    i_max = 0.
    if scale_wrt_max_intensity:
        if max_intensity is not None:
            i_max = max_intensity
        else:
            inside = [f.get_intensity() for f in features if f.is_flat or f.is_inside(window)]
            if not inside:
                raise ValueError('scaled boxes need the strongest peak inside the window, and no feature is inside: '
                                 'pass max_intensity')
            i_max = max(inside)

    half_widths = []
    for f in features:
        if f.is_flat:
            half_widths.append(0.)
            continue
        gamma = max(f.peak_widths())
        if scale_wrt_max_intensity:
            # the feature's peak over the floor, i / (i_max / dynrange);
            # 0 if every peak inside is 0
            k = dynrange * f.get_intensity() / i_max if i_max > 0. else 0.
        else:
            k = dynrange
        # where the intensity is down to 1/k of its peak, the amplitude to 1/sqrt(k); 0 for k <= 1
        reach = lineshape.distance(1. / np.sqrt(k), gamma) if k > 1. else 0.
        box_extent = reach * (1. + box_range_safety_margin)
        half_widths.append(max(box_extent, minimum_box_padding))
    return half_widths


@dataclass
class SpectralWindow:
    """
    The full spectrum range (a Box) with the features evaluated in it
    (made by from_features).
        full_features     the feature's location is inside the box (flat features too)
        contrib_features  the feature's location is outside the box, but its own box
                          overlaps the box, so its tail still adds to the spectrum
    """
    box: 'Box'
    full_features: list['SpectralFeature'] = field(default_factory=list)
    contrib_features: list['SpectralFeature'] = field(default_factory=list)

    @classmethod
    def from_features(cls, box: Box, features: Sequence['SpectralFeature'],
                      half_widths: Sequence[float]) -> 'SpectralWindow':
        """
        The window over `box` with the features that add amplitude inside it, each with its
        own box. In this order:
          1. a box around every feature: its location +- its half-width, the same on every
             axis. half_widths: cm-1, one per feature, in the order of `features`, e.g. from
             a box rule (BoxRule). The window does not depend on how they were chosen.
          2. keep the features whose box overlaps `box` (touching edges included, so a
             feature with half-width 0 on the edge of `box` stays):
             location inside `box` -> full_features, outside -> contrib_features
          No feature is dropped for its amplitude:
            - a weak feature stays, however weak
            - a feature whose rows cancel (amplitude 0) stays, as a record of the
              cancellation; it adds nothing to the grid
            - a term with coefficient 0 never becomes a row (build_contributions puts it
              in `zero`), so it never reaches the window

        e.g. half-width 55 cm-1, `box` A (0, 10): a feature at A = 60 is kept (its box
        starts at 5), a feature at A = 70 is not (its box starts at 15).
        A flat feature (is_flat) has no location to put a box around: its box is `box`
        itself, and it goes into full_features. Its half-width is not used.

        Returns copies with boxes; the input features stay as they are. Every feature needs
        rows and a propagator_param: the evaluation on the grid needs both.
        """
        if len(half_widths) != len(features):
            raise ValueError(f'one half-width per feature: {len(half_widths)} half-widths, {len(features)} features')

        dressed = []
        for f, half_width in zip(features, half_widths):
            _check_has_rows_and_width(f)
            if not half_width >= 0.:
                raise ValueError(f'half-width must be >= 0, got {half_width} for {f}')
            if f.is_flat:
                dressed.append(replace(f, feat_box=box))
                continue
            feat_box = Box({k: (v - half_width, v + half_width) for k, v in f.location.as_dict().items()})
            dressed.append(replace(f, feat_box=feat_box))

        kept = [f for f in dressed if f.is_flat or f.feat_box_overlaps(box)]

        return cls(box,
                   full_features=[f for f in kept if f.is_flat or f.is_inside(box)],
                   contrib_features=[f for f in kept if not (f.is_flat or f.is_inside(box))])
