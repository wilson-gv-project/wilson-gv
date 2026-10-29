"""
Entry point: terms + molecular data -> spectral features -> the spectrum on a grid.

Two calls, from terms already translated to axes and the data:
    result = evaluate_response_terms(terms, molsys_data=molsys, states_choice=..., laser_pol=...,
                                     propagator_param=..., exclude_modes=...)
                                                                      rows and features: FeatureResult
    spec   = spectrum_on_grid(result, window_box=..., n_points_per_axis=..., box_rule=...)
                                                                      the spectrum on a grid: EvalRspResult
A caller needs only these two calls. The inputs are plain values, the terms and a few small types (below).
evaluate_response_terms: compile_terms -> make_polarization_linear_comb -> compute_features.
The evaluator neither derives nor translates terms: no file of it imports wilson_experiment, derive.derive or
term_var_translate.
One call is for one set of term axes, e.g. ('A', 'B') (check_same_axis_set). Term axes: the axes in the pf of a
term's resonance conditions, without sign. Terms with more than one set, e.g. the 3 IR setup (('A', 'B') and
('A', 'B', 'C')): terms_by_axis_set splits them, and each set gets its own call.

The data is the caller's job: the evaluator reads no file and takes no DataOriginInfo. The caller makes one
MolSystemData (wilson_system) with the properties the terms use, prop_names(terms), e.g.
    molsys = MolSystemData.load(origin, prop_names(terms))     origin: DataOriginInfo; once, for any number of runs
    molsys = MolSystemData.from_datadict(...)                  from a data dict the caller already read
evaluate_response_terms checks that molsys has every property the terms use (MolSystemData.check_props).

Inputs
    terms             VibPerturbedTerm list, translated to axes: each condition's pf in axis labels, e.g. ['A', '-B']
                      (derive writes pulse IDs, e.g. [-1, 2]: plan.ResCondKey raises on them). row.term_id is the
                      index in the list.
    molsys_data       MolSystemData: the properties, frequencies and state sets of one molecule (see above)
    states_choice    'harmonic' or 'anharmonic'. Must follow the vib analysis regime: GVPT2, VPT2 -> 'anharmonic';
                      harmonic -> 'harmonic'. No default: the data cannot know the regime.
    laser_pol         the experiment's polarization_avg_vector, e.g. (1, 1, 1): pulses and detector along x.
                      The averaging rank fixes its length (rank 4: 3 numbers).
    propagator_param  the parameter of each condition's line shape (lineshape.py), cm-1, e.g. the
                      Lorentzian's half width gamma: a number, or a function (vd, freqs) -> cm-1
                      (features.PropagatorParam). It sizes both the resonance on the grid and its box.
    exclude_modes     mode numbers, from 0 as in the data, e.g. (5,): left out of the evaluation (the data keeps
                      them). Default: none.
    window_box        grid.Box, the range of each axis, cm-1, e.g. Box({'A': (600., 2800.), 'B': (5., 3100.)}).
                      Its axes are the term axes.
    n_points_per_axis grid points on each axis of window_box, both edges included, e.g. {'A': 501, 'B': 201}
    box_rule          features.BoxRule: (window_box, features) -> the half-width of each feat_box, cm-1. Knobs by
                      functools.partial, e.g. partial(amplitude_half_widths, dynrange=500.). dynrange: the plot's
                      dynamic range (max / floor), required. intensity_half_widths: by intensity, leaves out more.
    cut_tails         False (default): the feat_boxes choose the features, each chosen one on the whole grid.
                      True: each feature on its own feat_box only, faster, its tail beyond cut off.

Outputs
    FeatureResult     the evaluation, before any grid:
                        table           one ContributionRow per term and index set: term_id, motif, params,
                                        location (cm-1), coeff (au)
                        zero            the (term_id, params) pairs with coeff 0: no row
                        features        one SpectralFeature per (motif, location), made from the table
                        states_choice, exclude_modes, propagator_param: as passed
                        factors_by_index_set(row)   the summed index sets behind one row, each with its
                                        factors, computed again on demand
                        cancelling(rel_tol)   the features whose rows (nearly) cancel
    EvalRspResult     the spectrum on the grid:
                        axis_points     the points along each axis, cm-1: the grid is every combination of them
                        amplitude       complex amplitude in au, one per grid point; intensity = abs(amplitude)**2
                        mesh            every grid point, one full array per axis: what the renderer reads
                        spectral_window the features as evaluated on the grid, each with its box
                        feature_result  the FeatureResult it was made from

Names in the grid part, one per kind of data (the types are in grid.py), e.g. A from 2600 to 2800 with 5 points:
    window_box        a Box: the range of each axis        {'A': (2600., 2800.)}
    feat_box          a Box: the range of one feature on each axis
    n_points_per_axis the number of points on each axis    {'A': 5}
    axis_points       the points along each axis           {'A': [2600., 2650., 2700., 2750., 2800.]}
                      (grid.AxisPoints, made by Box.grid_points); the grid is every combination of them
    mesh              every grid point, one full array per axis (grid.Mesh, made by grid.make_mesh):
                      mesh['A'][i, j] = axis_points['A'][i]. Built where it is used, one cut at a time
    cut               where a feature's box sits in the grid: one slice per axis (grid.box_slices);
                      cut_points: the axis_points inside the cut
    amplitude         the complex amplitude, au, one value per grid point; intensity = abs(amplitude)**2
    _cm, _au          the unit, where both occur: evaluate.py converts cm-1 to Eh

CompiledTerm list
  │ 1. index sets: fix the motif labels (make_idx_sets over term.idx_nonsumm)
  │ 2. coeff: evaluate_term_coeff_sumover, sums the other labels     ← MolSystemData, polarization
  │    zero coeff -> `zero`, no row
  │ 3. location: solve_LSE_motif, once per (motif, params), cm-1     ← molsys_data.states[states_choice]
  │    always one point: compile_terms checked the motif. A term without conditions: the point without
  │    axes, a flat background on every grid point
  v
ContributionRow(term_id, motif, params, location, coeff)   "why" layer    evaluate.build_contributions
  │ 4. group by (motif, location), add the coeffs
  v
SpectralFeature(location, rows)                            "what" layer   features.features_from_rows
  │    evaluate_response_terms ends here (FeatureResult.features); spectrum_on_grid goes on:
  │ 5. checks (check_grid_axes): term axes = window_box.axes; n_points_per_axis has one whole number >= 1
  │    per axis of window_box
  │ 6. half-widths: box_rule(window_box, features), one per feature, cm-1   ← window_box, box_rule
  │    e.g. amplitude_half_widths: from each feature's amplitude; strong features get big feat_boxes
  │ 7. feat_boxes: SpectralWindow.from_features, a feat_box of location ± half-width per feature.
  │    Keeps a feature if its feat_box overlaps window_box (touching counts): full_features
  │    (location inside), contrib_features (outside). A flat feature's feat_box is window_box
  │ 8. grid: axis_points = window_box.grid_points(n_points_per_axis), both edges included   ← n_points_per_axis
  │ 9. values: the features of spectral_window added into one complex array, amplitudes before squaring:
  │    each on the whole grid (all_features_on_grid; default), or with cut_tails=True each on
  │    the grid points inside its feat_box (boxed_features_on_grid, box_slices)
  v
EvalRspResult(axis_points, amplitude, spectral_window)    the spectrum on the grid
  │ 10. intensity = abs(amplitude)**2, e.g. plot_rsp_func (wilson_analysis/render)
  v
the plot
"""
import numbers
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import TYPE_CHECKING

import numpy as np

from wilson_suite.wilson_intensities.amplitudes.averaging import (
    get_iso_f,
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    all_features_on_grid,
    boxed_features_on_grid,
    build_contributions,
    evaluate_term_coeff_sumover,
    included_modes,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    BoxRule,
    ContributionRow,
    ContributionTable,
    PropagatorParam,
    SpectralFeature,
    SpectralWindow,
    features_from_rows,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    AxisPoints,
    Box,
    Mesh,
    make_mesh,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResonanceMotif,
    compile_terms,
    split_props,
)
from wilson_suite.wilson_system.system_data import MolSystemData

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import CompiledTerm
    from wilson_suite.wilson_system.system_data import VibStatesData


PolarizationAveragingRecipe = dict[tuple[int, ...], float]  # (ax0, ax1, ...) -> coefficient


@dataclass(frozen=True)
class FeatureResult:
    """
    The evaluation of a term list before any grid: its rows and features (made by compute_features).
        table             one ContributionRow per (term, index set) with coeff != 0
        zero              the (term_id, params) pairs with coeff 0: no row
        propagator_param  cm-1, a number or a function (features.PropagatorParam)
    What the rows were computed from, kept by reference, not copied:
        compiled                  the compiled terms; row.term_id is the index in them
        molsys                    the data
        polarization_linear_comb  the orientation-average recipe
        states_choice             the state set, e.g. 'anharmonic'
        exclude_modes             sorted, e.g. (5,): the sums skip them (evaluate.included_modes)
    Their field options:
        default=None (exclude_modes: ())  a result made by hand may leave them out
        compare=False                     == ignores them: results with the same rows are equal
        repr=False                        print does not show compiled, molsys and the recipe: large
    factors_by_index_set needs all five. spectrum_on_grid needs molsys and states_choice (states).
    """
    table: ContributionTable
    zero: list[tuple[int, ParameterSet]]
    propagator_param: PropagatorParam | None = None
    compiled: tuple['CompiledTerm', ...] | None = field(default=None, compare=False, repr=False)
    molsys: MolSystemData | None = field(default=None, compare=False, repr=False)
    polarization_linear_comb: dict | None = field(default=None, compare=False, repr=False)
    states_choice: str | None = field(default=None, compare=False)
    exclude_modes: tuple[int, ...] = field(default=(), compare=False)

    @cached_property
    def features(self) -> list[SpectralFeature]:
        """
        One feature per (motif, location), built from the table on first use.
        A propagator_param function also needs the states (compute_features keeps them): each feature gets
        its conditions from them. A result made by hand has no states, so its features get no conditions.
        """
        states = None if self.molsys is None or self.states_choice is None else self.states
        return features_from_rows(self.table, self.propagator_param, states)

    @property
    def states(self) -> 'VibStatesData':
        """The state set the rows were computed with, molsys.states[states_choice]: boxed_features_on_grid needs it."""
        if self.molsys is None or self.states_choice is None:
            raise ValueError('this FeatureResult has no data or states choice (made by hand?): compute_features '
                             'keeps them')
        return self.molsys.states[self.states_choice]

    def factors_by_index_set(self, row: ContributionRow) -> dict:
        """
        The summed index sets behind one row, each with its factors.
            e.g. a row of a term with resonance labels a, b and summation label c, at a = 0, b = 1, 42 modes:
                 42 index sets, {ParameterSet({'a': 0, 'b': 1, 'c': 0}): {'NON_AVRG': ..., 'AVRG': ...,
                 'VIBDIFF_TERMS': ..., 'VIBENE_DENOM': ...}, ...,  ParameterSet({'a': 0, 'b': 1, 'c': 41}): {...}}
            An index set whose average or non-averaged part is 0 holds only the factors made before the stop,
            e.g. {'AVRG': 0.}.
        Computed again from row.term_id and row.params, with the terms, data, polarization and excluded modes of
        this result, so the index sets add up to row.coeff. build_contributions does not keep them: each row would
        keep one dict per summed index set.
        """
        if (self.compiled is None or self.molsys is None or self.polarization_linear_comb is None
                or self.states_choice is None):
            raise ValueError('this FeatureResult has no terms or data (made by hand?): compute_features keeps them, '
                             'and factors_by_index_set needs them')
        _, factors = evaluate_term_coeff_sumover(self.compiled[row.term_id], row.params.to_dict(), self.molsys,
                                                 self.polarization_linear_comb, states_choice=self.states_choice,
                                                 modes=included_modes(self.molsys, self.exclude_modes))
        return factors # type: ignore

    def cancelling(self, rel_tol: float) -> list[SpectralFeature]:
        """
        The features whose rows (nearly) cancel: net_fraction <= rel_tol.
            e.g. rel_tol 0.01: the sum keeps at most 1 % of the summed |coeffs|;  rel_tol 0: exact cancellation only
        Only rows count. A term with coefficient 0 is in `zero` and never a row, so it is no cancellation.
        """
        if not 0. <= rel_tol < 1.:
            raise ValueError(f'rel_tol must be at least 0 and below 1, got {rel_tol}')
        return [f for f in self.features if f.net_fraction is not None and f.net_fraction <= rel_tol]


def compute_features(compiled: Sequence['CompiledTerm'], molsys_data: MolSystemData,
                     polarization_linear_comb: dict,
                     propagator_param: PropagatorParam | None = None,
                     *,
                     states_choice: str,
                     exclude_modes: Collection[int] = ()) -> FeatureResult:
    """
    One SpectralFeature per (motif, location).
    polarization_linear_comb: the orientation-average recipe (make_polarization_linear_comb).
    propagator_param: cm-1, a number for every condition, or a function (vd, freqs) -> cm-1 for a width
        per condition and/or per frequency (features.PropagatorParam).
    states_choice: the state set of molsys_data to use, e.g. 'anharmonic'. It must follow the vib analysis regime:
        GVPT2, VPT2 -> 'anharmonic'; harmonic -> 'harmonic'. No default.
    exclude_modes: modes left out of the evaluation, e.g. (5,): no peak and no summed value uses them
    (evaluate.included_modes).
    row.term_id is the index in `compiled`, and so in the term list it was compiled from.
    The result keeps the inputs, for FeatureResult.factors_by_index_set and for the states (FeatureResult.states) on the grid.
    raises ValueError if the terms have more than one set of term axes (check_same_axis_set).
    """
    check_same_axis_set(compiled)
    table, zero = build_contributions(compiled, molsys_data, polarization_linear_comb, states_choice=states_choice,
                                      exclude_modes=exclude_modes)
    return FeatureResult(table, zero, propagator_param,
                         compiled=tuple(compiled), molsys=molsys_data,
                         polarization_linear_comb=polarization_linear_comb, states_choice=states_choice,
                         exclude_modes=tuple(sorted(set(exclude_modes))))


def _term_ids_by_axis_set(motifs: Iterable[ResonanceMotif]) -> dict[tuple[str, ...], list[int]]:
    """term ids grouped by their term axes, sets in first-seen order, e.g. {('A', 'B'): [0, 2], ('A',): [1]}"""
    term_ids = defaultdict(list)
    for term_id, motif in enumerate(motifs):
        term_ids[tuple(sorted(motif.get_max_different_freq_axes()))].append(term_id)
    return dict(term_ids)


def check_same_axis_set(compiled: Sequence['CompiledTerm']) -> None:
    """
    Checks that all terms have the same term axes: the axes in the pf of a term's resonance
    conditions, without sign, e.g. ('A', 'B') for the conditions pf ('-A',) and pf ('B',).
    One evaluation and its grid take one set of term axes.
    Raises ValueError otherwise; the message lists the term ids per set of term axes, e.g. the
    3 IR setup: {('A', 'B'): [0, ..., 13], ('A', 'B', 'C'): [14, 15]}.
    terms_by_axis_set splits such terms into one group per set.
    Two sets differ whenever their axes differ, also when one holds the other: ('A',) and ('A', 'B').
    A term without conditions (a flat term) has the term axes (): it fits any set and does not
    count here.
    compile_terms checks each term on its own, so this check is what catches a mix of sets.
    """
    term_ids = {axis_set: ids for axis_set, ids in _term_ids_by_axis_set(t.cmp_resmotf for t in compiled).items()
                if axis_set}
    if len(term_ids) > 1:
        raise ValueError(f'{len(term_ids)} different sets of term axes, and one evaluation takes one set; '
                         f'term ids by set: {term_ids}. Split them with terms_by_axis_set: '
                         f'one evaluate_response_terms call per set')


def check_grid_axes(features: Sequence[SpectralFeature], window_box: Box,
                    n_points_per_axis: Mapping[str, int]) -> None:
    """
    Checks that the grid fits the features. Raises ValueError unless:
        1. the term axes of the features (their location axes) are window_box.axes, e.g.
           ('A', 'B') on Box({'A': ..., 'B': ...}). A flat feature (from terms without
           conditions) has the term axes (): it fits any window_box.
        2. n_points_per_axis has one entry per axis of window_box
        3. each entry is a whole number, at least 1 (Box.grid_points checks the rest: 2 or more
           points, or 1 with min = max)
    """
    term_axes = {f.location.axes for f in features if not f.is_flat}
    if len(term_axes) > 1:
        raise ValueError(f'{len(term_axes)} different sets of term axes, and one grid takes one set: '
                         f'{sorted(term_axes)}')
    if term_axes and term_axes != {window_box.axes}:
        (axes,) = term_axes
        raise ValueError(f'term axes {axes} != window_box.axes {window_box.axes}')
    if set(n_points_per_axis) != set(window_box.axes):
        raise ValueError(f'n_points_per_axis axes {tuple(sorted(n_points_per_axis))} != '
                         f'window_box.axes {window_box.axes}')
    not_counts = {ax: n for ax, n in n_points_per_axis.items()
                  if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1}
    if not_counts:
        raise ValueError(f'n_points_per_axis needs a whole number >= 1 per axis, got {not_counts}')


def terms_by_axis_set(terms: Sequence['VibPerturbedTerm']) -> dict[tuple[str, ...], list[int]]:
    """
    The term ids (index in `terms`) grouped by their term axes, e.g. the 3 IR setup:
    {('A', 'B'): [0, ..., 13], ('A', 'B', 'C'): [14, 15]}. Terms without conditions: term axes ().
    They are a flat background, the same on any grid, and check_same_axis_set does not count them, so they
    can join the call of any set, e.g. IR absorption: {(): [0, 1, 2], ('A',): [3, 4]}, all 5 terms in one call.
    terms: translated to axes, as for evaluate_response_terms.
    For the caller: one evaluate_response_terms call per set, e.g.
        for axis_set, ids in terms_by_axis_set(terms).items():
            result = evaluate_response_terms([terms[i] for i in ids], ...)
    There row.term_id is the index in that call's list: ids[row.term_id] is the index in `terms`.
    """
    return _term_ids_by_axis_set(ResonanceMotif.from_conditions(t.res) for t in terms)


def prop_names(terms: Sequence['VibPerturbedTerm']) -> list[str]:
    """
    Every property the terms use, each once, e.g. ['polgrad', 'dipgrad', 'cff', ...]: what the caller loads,
    e.g. MolSystemData.load(origin, prop_names(terms)).
    Only the props are read (plan.split_props).
    Per term the averaged props first, as in CompiledTerm.prop_names.
    """
    return list(dict.fromkeys(name for term in terms for part in split_props(term) for name in part.names))


def averaging_rank(terms: Sequence['VibPerturbedTerm']) -> int:
    """
    Rank of the orientation average: the number of Cartesian slots (op.o) in a term's averaged part.
        e.g. polhess[0,3] * dipgrad[1] * dipgrad[2] -> slots 0, 1, 2, 3 -> rank 4

    compute_features uses one recipe for every term, so all terms must have the same slots.
    The recipe puts slot i at position i, so the slots must be 0 .. rank-1, each once.
    """
    if not terms:
        raise ValueError('no terms: no averaging rank')

    term_ids_by_slots = defaultdict(list)
    for term_id, term in enumerate(terms):
        avrg_props, _ = split_props(term)
        term_ids_by_slots[tuple(sorted(avrg_props.get_cart_axes()))].append(term_id)

    if len(term_ids_by_slots) > 1:
        raise ValueError(f'terms differ in their averaged slots, one recipe cannot serve them all; '
                         f'slots: term ids {dict(term_ids_by_slots)}')
    (slots,) = term_ids_by_slots
    if slots != tuple(range(len(slots))):
        raise ValueError(f'averaged slots must be 0 .. {len(slots) - 1}, each once; got {slots}')
    return len(slots)


def make_polarization_linear_comb(terms: Sequence['VibPerturbedTerm'],
                                  laser_pol: Sequence[float]) -> PolarizationAveragingRecipe:
    """
    The orientation-average recipe {(x/y/z per slot): coefficient} that compute_features takes,
    from getGeneralPolarizationAveragingExpression with the rank of the terms (averaging_rank).
    Only the props of the terms are read (plan.split_props).

    laser_pol: the experiment's polarization_avg_vector. Its length is fixed by the rank
    (rank 4 -> 3 numbers, e.g. (1, 1, 1) when all pulses and the detector are along x).
    getGeneralPolarizationAveragingExpression would ignore extra numbers without an error.

    The recipe depends only on the rank and laser_pol: it is made once per (rank, laser_pol) and kept
    (_averaging_recipe). Each call returns its own copy, so changing one copy changes no other result.
    """
    rank = averaging_rank(terms)
    if not 2 <= rank <= 6:
        raise ValueError(f'orientation averaging supports rank 2 to 6, the terms have rank {rank}')

    n_expected = len(get_iso_f(rank))
    if len(laser_pol) != n_expected:
        raise ValueError(f'rank {rank} needs laser_pol with {n_expected} numbers, got {len(laser_pol)}: {laser_pol}')
    
    # copies only the outer dict
    return dict(_averaging_recipe(rank, tuple(laser_pol)))


@cache
def _averaging_recipe(rank: int, laser_pol: tuple[float, ...]) -> PolarizationAveragingRecipe:
    """getGeneralPolarizationAveragingExpression, made once per (rank, laser_pol); make_polarization_linear_comb copies it"""
    return getGeneralPolarizationAveragingExpression(rank=rank, laser_pol=laser_pol)


def evaluate_response_terms(terms: Sequence['VibPerturbedTerm'], *,
                            molsys_data: MolSystemData,
                            states_choice: str,
                            laser_pol: Sequence[float],
                            propagator_param: PropagatorParam,
                            exclude_modes: Collection[int] = ()) -> FeatureResult:
    """
    Terms translated to axes + data loaded before + experiment settings -> rows and features.
        1. compile                                          compile_terms (raises on a condition in pulse IDs:
                                                            plan.ResCondKey)
        2. orientation-average recipe, rank from the terms  make_polarization_linear_comb (made once per
                                                            rank and laser_pol)
        3. rows, zero pairs, features                       compute_features (raises first if the terms have
                                                            more than one set of term axes: check_same_axis_set;
                                                            terms_by_axis_set splits them)
    Next step: spectrum_on_grid puts the features of the result on a grid. One result serves any number
    of grids, e.g. other window_box values: the rows are computed here once, not again for each grid.

    terms             translated to axes: each condition's pf in axis labels, e.g. ['A', '-B']
    molsys_data      MolSystemData, made by the caller, e.g. MolSystemData.load(origin, prop_names(terms)):
                      loaded once, for any number of runs
    states_choice     must follow the vib analysis regime: GVPT2, VPT2 -> 'anharmonic'; harmonic -> 'harmonic'
    laser_pol         the experiment's polarization_avg_vector, e.g. (1, 1, 1): pulses and detector along x
    propagator_param  cm-1: a number, or a function (vd, freqs) -> cm-1 (features.PropagatorParam), e.g.
                      lambda vd, freqs: 0.01 * abs(vd.energy_difference())
    exclude_modes     modes left out of the evaluation, counted from 0 as in the data, e.g. (5,)
                      (VibAnaSetup.exclude_modes): no peak and no summed value uses them; the data keeps them. The
                      other peaks change too: their sums lose the excluded modes (evaluate.included_modes).
                      Default: none.
    Every argument after the terms is keyword-only: two swapped arguments raise a TypeError.
    row.term_id is the index in `terms`.
    """
    compiled = compile_terms(terms)
    recipe = make_polarization_linear_comb(terms, laser_pol)
    return compute_features(compiled, molsys_data, recipe, propagator_param=propagator_param,
                            states_choice=states_choice, exclude_modes=exclude_modes)


@dataclass(frozen=True)
class EvalRspResult:
    """
    The spectrum on a grid, made by spectrum_on_grid. Every field is set.
    feature_result   rows, features, and what they came from (molsys, polarization recipe, states_choice)
    axis_points      the points along each axis, cm-1, both window edges included (grid.AxisPoints), e.g. A and B
                     with 201 points each: the grid is every combination of them
    amplitude        complex amplitude in au, one value per grid point; array dimension k is the k-th axis of
                     axis_points, e.g. shape (201, 201). intensity = abs(amplitude)**2
    spectral_window  the window as used: its features, each with its box (evaluated on the whole grid, or on
                     its own box only with cut_tails=True in spectrum_on_grid).
                     full_features: inside the window; contrib_features: outside, with a box that reaches in.
    mesh             (property) every grid point, one full array per axis (grid.Mesh, grid.make_mesh),
                     mesh['A'][i, j] = axis_points['A'][i]: what the renderer reads
    """
    feature_result: FeatureResult
    axis_points: AxisPoints
    amplitude: np.ndarray
    spectral_window: SpectralWindow

    @cached_property
    def mesh(self) -> Mesh:
        return make_mesh(self.axis_points)


def spectrum_on_grid(feature_result: FeatureResult, *,
                     window_box: Box,
                     n_points_per_axis: dict[str, int],
                     box_rule: BoxRule,
                     cut_tails: bool = False) -> EvalRspResult:
    """
    The features of feature_result on a grid over window_box, in five steps:
        1. check_grid_axes: term axes = window_box.axes, and n_points_per_axis has one whole number
           >= 1 per axis of window_box
        2. box_rule(window_box, features): the half-width of each feature's feat_box, cm-1
        3. SpectralWindow.from_features: each feature gets its feat_box, location +- half-width;
           keeps the features whose feat_box overlaps window_box (touching counts)
        4. window_box.grid_points: the grid, n_points_per_axis[ax] points on axis ax, both edges included
        5. the features of spectral_window (spectral_window.features) on the grid, all added into one array:
               cut_tails=False (default)  all_features_on_grid: each feature on the whole grid.
                                          The feat_boxes only choose the features.
               cut_tails=True             boxed_features_on_grid: each feature only on the grid
                                          points inside its feat_box (the steps and an example are there)
    The rows are not computed again (the slow part: build_contributions), so a new window_box,
    grid, propagator_param or box rule runs only the grid part:
        rule = partial(amplitude_half_widths, dynrange=500.)
        again = spectrum_on_grid(result, window_box=other_window_box, n_points_per_axis={'A': 401, 'B': 401},
                                 box_rule=rule)
        wider = spectrum_on_grid(replace(result, propagator_param=8.), window_box=window_box,
                                 n_points_per_axis=n_points_per_axis, box_rule=rule)
    replace (dataclasses): a new FeatureResult with the same table; its features are made again, with the
    new propagator_param, on first use (features_from_rows).

    window_box         the range of each axis, cm-1, a Box, e.g. Box({'A': (2600., 2800.), 'B': (-100., 100.)}).
                       Its axes are the term axes. If no term has conditions: any axes.
    n_points_per_axis  grid points on each axis of window_box, both edges included, e.g. {'A': 201, 'B': 201}
    box_rule           the half-width of each feat_box: a function (window_box, features) -> cm-1, one per
                       feature (features.BoxRule). Two exist in features.py; functools.partial gives them
                       their knobs:
                           partial(amplitude_half_widths, dynrange=500.)   by amplitude: what the feat_boxes
                                                                           leave out changes the intensity by
                                                                           less than eps * the plot's floor
                           partial(intensity_half_widths, dynrange=500.)   by intensity: leaves out more
                       dynrange: the plot's dynamic range (max / floor). Required by both: the plot decides it.
                       Any other function works too, e.g. lambda window_box, features: [20.] * len(features).
    cut_tails          False (default): each kept feature on the whole grid, so only the dropped features are
                       missing; the run time grows with (kept features) x (grid points). True: each feature
                       only on its feat_box, its tail beyond cut off; the run time grows with the number of
                       grid points in all feat_boxes.
    raises ValueError if the grid does not fit the terms (check_grid_axes).
    feature_result needs its states (FeatureResult.states), so it must come from compute_features.
    """
    features = feature_result.features
    check_grid_axes(features, window_box, n_points_per_axis)

    # the half-width of each feature's feat_box; from_features makes the feat_boxes from them
    half_widths = box_rule(window_box, features)
    spectral_window = SpectralWindow.from_features(window_box, features, half_widths)

    axis_points = window_box.grid_points(n_points_per_axis)

    # the features of spectral_window on the grid, amplitudes added
    if cut_tails:
        amplitude = boxed_features_on_grid(spectral_window, feature_result.states, axis_points)
    else:
        amplitude = all_features_on_grid(spectral_window.features, feature_result.states, axis_points)

    return EvalRspResult(feature_result, axis_points, amplitude, spectral_window)
