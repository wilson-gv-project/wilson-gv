"""
Entry point: terms + molecular data -> spectral features -> the spectrum on a grid.

Three calls, from the terms derive writes (in pulse IDs):
    molsys = load_molsys_data(terms, data_origin)                     once, for any number of runs
    result = evaluate_response_terms(terms, axes=..., molsys_data=molsys, states_choice=..., laser_pol=...,
                                     propagator_param=...)            rows and features: FeatureResult
    spec   = draw_spectrum(result, window, dim_sizes, dynrange=...)   the spectrum on a grid: EvalRspResult
evaluate_response_terms: translate to axes -> compile_terms -> make_polarization_linear_comb -> compute_features.
load_molsys_data and make_polarization_linear_comb read only the props, so they take the terms in pulse IDs.

CompiledTerm list
  │ 1. index sets: fix the motif labels (make_idx_sets over term.idx_nonsumm)
  │ 2. coeff: evaluate_term_coeff_sumover, sums the other labels     ← MolSystemData, polarization
  │    zero coeff -> `zero`, no row
  │ 3. location: solve_LSE_motif, once per (motif, params), cm-1     ← molsys_data.states[states_choice]
  │    always one point: compile_terms checked the motif (B7)
  v
ContributionRow(term_id, motif, params, location, coeff)   "why" layer    evaluate.build_contributions
  │ 4. group by (motif, location), add the coeffs
  v
SpectralFeature(location, rows)                            "what" layer   features.features_from_rows
  │ 5. peak width, boxes by amplitude (features.amplitude_half_widths), spectral window
  v
spectrum array on a grid                                   draw_spectrum (evaluate.draw_window)
"""
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import TYPE_CHECKING

import numpy as np

from wilson_suite.wilson_derive import term_var_translate
from wilson_suite.wilson_intensities.amplitudes.averaging import (
    get_iso_f,
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    build_contributions,
    draw_window,
    evaluate_term_coeff_sumover,
    included_modes,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionRow,
    ContributionTable,
    PropagatorParam,
    SpectralFeature,
    SpectralWindow,
    amplitude_half_widths,
    features_from_rows,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    compile_terms,
    split_props,
)
from wilson_suite.wilson_system.system_data import (
    DataOriginInfo,
    MolSystemData,
)
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    from wilson_suite.wilson_experiment.indep_vars_and_axes import SpectralAxisSet
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import CompiledTerm
    from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
    from wilson_suite.wilson_system.system_data import VibStatesData


PolarizationAveragingRecipe = dict[tuple[int, ...], float]  # (ax0, ax1, ...) -> coefficient


@dataclass(frozen=True)
class FeatureResult:
    table: ContributionTable                          # all rows: the source
    zero: list[tuple[int, ParameterSet]]              # pairs with coeff == 0
    propagator_param: PropagatorParam | None = None   # cm-1: a number or a function (features.PropagatorParam)
    # What the rows were computed from (compute_features fills them; None in a result made by hand).
    # leaves() needs the first four, and sums over the modes left after exclude_modes; drawing needs the states
    # (FeatureResult.states). References only: nothing is copied. compare=False: == leaves them out.
    compiled: tuple['CompiledTerm', ...] | None = field(default=None, compare=False, repr=False)
    molsys: MolSystemData | None = field(default=None, compare=False, repr=False)
    polarization_linear_comb: dict | None = field(default=None, compare=False, repr=False)
    states_choice: str | None = field(default=None, compare=False)
    exclude_modes: tuple[int, ...] = field(default=(), compare=False)   # sorted, e.g. (5,): evaluate.included_modes

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
        """The state set the rows were computed with, molsys.states[states_choice]: what draw_window needs."""
        if self.molsys is None or self.states_choice is None:
            raise ValueError('this FeatureResult has no data or states choice (made by hand?): compute_features '
                             'keeps them')
        return self.molsys.states[self.states_choice]

    def leaves(self, row: ContributionRow) -> dict:
        """
        The summed index sets behind one row, each with its factors: feature -> rows -> leaves.
            e.g. a row of a term with resonance labels a, b and summation label c, at a = 0, b = 1, 42 modes:
                 42 leaves, {ParameterSet({'a': 0, 'b': 1, 'c': 0}): {'NON_AVRG': ..., 'AVRG': ...,
                 'VIBDIFF_TERMS': ..., 'VIBENE_DENOM': ...}, ...,  ParameterSet({'a': 0, 'b': 1, 'c': 41}): {...}}
            An index set whose average or non-averaged part is 0 holds only the factors made before the stop,
            e.g. {'AVRG': 0.}.
        Computed again from row.term_id and row.params, with the terms, data, polarization and excluded modes of
        this result, so the leaves add up to row.coeff. About 1 ms per row on ACDM (42 modes); build_contributions
        does not keep them, since every row's leaves would take 357 MB there.
        """
        if (self.compiled is None or self.molsys is None or self.polarization_linear_comb is None
                or self.states_choice is None):
            raise ValueError('this FeatureResult has no terms or data (made by hand?): compute_features keeps them, '
                             'and leaves need them')
        _, leaves = evaluate_term_coeff_sumover(self.compiled[row.term_id], row.params.to_dict(), self.molsys,
                                                self.polarization_linear_comb, states_choice=self.states_choice,
                                                modes=included_modes(self.molsys, self.exclude_modes))
        return leaves # type: ignore

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
    exclude_modes: modes removed everywhere, e.g. (5,): no peak and no summed value uses them (evaluate.included_modes).
    row.term_id is the index in `compiled`, and so in the term list it was compiled from.
    The result keeps the inputs, for FeatureResult.leaves and for the states (FeatureResult.states) in drawing.
    """
    table, zero = build_contributions(compiled, molsys_data, polarization_linear_comb, states_choice=states_choice,
                                      exclude_modes=exclude_modes)
    return FeatureResult(table, zero, propagator_param,
                         compiled=tuple(compiled), molsys=molsys_data,
                         polarization_linear_comb=polarization_linear_comb, states_choice=states_choice,
                         exclude_modes=tuple(sorted(set(exclude_modes))))


def load_molsys_data(terms: Sequence['VibPerturbedTerm'], data_origin: DataOriginInfo) -> MolSystemData:
    """
    One MolSystemData for all terms, each property once: MolSystemData.load with the terms' prop_names.
    terms: as derive writes them (pulse IDs) or translated to axes: only the props are read (plan.split_props).
    It holds every state set the source has; the evaluation chooses one (compute_features, states_choice).
    The obtainer is this module's wilson_data_obtainer, so a test can replace it here (c3_spectrum.py does).
    """
    return MolSystemData.load(data_origin, prop_names(terms), obtainer=wilson_data_obtainer)


def prop_names(terms: Sequence['VibPerturbedTerm']) -> list[str]:
    """
    Every property the terms use, each once, e.g. ['polgrad', 'dipgrad', 'cff', ...]: input of MolSystemData.load.
    Per term the averaged props first, as in CompiledTerm.prop_names.
    """
    return list(dict.fromkeys(name for term in terms for part in split_props(term) for name in part.names))


def averaging_rank(terms: Sequence['VibPerturbedTerm']) -> int:
    """
    Rank of the orientation average: the number of Cartesian slots (op.o) in a term's averaged part.
        e.g. polhess[0,3] * dipgrad[1] * dipgrad[2] -> slots 0, 1, 2, 3 -> rank 4 (all 14 EVV terms)

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
    terms: as derive writes them (pulse IDs) or translated to axes: only the props are read (plan.split_props).

    laser_pol: the experiment's polarization_avg_vector. Its length is fixed by the rank
    (rank 4 -> 3 numbers, e.g. (1, 1, 1) when all pulses and the detector are along x).
    getGeneralPolarizationAveragingExpression would ignore extra numbers without an error.

    The recipe depends only on the rank and laser_pol: it is made once per (rank, laser_pol) and kept
    (_averaging_recipe). Each call returns its own copy, so changing one copy changes no other result.
    Measured 2026-10-13: making a recipe takes 0.13 ms at rank 4, about 5 ms at ranks 5 and 6;
    averaging_rank (run on every call) 1 ms for the 14 EVV terms.
    """
    rank = averaging_rank(terms)
    if not 2 <= rank <= 6:
        raise ValueError(f'orientation averaging supports rank 2 to 6, the terms have rank {rank}')

    n_expected = len(get_iso_f(rank))
    if len(laser_pol) != n_expected:
        raise ValueError(f'rank {rank} needs laser_pol with {n_expected} numbers, got {len(laser_pol)}: {laser_pol}')

    return dict(_averaging_recipe(rank, tuple(laser_pol)))


@cache
def _averaging_recipe(rank: int, laser_pol: tuple[float, ...]) -> PolarizationAveragingRecipe:
    """getGeneralPolarizationAveragingExpression, made once per (rank, laser_pol); make_polarization_linear_comb copies it"""
    return getGeneralPolarizationAveragingExpression(rank=rank, laser_pol=laser_pol)


def evaluate_response_terms(terms: Sequence['VibPerturbedTerm'], *,
                            axes: 'SpectralAxisSet',
                            molsys_data: MolSystemData,
                            states_choice: str,
                            laser_pol: Sequence[float],
                            propagator_param: PropagatorParam,
                            exclude_modes: Collection[int] = ()) -> FeatureResult:
    """
    Terms as derive writes them (in pulse IDs) + data loaded before + experiment settings -> rows and features.
        1. translate the terms to the chosen axes           term_var_translate (copies, keeps the order)
        2. compile                                          compile_terms
        3. orientation-average recipe, rank from the terms  make_polarization_linear_comb (made once per
                                                            rank and laser_pol)
        4. rows, zero pairs, features                       compute_features
    The spectrum on a grid: draw_spectrum(result, window, dim_sizes), as often as needed; the rows stay.

    axes              e.g. make_SpectralAxisSet({'A': [1], 'B': [-1, 2]})
    molsys_data       load_molsys_data(terms, data_origin): loaded once, for any number of runs
    states_choice     must follow the vib analysis regime: GVPT2, VPT2 -> 'anharmonic'; harmonic -> 'harmonic'
    laser_pol         the experiment's polarization_avg_vector, e.g. (1, 1, 1): pulses and detector along x
    propagator_param  cm-1: a number, or a function (vd, freqs) -> cm-1 (features.PropagatorParam), e.g.
                      lambda vd, freqs: 0.01 * abs(vd.energy_difference())
    exclude_modes     modes removed everywhere, counted from 0 as in the data, e.g. (5,) (VibAnaSetup.exclude_modes):
                      no peak and no summed value uses them, as if the molecule did not have them. The other peaks
                      change too: their sums lose the excluded modes (evaluate.included_modes). Default: none.
    Every argument after the terms is keyword-only: two swapped arguments raise a TypeError.
    row.term_id is the index in `terms`.
    """
    translated = term_var_translate.translate_terms_to_axis_variables(list(terms), axes)
    compiled = compile_terms(translated)
    recipe = make_polarization_linear_comb(terms, laser_pol)
    return compute_features(compiled, molsys_data, recipe, propagator_param=propagator_param,
                            states_choice=states_choice, exclude_modes=exclude_modes)


@dataclass(frozen=True)
class EvalRspResult:
    """
    The spectrum on a grid, made by draw_spectrum. Every field is set.
    feature_result   rows, features, and what they came from (molsys, polarization recipe, states_choice)
    grid_axes        1D points per axis, cm-1, both window edges included, e.g. A and B with 201 points each
    grid_result      complex amplitude in au, one value per grid point; array dimension k is the k-th axis of
                     grid_axes, e.g. shape (201, 201). intensity = abs(grid_result)**2
    spectral_window  the window as drawn: its features, each with its box (each drawn on its own box only).
                     full_features: inside the window; contrib_features: outside, with a box that reaches in.
    grid_mesh        (property) the grid points as one full array per axis (ij meshgrid), grid_mesh['A'][i, j]:
                     what the old EvaluatedSpectrum.axes held and the renderer reads
    """
    feature_result: FeatureResult
    grid_axes: dict[str, np.ndarray]
    grid_result: np.ndarray
    spectral_window: SpectralWindow

    @cached_property
    def grid_mesh(self) -> dict[str, np.ndarray]:
        return dict(zip(self.grid_axes, np.meshgrid(*self.grid_axes.values(), indexing='ij')))


def draw_spectrum(feature_result: FeatureResult,
                  window: 'Box',
                  dim_sizes: dict[str, int],
                  *,
                  dynrange: float,
                  eps: float = 0.1,
                  min_gammas: float = 1.,
                  max_cm: float | None = None,
                  a_max: float | None = None) -> EvalRspResult:
    """
    The features of feature_result on a grid over `window`: boxes, grid, drawing. The rows are not computed
    again (the slow part: build_contributions), so a new window, grid, width or box setting runs only the drawing:
        again = draw_spectrum(result, other_window, {'A': 401, 'B': 401}, dynrange=500.)
        wider = draw_spectrum(replace(result, propagator_param=8.), window, dim_sizes, dynrange=500.)
    replace (dataclasses): a new FeatureResult with the same table; its features are made again, with the
    new width, on first use (features_from_rows: 0.02 s for 3524 features on ACDM, F2).

    window     the spectral window, cm-1, one range per axis, e.g. Box({'A': (2600., 2800.), 'B': (-100., 100.)})
    dim_sizes  grid points per axis, both edges included, e.g. {'A': 201, 'B': 201}
    dynrange   the plot's dynamic range (max / floor): the boxes are sized so the cut stays below what the plot
               shows. Required: the plot decides it.
    eps, min_gammas, max_cm, a_max: the box knobs, see features.amplitude_half_widths
    feature_result needs its states (FeatureResult.states), so it must come from compute_features.

    Another box rule: the same three steps, with your own half-widths (cm-1, one per feature):
        spectral_window = SpectralWindow.from_features(window, result.features, half_widths)
        grid_axes, _ = spectral_window.sample_grid(dim_sizes)
        amplitude = draw_window(spectral_window, result.states, grid_axes)
    """
    features = feature_result.features
    half_widths = amplitude_half_widths(window, features, dynrange=dynrange, eps=eps, min_gammas=min_gammas,
                                        max_cm=max_cm, a_max=a_max)                       # boxes: width at the peak
    spectral_window = SpectralWindow.from_features(window, features, half_widths)
    grid_axes, _ = spectral_window.sample_grid(dim_sizes)
    grid_result = draw_window(spectral_window, feature_result.states, grid_axes)          # width on the grid
    return EvalRspResult(feature_result, grid_axes, grid_result, spectral_window)
