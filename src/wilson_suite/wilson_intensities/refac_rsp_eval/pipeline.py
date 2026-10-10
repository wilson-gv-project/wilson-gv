"""
Entry point: compiled terms + molecular data -> spectral features.

compute_features_from_terms does everything from the terms derive writes (in pulse IDs):
translate to axes -> compile_terms -> load_molsys_data -> make_polarization_linear_comb -> compute_features.

CompiledTerm list
  │ 1. index sets: fix the motif labels (make_idx_sets over term.idx_nonsumm)
  │ 2. coeff: evaluate_term_coeff_sumover, sums the other labels     ← MolSystemData, polarization
  │    zero coeff -> `zero`, no row
  │ 3. location: solve_LSE_motif, once per (motif, params), cm-1     ← molsys_data.states
  │    always one point: compile_terms checked the motif (B7)
  v
ContributionRow(term_id, motif, params, location, coeff)   "why" layer    evaluate.build_contributions
  │ 4. group by (motif, location), add the coeffs
  v
SpectralFeature(location, rows)                            "what" layer   features.features_from_rows
  │ 5. peak width, spectral window, boxes
  v
grid.py → spectrum array                                   E2, not built yet
"""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from wilson_suite.wilson_derive import term_var_translate
from wilson_suite.wilson_intensities.amplitudes.averaging import (
    get_iso_f,
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    PrecalculatedData,
    build_contributions,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionTable,
    SpectralFeature,
    features_from_rows,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    compile_terms,
)
from wilson_suite.wilson_system.system_data import (
    DataOriginInfo,
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    build_data_request_for_term,
)
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    from wilson_suite.wilson_experiment.indep_vars_and_axes import SpectralAxisSet
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import CompiledTerm


@dataclass(frozen=True)
class FeatureResult:
    table: ContributionTable                          # all rows: the source
    zero: list[tuple[int, ParameterSet]]              # pairs with coeff == 0
    lineshape_parameter: float | None = None          # cm-1

    @cached_property
    def features(self) -> list[SpectralFeature]:
        """One feature per (motif, location), built from the table on first use."""
        return features_from_rows(self.table, self.lineshape_parameter)

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
                     polarization_linear_comb: dict | None = None,
                     precalculated_data: PrecalculatedData | None = None,
                     lineshape_parameter: float | None = None) -> FeatureResult:
    """
    One SpectralFeature per (motif, location). lineshape_parameter in cm-1.
    row.term_id is the index in `compiled`, and so in the term list it was compiled from.
    """
    table, zero = build_contributions(compiled, molsys_data, polarization_linear_comb, precalculated_data)
    return FeatureResult(table, zero, lineshape_parameter)


def load_molsys_data(compiled: Sequence['CompiledTerm'], data_origin: DataOriginInfo,
                     states_choice: str) -> MolSystemData:
                    #  vibstates_regime: str) -> MolSystemData:
    """
    One MolSystemData for all terms, each property once: one wilson_data_obtainer call, one from_datadict.
    states_choice must follow the vib analysis regime: GVPT2, VPT2 -> 'anharmonic'; harmonic -> 'harmonic'.
    """
    # if vibstates_regime not in ('harmonic', 'VPT2', 'GVPT2'):
    #     raise ValueError(f'vibstates_regime must be "harmonic" or "VPT2" or "GVPT2", got {vibstates_regime}')
    
    if states_choice not in ('harmonic', 'anharmonic'):
        raise ValueError(f'states_choice must be "harmonic" or "anharmonic", got {states_choice}')
    
    request, props = {}, {}
    for term in compiled:
        request.update(build_data_request_for_term(term, data_origin))
        for p in term.all_props:
            mp = MolecularProperty.from_polprop(p)
            props.setdefault(mp.trivial_name, mp)
    datadict = wilson_data_obtainer(requested_data_dict=request)
    
    return MolSystemData.from_datadict(MolPropsCollection(list(props.values())), datadict,
                                       states_choice=states_choice)


def averaging_rank(compiled: Sequence['CompiledTerm']) -> int:
    """
    Rank of the orientation average: the number of Cartesian slots (op.o) in a term's averaged part.
        e.g. polhess[0,3] * dipgrad[1] * dipgrad[2] -> slots 0, 1, 2, 3 -> rank 4 (all 14 EVV terms)

    compute_features uses one recipe for every term, so all terms must have the same slots.
    The recipe puts slot i at position i, so the slots must be 0 .. rank-1, each once.
    """
    if not compiled:
        raise ValueError('no terms: no averaging rank')

    term_ids_by_slots = defaultdict(list)
    for term_id, term in enumerate(compiled):
        term_ids_by_slots[tuple(sorted(term.avrg_props.get_cart_axes()))].append(term_id)

    if len(term_ids_by_slots) > 1:
        raise ValueError(f'terms differ in their averaged slots, one recipe cannot serve them all; '
                         f'slots: term ids {dict(term_ids_by_slots)}')
    (slots,) = term_ids_by_slots
    if slots != tuple(range(len(slots))):
        raise ValueError(f'averaged slots must be 0 .. {len(slots) - 1}, each once; got {slots}')
    return len(slots)


def make_polarization_linear_comb(compiled: Sequence['CompiledTerm'],
                                  laser_pol: Sequence[float]) -> dict[tuple, float]:
    """
    The orientation-average recipe {(x/y/z per slot): coefficient} that compute_features takes,
    from getGeneralPolarizationAveragingExpression with the rank of the terms (averaging_rank).

    laser_pol: the experiment's polarization_avg_vector. Its length is fixed by the rank
    (rank 4 -> 3 numbers, e.g. (1, 1, 1) when all pulses and the detector are along x).
    getGeneralPolarizationAveragingExpression would ignore extra numbers without an error.
    """
    rank = averaging_rank(compiled)
    if not 2 <= rank <= 6:
        raise ValueError(f'orientation averaging supports rank 2 to 6, the terms have rank {rank}')

    n_expected = len(get_iso_f(rank))
    if len(laser_pol) != n_expected:
        raise ValueError(f'rank {rank} needs laser_pol with {n_expected} numbers, got {len(laser_pol)}: {laser_pol}')

    return getGeneralPolarizationAveragingExpression(rank=rank, laser_pol=tuple(laser_pol))


def compute_features_from_terms(terms: Sequence['VibPerturbedTerm'], *,
                                axes: 'SpectralAxisSet',
                                data_origin: DataOriginInfo,
                                states_choice: str,
                                laser_pol: Sequence[float],
                                lineshape_parameter: float | None = None) -> FeatureResult:
    """
    Terms as derive writes them (in pulse IDs) + data file + experiment settings -> features.
        1. translate the terms to the chosen axes          term_var_translate (copies, keeps the order)
        2. compile                                         compile_terms
        3. load the data for all terms, each property once  load_molsys_data
        4. orientation-average recipe, rank from the terms  make_polarization_linear_comb
        5. rows, zero pairs, features                      compute_features

    axes                 e.g. make_SpectralAxisSet({'A': [1], 'B': [-1, 2]})
    states_choice        must follow the vib analysis regime: GVPT2, VPT2 -> 'anharmonic'; harmonic -> 'harmonic'
    laser_pol            the experiment's polarization_avg_vector, e.g. (1, 1, 1): pulses and detector along x
    lineshape_parameter  cm-1
    row.term_id is the index in `terms`.
    """
    translated = term_var_translate.translate_terms_to_axis_variables(list(terms), axes)
    compiled = compile_terms(translated)
    molsys = load_molsys_data(compiled, data_origin, states_choice)
    polarization = make_polarization_linear_comb(compiled, laser_pol)
    return compute_features(compiled, molsys, polarization, lineshape_parameter=lineshape_parameter)

 