"""

[] 
[] 
---

==> list[SpectralFeature]
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionRow,
    ContributionTable,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResLocPoint,
)
from wilson_suite.wilson_system.system_data import (
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    VibDiff,
)
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.refac_rsp_eval.features import SpectralFeature
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
        CompiledTerm,
        FreqTermsCollection,
        PropsCollection,
        ResCondKey,
        ResonanceMotif,
    )
    from wilson_suite.wilson_system.system_data import VibStatesData


@dataclass
class PrecalculatedData:
    """
    TODO/FIXME: avrg_tensors should carry polarization info.

    if avrg_expr in pre.avrg_expr_tensor_mapping:
    """
    avrg_tensors: dict = field(default_factory=dict)
    avrg_expr_tensor_mapping: dict = field(default_factory=dict)
    vibenedenoms_tensors: dict = field(default_factory=dict)
    polarization_vec: tuple = ()

"""
The data then passes through these steps:

1. build the request      "please fetch 'polgrad' from CFOUR"          plan.py:158
2. obtainer returns       data_dict = {'polgrad': array, ...}
3. fill mol_props         put data_dict['polgrad'] into the 'polgrad' slot   (slot name from from_polprop, evaluate.py:423)
4. evaluate               read mol_props['polgrad'].vals[mode, i, j]   evaluate.py:725 (averaged)                                                                       evaluate.py:618 (non-averaged)
"""

## ----------------------------------------------------
##          coefficient per index set/ParameterSet
## ----------------------------------------------------

# EVALUATION OF A SINGLE TERM - ONE INDEX SET: SUM OVER SET
def evaluate_term_coeff_sumover(compl_term: 'CompiledTerm',
                         idx_dict: dict,
                         molsys_data: 'MolSystemData',
                         polarization_linear_comb: dict | None = None,
                         precalculated_data: PrecalculatedData | None = None,
                         zero_tol: float = 1e-18) -> tuple[float, dict]:
    """
    Coefficient of one term at one resonance location.

    idx_dict must fix every resonance label (compl_term.idx_nonsumm): these labels decide where
    the peak sits. Every label left out is summed over all modes. The summed pieces all sit at
    the same location, so adding them is safe.

    e.g. a term with resonance labels (a, b) and summation label c:
        {'a': 0, 'b': 1}  ->  f(0,1,0) + f(0,1,1) + ... + f(0,1,n_modes-1)
        {'a': 0}          ->  ValueError: summing over b would add peaks at different positions

    returns (total, leaves): leaves maps each full index set to its factors
    """
    not_fixed = [i for i in compl_term.idx_nonsumm if i not in idx_dict]
    if not_fixed:
        raise ValueError(f'resonance labels {not_fixed} must be fixed, not summed')

    term_idx_all = sorted(compl_term.idx_summ + compl_term.idx_nonsumm)
    
    if molsys_data.eigenvals is not None:
        n_modes = len(molsys_data.eigenvals)
    else: raise ValueError('molsys_data.eigenvals is absent - number of modes is required')

    # only needed when the avrg value isn't looked up from a precalculated tensor
    avrg_func = None
    if precalculated_data is None or compl_term.avrg_props not in precalculated_data.avrg_expr_tensor_mapping:
        if polarization_linear_comb is None:
            raise ValueError('polarization_vec is required when the avrg tensor is not precalculated')
        avrg_func = _make_func_to_compute_avrg(avrg_expression=compl_term.avrg_props,
                                               polarization_linear_comb=polarization_linear_comb)

    def sum_over(index_dict: dict, remaining: list, leaves: dict) -> float:
        if not remaining:
            value, contribs = evaluate_full_index_dict(compl_term, index_dict, 
                                                       molsys_data,
                                                       avrg_func,
                                                       precalculated_data,
                                                       zero_tol)
            leaves[ParameterSet(index_dict)] = contribs
            return value
        current, rest = remaining[0], remaining[1:]
        return sum(sum_over({**index_dict, current: v}, rest, leaves) for v in range(n_modes))
    
    missing = [i for i in term_idx_all if i not in idx_dict]
    leaves = {}
    return sum_over(idx_dict, missing, leaves), leaves


# EVALUATION OF A SINGLE TERM - ONE INDEX SET: FULL INDEX SET
def evaluate_full_index_dict(compl_term: 'CompiledTerm', index_dict: dict,
                               molsys_data: 'MolSystemData',
                               avrg_func: Callable | None = None,
                               precalculated_data: PrecalculatedData | None = None,
                               zero_tol: float = 1e-18) -> tuple[float, dict]:
    """
    Evaluate the term for a single index dictionary.
    Parameters:
        (same as evaluate_term_coeffs)
    
    Returns:
        float: The computed coefficient for the given index dictionary.
    ---
        compl_term: 'CompiledTerm',    # term
        index_dict: dict,            # index choice (a,b,c...)
    """
    # requested nm indices dict should hold values for all indices in the term
    t_indices = sorted(set(compl_term.idx_summ+compl_term.idx_nonsumm))
    if not all(index in list(index_dict.keys()) for index in t_indices):
        raise ValueError('term has indices that do not have values in index_dict.')


    # Evaluate AVRG
    AVRG = eval_avrg_per_indexdict(compl_term.avrg_props, index_dict,
                                   avrg_func=avrg_func,
                                   molsys_data=molsys_data, 
                                   precalculated_data=precalculated_data,
                                   zero_tol=zero_tol)

    if AVRG == 0.0:
        return 0.0, {'AVRG': AVRG}

    # Evaluate NON_AVRG
    NON_AVRG = eval_non_avrg_per_indexdict(compl_term.non_avrg_props, index_dict, molsys_data, zero_tol)
    if NON_AVRG == 0.0:
        return 0.0, {'NON_AVRG': NON_AVRG, 'AVRG': AVRG}

    # Evaluate VIBDIFF_TERMS
    extra_freqterms = compl_term.cmp_freqdenom.get_pert_wf_diff()

    VIBDIFF_TERMS = otf_vibdiffdenom(extra_freqterms, index_dict, molsys_data)

    # Evaluate VIBENE_DENOM
    freqterms = compl_term.cmp_freqdenom.get_vibenedenom()

    if precalculated_data is None or not precalculated_data.vibenedenoms_tensors:

        VIBENE_DENOM = harmonic_denom(freqterms, index_dict, molsys_data)
    else:
        VIBENE_DENOM = eval_vibenedenom(freqterms, index_dict, precalculated_data)


    # Compute the product
    product_all = NON_AVRG * AVRG * VIBDIFF_TERMS * VIBENE_DENOM

    dict_contribs = {'NON_AVRG': NON_AVRG, 'AVRG': AVRG, 'VIBDIFF_TERMS': VIBDIFF_TERMS, 'VIBENE_DENOM': VIBENE_DENOM}

    return float(compl_term.frac_factor) * float(product_all), dict_contribs


# EVALUATION OF TERM PARTS FOR ONE INDEX SET
def eval_non_avrg_per_indexdict(non_avrg_expr: 'PropsCollection',
                                index_dict: dict, 
                                molsys_data: 'MolSystemData',
                                zero_tol: float = 1e-18):
    """
    non_avrg_expr - extracted part of VibPerturbed term 

    order of indices generally: a,b,c,... 
    E.g. in CFF tensor index tuple is (a,b,c)

    TODO: error handling for missing or invalid data for all functions below
    """
    product_all = 1.
    
    for non_avrg_prop in non_avrg_expr:
        # accessing values for non-averaged properties from data
        if non_avrg_prop.inds is None or len(non_avrg_prop.inds) == 0:
            # If there are no indices, we assume it's a scalar property
            na_prop_inds = ()
        else:
            na_prop_inds = tuple([index_dict[i] for i in non_avrg_prop.inds])
        
        triv_name = prop_trivname(ord_el=len(non_avrg_prop.ops), ord_geo=non_avrg_prop.dord)

        prop: MolecularProperty = molsys_data.mol_props[triv_name]
        NON_AVRG = prop.vals[na_prop_inds]

        if np.isclose(NON_AVRG, 0.0, atol=zero_tol, rtol=0.0):
            return 0.
        else:
            product_all *= NON_AVRG
    
    return product_all


def _get_ind_tuple_from_base(expr: 'PropsCollection', base_expr: 'PropsCollection', index_dict: dict):
    """
    Map expr to indices according to base expression's unique symbols.
    
    TODO: expand docs
    """
    base_unique = sorted(set(base_expr.get_mode_indices()))
    expr_inds = expr.get_mode_indices()

    if len(base_unique) < len(expr_inds):
        # walk through only base_unique ind labels
        return tuple(index_dict[sym] for sym in base_unique)
    elif len(base_unique) == len(expr_inds):
        # walk through all expr_inds labels, there are repeated labels
        return tuple(index_dict[sym] for sym in expr_inds)
    else:
        # this should not be possible in the worflow
        raise ValueError('This base_expr cannot be a base expression for this expr')


# EVALUATION OF TERM PARTS FOR ONE INDEX SET
def eval_avrg_per_indexdict(avrg_expr: 'PropsCollection',
                            index_dict: dict, *,
                            avrg_func: Callable | None = None,
                            molsys_data: MolSystemData | None = None,
                            precalculated_data: PrecalculatedData | None = None,
                            zero_tol: float = 1e-18):
    """
    if precalculated_data provided - use it;
    otherwise - compute with molsys_data
    ---
    Averaging papers
    https://pubs.aip.org/aip/jcp/article/67/11/5026/788630/On-three-dimensional-rotational-averages
    https://pubs.aip.org/aip/jcp/article/141/20/204103/193500/Rotational-averaging-of-multiphoton-absorption
    """
    if precalculated_data is not None and avrg_expr in precalculated_data.avrg_expr_tensor_mapping:
        avrg_tensor_expr = precalculated_data.avrg_expr_tensor_mapping[avrg_expr]
        avrg_tensor = precalculated_data.avrg_tensors[avrg_tensor_expr]
        result = avrg_tensor[_get_ind_tuple_from_base(avrg_expr, avrg_tensor_expr, index_dict)]
    elif avrg_func is not None and molsys_data is not None:
        result = avrg_func(index_dict, molsys_data.mol_props)
    else:
        raise ValueError('avrg_expr not in precalculated_data and no avrg_func/molsys_data given')

    return 0. if np.isclose(result, 0.0, atol=zero_tol, rtol=0.0) else result


def _make_func_to_compute_avrg(*,
                              avrg_expression: 'PropsCollection',
                              polarization_linear_comb: dict
                              ) -> Callable[[dict, 'MolPropsCollection'], float]:
    """
    for an expression with properties data values,
    compute average with given polarization setup for a choice of normal mode indices

    """

    def compute_for_idx_choice(index_choices: dict, props_data: 'MolPropsCollection') -> float:
        """
        index_choices: dict, props_data: 'MolPropsCollection'
        """

        if not isinstance(props_data, MolPropsCollection):
            raise TypeError(
                f"props_data must be a MolPropsCollection, got {type(props_data).__name__}"
            )

        # Validate index_choices has all required keys
        required_inds = {i for prop in avrg_expression for i in prop.inds} # type: ignore
        missing = required_inds - index_choices.keys()
        if missing:
            raise KeyError(
                f"index_choices is missing required mode indices: {missing}"
            )

        from wilson_suite.wilson_utils.prop_trivname import prop_trivname

        total = 0.

        for cart_axes in polarization_linear_comb:

            # Comment (MR): Noting that I considered if there would be any issues with this in generalized routine,
            # couldn't think of any but want to discuss and double check for safety

            product = 1.

            for prop in avrg_expression:

                prop_tuple_key = prop_trivname(ord_el=len(prop.ops), ord_geo=prop.dord)

                nm_inds = tuple([index_choices[i] for i in prop.inds]) # type: ignore
                cart_inds = tuple([cart_axes[i.o] for i in prop.ops])
                all_inds = (*nm_inds, *cart_inds)

                # retrieve data for property (prop_key) and idxs_key which is (tuple(mode inds), tuple(cart inds))
                product *= props_data[prop_tuple_key].vals[all_inds]
                
            # if product != 0.:
            #     logger.debug(f"Avrg prop contribution for indices {index_choices} and cart axes {cart_axes} with coefficient {polarization_linear_comb[cart_axes]}: {product}")

            total += product * polarization_linear_comb[cart_axes]

        return total

    return compute_for_idx_choice


# EVALUATION OF THE FULL AVRG TENSOR
def calculate_avrg_tensor(avrg_expression: 'PropsCollection',
                          props_data: 'MolPropsCollection',
                          number_of_nmodes: int,
                          polarization_linear_comb: dict | None = None,
                          modes_to_fill: list[int] | None = None) -> np.ndarray:
    """
    Precalculating the full tensor for given avrg_expression

    modes_to_fill - could be generated with for all normal modes with:
        modes_to_fill: list[int] = list(range(number_of_nmodes))

    """
    if polarization_linear_comb is None:
        polarization_linear_comb = {}
    if modes_to_fill is None:
        modes_to_fill = list(range(number_of_nmodes))
    if max(modes_to_fill) >= number_of_nmodes:
        raise ValueError(f"modes_to_fill contains indices exceeding number_of_nmodes ({number_of_nmodes})")
    
    # so indices are in alphabetical order in full_tensor below
    mode_inds = sorted(set(avrg_expression.get_mode_indices()))  # list, deterministic order

    from wilson_suite.wilson_intensities.amplitudes.utils import (
        generate_index_choices_general,
    )

    ind_choices: list[dict[str, int]] = generate_index_choices_general(indlabels_in_motif=mode_inds, 
                                                                       labels=modes_to_fill)

    # Indicating generalized version for updating
    func_general = _make_func_to_compute_avrg(avrg_expression=avrg_expression,
                                              polarization_linear_comb=polarization_linear_comb)

    full_tensor = np.zeros((number_of_nmodes,)*len(mode_inds))

    for idx in ind_choices:
        # so indices are in alphabetical order
        full_tensor[tuple(idx[k] for k in mode_inds)] = func_general(idx, props_data)

    return full_tensor


# EVALUATION OF TERM PARTS FOR ONE INDEX SET - from precalculated_data
def eval_vibenedenom(freqterms: 'FreqTermsCollection',
                     index_dict: dict,
                     precalculated_data: PrecalculatedData):
    """
    try without precalculated data
    """
    vibenedenoms_tensor = precalculated_data.vibenedenoms_tensors[freqterms.get_num_indices_vibenedenom()]

    vibeneden_index_tuple = tuple([index_dict[i] for i in freqterms.get_num_indices_vibenedenom()])

    return vibenedenoms_tensor[vibeneden_index_tuple]


# EVALUATION OF TERM PARTS FOR ONE INDEX SET - on the fly
def otf_vibdiffdenom(freqterms: 'FreqTermsCollection', index_dict: dict, molsys_data: 'MolSystemData'):
    """
    without precalculated data
    """
    product_all = 1.

    for vibdiff in freqterms:
        vib_diff_w_value = VibDiff.from_symbolic(vibdiff, index_dict, molsys_data.states)

        if vib_diff_w_value.energy_difference(au=True) == 0.:
            raise ZeroDivisionError("VibDiff is zero - division by zero")
        product_all *= 1./ vib_diff_w_value.energy_difference(au=True)

    return product_all

def harmonic_denom(freqterms: 'FreqTermsCollection', index_dict: dict, molsys_data: 'MolSystemData') -> float:
    """product of 1/omega_i, harmonic omega in Eh; one mode per denominator"""
    if molsys_data.eigenvals is not None:
        product = 1.
        for vd in freqterms:
            if vd.sr.q: # type: ignore
                raise ValueError(f"this VibDiffTerm's ket state is not a ground state: {vd}")
            
            (label,) = vd.sl.q          # type: ignore ; raises if a denominator has more than one mode

            product /= convNu2Ene(molsys_data.eigenvals[index_dict[label]])
        return product
    else:
        raise ValueError("molsys_data.eigenvals is None")


## ----------------------------------------------------------------
##          resonanace on freq grid per index set/ParameterSet
## ----------------------------------------------------------------


def generate_LHS_motif(motif: 'ResonanceMotif',
                       axes: Iterable[str] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    motif is a tuple/collection of res_conditions
        res_conditions is a tuple of (vib_difference, axes)
            vib_difference is a tuple of states indices

    returns (coeff_matrix, axes): one row per condition, one column per axis in `axes`
    (sorted; defaults to the distinct axes of the motif). Axes the motif does not
    mention get an all-zero column.
    """
    motif_axes = motif.get_max_different_freq_axes()
    all_axes = tuple(sorted(motif_axes if axes is None else set(axes)))
    if not motif_axes <= set(all_axes):
        raise ValueError(f"axes {all_axes} do not cover the axes of {motif}: {sorted(motif_axes)}")
    
    col = {ax: j for j, ax in enumerate(all_axes)}

    coeff_matrix = np.zeros((len(motif), len(all_axes)))

    for i, r_cond_key in enumerate(motif):
        # ('A', '-B') --> {'A': 1, 'B': -1}
        coeffs = {var.strip('-') : 1 if '-' not in var else -1 for var in r_cond_key.pf}

        for alpha_label, coefficient in coeffs.items():
             # minus the axis sign: 'A' -> -1, '-B' -> +1. With get_RHS_motif's -E the row reads
             # -(signed sum of pf) = -E, i.e. E - (signed sum of pf) = 0 (derive's convention)
             coeff_matrix[i, col[alpha_label]] = -1 * np.sign(coefficient)

    return coeff_matrix, all_axes


def get_RHS_motif(motif: 'ResonanceMotif',
            parameters: ParameterSet, vibstates_data: 'VibStatesData',
            unit: str='Eh'):
    """
    making a constants vector from a list of tuples
    resonance_tuples = [(1, (-1,)), (2, (-1, 2)), (3, (-2, 3))]
    ind_tuple = (1, 2, 3) --- 
    vibdiffbank: VibDiffBank instance

    output: [5, -3, 2]
    """
    if unit not in ('Eh', 'cm-1'):
        raise ValueError(f"unit must be 'Eh' or 'cm-1', got {unit!r}")

    constants = []

    for res_cond_key in motif:
        vib_diff_w_value = VibDiff.from_quanta(*res_cond_key.diff, parameters.to_dict(), vibstates_data)
        constants.append((-1)*vib_diff_w_value.energy_difference(au=(unit=='Eh')))

    return constants


def solve_LSE_motif(motif: 'ResonanceMotif',
                    parameters: ParameterSet, vibdata: 'VibStatesData',
                    unit: str='Eh') -> ResLocPoint:
    """
    Find the point in frequency space where all resonance conditions of `motif` hold at once.

    Each condition is one linear equation in the perturbing frequencies, so the motif is
    a linear system A @ w = b:
        A  one row per condition, one column per axis of the motif (from generate_LHS_motif)
        b  vibrational energy differences for this index assignment (from get_RHS_motif)
        w  the frequency of each axis at resonance - what we solve for

    e.g. axes (A, B), conditions on A and on A - B:
        A = [[-1,  0],       b = [b0, b1]    ->  w_A from row 0,
             [-1,  1]]                           then w_B from row 1

    The location is {axis: w} over the motif's own axes: always a point, every axis has a value.

    raises np.linalg.LinAlgError if the location is not a point (fewer independent conditions
    than axes, e.g. a single condition on A + B) or the conditions are inconsistent.
    """
    if len(motif) == 0:
        raise ValueError('motif has no resonance conditions: no resonance location')

    A, all_axes = generate_LHS_motif(motif)
    b = np.array(get_RHS_motif(motif, parameters, vibdata, unit))

    # lstsq always returns an answer: the exact solution if there is one, otherwise the
    # w minimising |A @ w - b|. So both checks below are needed to trust it.
    # rank = number of independent conditions.
    solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

    # A point needs one value per axis, so as many independent conditions as axes.
    # Fewer leaves a line (or more): e.g. one condition on A + B gives rank 1 < 2,
    # and resonance is the whole diagonal w_A + w_B = b.
    if rank < len(all_axes):
        raise np.linalg.LinAlgError(f"resonance location of {motif} is not a point "
                                    f"(rank {rank} < {len(all_axes)} axes)")
    # More conditions than unknowns may contradict each other (same axis, different energies);
    # then lstsq's best compromise does not actually satisfy A @ w = b.
    if not np.allclose(A @ solution, b):
        raise np.linalg.LinAlgError(f"resonance conditions of {motif} are inconsistent: no resonance location")

    return ResLocPoint({ax: float(val) for ax, val in zip(all_axes, solution)})


## ------------------------------------------------------------------
##          loops over collections [terms,index sets,res motifs]
## ------------------------------------------------------------------

def make_idx_sets(n_modes: int, mode_labels: Sequence) -> list[dict[str, int]]:
    """
    n_modes - number of normal modes
    mode_labels = sorted(set(avrg_expression.get_mode_indices()))
    """
    # all possible labels - tuple(range(n_modes))
    nm_inds_choices = tuple(range(n_modes))

    import itertools
    return [dict(zip(mode_labels, combo)) for combo in itertools.product(nm_inds_choices, repeat=len(mode_labels))]


def build_contributions(terms: Sequence['CompiledTerm'], molsys_data: MolSystemData,
                        polarization_linear_comb=None, precalculated_data=None
                        ) -> tuple[ContributionTable, list[tuple[int, ParameterSet]], list[tuple[int, ParameterSet, float]]]:
    """
    One row per (term, index set) that adds to the spectrum. Index sets fix the term's
    resonance labels; evaluate_term_coeff_sumover sums the rest. Locations are in cm-1.
    The number of modes comes from molsys_data.eigenvals, the state energies from molsys_data.states.

    returns (rows, zero, failed), so that len(rows) + len(zero) + len(failed) == number of pairs:
        zero   - (term_id, params) with coeff == 0: no row, no location solved
        failed - (term_id, params, coeff) with no single resonance point (LinAlgError)
    """
    if molsys_data.eigenvals is None:
        raise ValueError('molsys_data.eigenvals is absent - number of modes is required')
    n_modes = len(molsys_data.eigenvals)

    locations: dict[tuple[ResonanceMotif, ParameterSet], ResLocPoint] = {}
    rows, zero, failed = [], [], []
    for term_id, term in enumerate(terms):
        motif = term.cmp_resmotf
        for idxset in make_idx_sets(n_modes, sorted(term.idx_nonsumm)):
            ps = ParameterSet(idxset)

            coeff, _ = evaluate_term_coeff_sumover(term, idxset, molsys_data,
                                                   polarization_linear_comb, precalculated_data)
            # not saving 0 coeffs
            if coeff == 0.:
                zero.append((term_id, ps))
                continue
            key = (motif, ps)
            
            # one solve per (motif, params)
            if key not in locations:
                try:
                    locations[key] = solve_LSE_motif(motif, ps, molsys_data.states, unit='cm-1')
                
                # Case 1: the resonance is a line, not a point. 
                # Case 2: the resonance conditions are inconsistent, no solution.
                except np.linalg.LinAlgError:
                    failed.append((term_id, ps, coeff))
                    continue
            rows.append(ContributionRow(term_id, motif, ps, locations[key], coeff))
    return ContributionTable(rows), zero, failed


## -----------------------------------------------------------------------------
##          evaluating resonance conditions (resonance factors) on the grid
## -----------------------------------------------------------------------------

## evaluating func(resloc, parameter) for each resonance condition in a motif, then multiplying them together


def lorentzian_propagator(resloc: float | np.ndarray, gamma: float | np.ndarray) -> np.ndarray | complex:
    """
    1 / (resloc - i*gamma); gamma in the same units as resloc, one number or one per grid point.
    With resloc = E - freq (eval_rescond_resloc) this is 1 / (E - freq - i*gamma).
    """
    return 1 / (resloc - 1j * gamma)


# evaluating resloc for, e.g., 1 / (resloc - i*gamma) - lorentzian propagator
def eval_rescond_resloc(res_cond_key: 'ResCondKey',
                         vib_diff: VibDiff,
                         grid: dict[str, np.ndarray]) -> float | np.ndarray:
    """
    Distance from resonance of one condition at every grid point: E - freq, in Eh.
        E    = vib_diff.energy_difference(au=True) = E_left - E_right
        freq = signed sum of the grid axes in res_cond_key.pf, e.g. ('-A', 'B') -> B - A
    Zero where freq = E: the same point solve_LSE_motif finds. grid must be in Eh.

    Sign: E - freq, because derive defines a resonance condition that way.
      - ResonanceCondition (wilson_derive/abstractions.py): "perturbing frequencies to be subtracted".
        VibPerturbedTerm.to_str prints each condition as (<sl sr> - pf - iG).
        So one condition is the denominator (E_sl - E_sr) - (signed sum of pf).
      - pf signs: '-A' means axis A enters that sum with -1. The sign comes from the pulse
        interaction (+w or -w, dressWithPulseInteractions) and from the axis choice
        (e.g. B = -w1 + w2, term_var_translate).
      - The term coefficients from derive are made for this denominator, e.g. the -1 / +1 of the
        D1 / D2 terms in vib_rsp_sos. freq - E would flip every condition: an extra (-1)^N for
        a motif with N conditions, which the coefficients do not know about.
      - -i*gamma is not fixed by derive (no damping there). +i*gamma gives the complex conjugate
        everywhere, so |amplitude|^2 stays the same. -i*gamma matches to_str and the old code
        (amplitudes/evaluators.py, evaluate_resonance_motif).
    """
    freq = sum((-1 if ax.startswith('-') else 1) * grid[ax.lstrip('-')] for ax in res_cond_key.pf)
    return vib_diff.energy_difference(au=True) - freq


# evaluating full resonance factor, e.g., 1 / (resloc1 - i*gamma1) / ...
def eval_resonance_factor(motif: 'ResonanceMotif',
                          parameters: ParameterSet,
                          vibstates_data: 'VibStatesData',
                          grid: dict[str, np.ndarray],
                          propagator: Callable[[float|np.ndarray, float|np.ndarray], complex],
                          propagator_param: Callable[[VibDiff], float|np.ndarray]) -> complex | np.ndarray: # FIXME change VibDiff to any or smth
    """
    One motif consists of resonance conditions from a single term.

    propagator - function that takes (resloc, parameter) and returns the resonance factor, e.g., lorentzian_propagator
    """
    result = 1.

    for res_cond_key in motif:

        vd = VibDiff.from_quanta(*res_cond_key.diff, index_dict=parameters.to_dict(), vibstates_data=vibstates_data)

        resloc = eval_rescond_resloc(res_cond_key, vd, grid)

        result *= propagator(resloc, propagator_param(vd))

    return result


# feature is one amplitude (the sum of the row coefficients) times one resonance factor
def eval_feature_on_grid(feature: 'SpectralFeature',
                         vibstates_data: 'VibStatesData',
                         grid_cm: dict[str, np.ndarray],
                         propagator: Callable=lorentzian_propagator,
                         propagator_param: Callable[[VibDiff], float | np.ndarray] | None = None) -> complex | np.ndarray:
    """
    amplitude_coeff * resonance factor of the feature's motif. grid in cm-1, result in au.

    propagator_param - the propagator's parameter for each condition, in cm-1 (converted to Eh here),
                       e.g. lambda vd: 5. for a 5 cm-1 width on every condition.
                       None -> feature.lineshape_parameter on every condition.
                       For a parameter that depends on the frequencies, read grid_cm inside the function
                       and return an array of the grid's shape, e.g. lambda vd: 0.01 * grid_cm['A'].

    FIXME (todo.txt E5): feat_box and get_intensity use feature.lineshape_parameter, not propagator_param.
          A propagator_param with another width (or one that changes over the grid) gives a peak they do
          not match, e.g. a box too small for a wider peak.
    """
    if propagator_param is None:
        if feature.lineshape_parameter is None:
            raise ValueError(f'feature {feature} has no lineshape_parameter and no propagator_param was given')
        gamma_au = convNu2Ene(feature.lineshape_parameter)
        param_au = lambda _vd: gamma_au
    else:
        param_au = lambda vd: convNu2Ene(propagator_param(vd))

    if feature.motif is None:
        raise ValueError(f'feature {feature} has no motif')
    
    if feature.amplitude_coeff is None:
        raise ValueError(f'feature {feature} has no amplitude_coeff')
    
    grid_au = {ax: convNu2Ene(v) for ax, v in grid_cm.items()}

    resonance_factor = eval_resonance_factor(feature.motif, feature.rows[0].params, vibstates_data,
                                             grid_au, propagator, propagator_param=param_au)

    return feature.amplitude_coeff * resonance_factor
    # return np.asarray(feature.amplitude_coeff * resonance_factor)



"""
1. compiled terms
2. data request dict
3. molsys data
4. evaluating `term coeff parts` per index set  [eval coeff]    [x]
5. evaluating `term coeff full` per index set   [eval coeff]    [x]
6. evaluating `term res cond - res location`    [res loc]       [x]
7. evaluating `term res cond - on the grid`     [res loc]       []

---------
ResonanceMotif + ParameterSet + VibStatesData ==> ResLocPoint

iterate over:
    1. Sequence[ResonanceMotif]
    2. Sequence[ParameterSet] for one ResonanceMotif

-----

coefficient should be attached to resonance with the same ParameterSet

"""

