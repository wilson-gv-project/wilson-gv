"""

[] 
[] 
---

==> list[SpectralFeature]
"""

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.features import SpectralFeature
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
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
        CompiledTerm,
        FreqTermsCollection,
        PropsCollection,
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
             # Reverse the sign (FIXME???) and place it in the correct position
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


def coefficient_compute_loop(ids_sets: list[dict[str, int]], 
                                 term: 'CompiledTerm', molsys_data: MolSystemData,
                                 polarization_linear_comb: dict | None=None, 
                                 precalculated_data: PrecalculatedData | None=None) -> dict[ParameterSet, tuple[float, dict]]:
    
    results = {}
    for idxset in ids_sets:
        coeff_evals = evaluate_term_coeff_sumover(term, idxset, molsys_data, 
                                                  polarization_linear_comb, precalculated_data)
        results[ParameterSet(idxset)] = coeff_evals
    
    return results


def resonance_compute_loop(ids_sets: list[dict[str, int]], 
                               motif: 'ResonanceMotif', 
                               vibstates_data: 'VibStatesData') -> dict[ParameterSet, ResLocPoint]:
    
    results = {}
    for idxset in ids_sets:
        reslocpoint = solve_LSE_motif(motif, ParameterSet(idxset), vibstates_data)
        results[ParameterSet(idxset)] = reslocpoint
    
    return results


def get_motifs_feats(motifs: Sequence['ResonanceMotif'], 
                    ids_sets: list[dict[str, int]], 
                    vibstates_data: 'VibStatesData') -> dict[ParameterSet, dict['ResonanceMotif', 'SpectralFeature']]:
    """
    motifs - sequence of ResonanceMotif
    ids_sets - list of index dictionaries, one per index set
    vibstates_data - VibStatesData instance

    returns dict[ParameterSet, dict[ResonanceMotif, SpectralFeature]]
    """
    results = {}
    for idxset in ids_sets:
        ps = ParameterSet(idxset)
        results[ps] = {}
        for motif in motifs:
            try:
                resloc = solve_LSE_motif(motif, ps, vibstates_data)
                results[ps][motif] = SpectralFeature(resloc)
            except np.linalg.LinAlgError:
                pass

    return results

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
ContributionRow(term_id, term, params, location, coeff)    "why" layer
  │ 4. group by rounded location, add the coeffs
  v
SpectralFeature(location, rows)                            "what" layer
  │ 5. add peak width, filter to the spectral window
  v
grid.py → spectrum array                                   the picture

"""

@dataclass(frozen=True)
class ContributionRow:
    term_id: int
    term: 'CompiledTerm'
    params: ParameterSet
    location: ResLocPoint
    coeff: float

    @property
    def motif(self) -> 'ResonanceMotif':
        return self.term.cmp_resmotf


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

    def group_by(self, key: Callable[[ContributionRow], Hashable]) -> dict[Hashable, 'ContributionTable']:
        groups = defaultdict(list)
        for r in self._rows:
            groups[key(r)].append(r)
        return {k: ContributionTable(v) for k, v in groups.items()}

    # named shortcuts for frequent questions
    def by_params(self):
        return self.group_by(lambda r: r.params)

    def by_motif(self):
        return self.group_by(lambda r: r.motif)

    def by_location(self, decimals=6):
        return self.group_by(lambda r: tuple((ax, round(v, decimals)) for ax, v in r.location.coordinates))

    def axis_range(self, axis):
        vals = [r.location[axis] for r in self._rows if axis in r.location.axes]
        return min(vals), max(vals)


def build_contributions(terms, n_modes, molsys_data, vibstates_data,
                        polarization_linear_comb=None, precalculated_data=None) -> 'ContributionTable':
    locations: dict[tuple[ResonanceMotif, ParameterSet], ResLocPoint] = {}
    rows = []
    for term_id, term in enumerate(terms):
        for idxset in make_idx_sets(n_modes, sorted(term.idx_nonsumm)):
            ps = ParameterSet(idxset)
            key = (term.cmp_resmotf, ps)
            if key not in locations:         # one solve per (motif, params)
                locations[key] = solve_LSE_motif(term.cmp_resmotf, ps, vibstates_data)
            coeff, _ = evaluate_term_coeff_sumover(term, idxset, molsys_data,
                                                   polarization_linear_comb, precalculated_data)
            rows.append(ContributionRow(term_id, term, ps, locations[key], coeff))
    return ContributionTable(rows)



"""
coefficient should be attached to resonance with the same ParameterSet
"""

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

"""

