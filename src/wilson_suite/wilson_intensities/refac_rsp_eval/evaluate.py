"""
The numbers: term coefficients, resonance locations, and the spectrum on a grid.

  in:  CompiledTerm list (plan.py), MolSystemData, the orientation-average recipe, a states choice
  out: rows and zero pairs (build_contributions); the complex amplitude on a grid (all_features_on_grid,
       boxed_features_on_grid)

  holds: the four factors of one index set (evaluate_full_index_dict), the sum over the summation labels
         (evaluate_term_coeff_sumover), the averaged part as one einsum per term (AvrgTensor), the resonance
         location (solve_LSE_motif), the resonance factors on the grid, the features on the grid
"""

import string
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionRow,
    ContributionTable,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import box_slices
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResLocPoint,
    generate_LHS_motif,
)
from wilson_suite.wilson_system.system_data import (
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    VibDiff,
)
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
        SpectralFeature,
        SpectralWindow,
    )
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
        CompiledTerm,
        FreqTermsCollection,
        PropsCollection,
        ResCondKey,
        ResonanceMotif,
    )
    from wilson_suite.wilson_system.system_data import VibStatesData


"""
The data then passes through these steps; the property name (PropsCollection.names) is the key all the way:

1. build the request      "please fetch 'polgrad' from CFOUR"      MolSystemData.load, names from CompiledTerm.prop_names
2. obtainer returns       data_dict = {'polgrad': array, ...}
3. fill mol_props         put data_dict['polgrad'] into the 'polgrad' slot      MolSystemData.from_datadict
4. evaluate               read mol_props['polgrad'].vals[mode, i, j]    calculate_avrg_tensor_einsum (averaged),
                                                                        eval_non_avrg_per_indexdict (non-averaged)
"""

## ----------------------------------------------------
##          coefficient per index set/ParameterSet
## ----------------------------------------------------

# EVALUATION OF A SINGLE TERM - ONE INDEX SET: SUM OVER SET
def evaluate_term_coeff_sumover(compl_term: 'CompiledTerm',
                         idx_dict: dict,
                         molsys_data: 'MolSystemData',
                         polarization_linear_comb: dict | None = None,
                         zero_tol: float = 1e-18,
                         avrg_tensor: 'AvrgTensor | None' = None,
                         keep_leaves: bool = True,
                         *,
                         states_choice: str,
                         modes: Sequence[int] | None = None) -> tuple[float, dict | None]:
    """
    Coefficient of one term at one resonance location.
    states_choice: which state set of molsys_data gives the energy differences, e.g. 'anharmonic';
        no default (see MolSystemData.states).
    modes: the values every summed label runs over, e.g. (0, 1, 2, 3, 4) with mode 5 excluded (included_modes).
        None -> every mode, 0 .. n_modes-1. The caller fixes the resonance labels, from the same modes.

    idx_dict must fix every resonance label (compl_term.idx_nonsumm): these labels decide where
    the peak sits. Every label left out is summed over all modes (or over `modes`). The summed pieces all
    sit at the same location, so adding them is safe.

    e.g. a term with resonance labels (a, b) and summation label c:
        {'a': 0, 'b': 1}  ->  f(0,1,0) + f(0,1,1) + ... + f(0,1,n_modes-1)
        {'a': 0}          ->  ValueError: summing over b would add peaks at different positions

    avrg_tensor: the term's averaged part for every index set (calculate_avrg_tensor_einsum).
        None -> made here from polarization_linear_comb, which is then required.
        build_contributions makes it once per term, so it is not made again for every resonance index set.

    returns (total, leaves): leaves maps each full index set to its factors, e.g. for a = 0, b = 1, 2 modes:
        {ParameterSet({'a': 0, 'b': 1, 'c': 0}): {'NON_AVRG': ..., 'AVRG': ..., 'VIBDIFF_TERMS': ..., 'VIBENE_DENOM': ...},
         ParameterSet({'a': 0, 'b': 1, 'c': 1}): {...}}
    keep_leaves False -> leaves is None. build_contributions does not keep them: every row's leaves would take
        357 MB for ACDM (42 modes). FeatureResult.leaves computes one row's leaves again when they are asked for.
    """
    not_fixed = [i for i in compl_term.idx_nonsumm if i not in idx_dict]
    if not_fixed:
        raise ValueError(f'resonance labels {not_fixed} must be fixed, not summed')
    molsys_data.check_states_choice(states_choice)

    term_idx_all = sorted(compl_term.idx_summ + compl_term.idx_nonsumm)
    
    if molsys_data.eigenvals is not None:
        n_modes = len(molsys_data.eigenvals)
    else: raise ValueError('molsys_data.eigenvals is absent - number of modes is required')
    summed_values = range(n_modes) if modes is None else modes

    if avrg_tensor is None:
        if polarization_linear_comb is None:
            raise ValueError('polarization_linear_comb is required when no avrg_tensor is given')
        avrg_tensor = calculate_avrg_tensor_einsum(compl_term.avrg_props, molsys_data.mol_props,
                                                   polarization_linear_comb)

    def sum_over(index_dict: dict, remaining: list, leaves: dict | None) -> float:
        if not remaining:
            value, contribs = evaluate_full_index_dict(compl_term, index_dict,
                                                       molsys_data,
                                                       avrg_tensor,
                                                       zero_tol,
                                                       states_choice=states_choice)
            if leaves is not None:
                leaves[ParameterSet(index_dict)] = contribs
            return value
        current, rest = remaining[0], remaining[1:]
        return sum(sum_over({**index_dict, current: v}, rest, leaves) for v in summed_values)
    
    missing = [i for i in term_idx_all if i not in idx_dict]
    leaves = {} if keep_leaves else None
    return sum_over(idx_dict, missing, leaves), leaves


# EVALUATION OF A SINGLE TERM - ONE INDEX SET: FULL INDEX SET
def evaluate_full_index_dict(compl_term: 'CompiledTerm', index_dict: dict,
                               molsys_data: 'MolSystemData',
                               avrg_tensor: 'AvrgTensor',
                               zero_tol: float = 1e-18,
                               *,
                               states_choice: str) -> tuple[float, dict]:
    """
    Evaluate the term for a single index dictionary.
    avrg_tensor: the term's averaged part for every index set (calculate_avrg_tensor_einsum)
    states_choice: the state set for the energy differences, molsys_data.states[states_choice]
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
    AVRG = eval_avrg_per_indexdict(avrg_tensor, index_dict, zero_tol)

    if AVRG == 0.0:
        return 0.0, {'AVRG': AVRG}

    # Evaluate NON_AVRG
    NON_AVRG = eval_non_avrg_per_indexdict(compl_term.non_avrg_props, index_dict, molsys_data, zero_tol)
    if NON_AVRG == 0.0:
        return 0.0, {'NON_AVRG': NON_AVRG, 'AVRG': AVRG}

    # Evaluate VIBDIFF_TERMS (split once per term, not per index set: todo.txt F3)
    extra_freqterms = compl_term.freq_pert_wf_diff

    VIBDIFF_TERMS = otf_vibdiffdenom(extra_freqterms, index_dict, molsys_data, states_choice=states_choice)

    # Evaluate VIBENE_DENOM
    VIBENE_DENOM = harmonic_denom(compl_term.freq_vibenedenom, index_dict, molsys_data)

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
    
    # names: made once per collection (PropsCollection.names), not once per index set
    for non_avrg_prop, triv_name in zip(non_avrg_expr, non_avrg_expr.names):
        # accessing values for non-averaged properties from data
        if non_avrg_prop.inds is None or len(non_avrg_prop.inds) == 0:
            # If there are no indices, we assume it's a scalar property
            na_prop_inds = ()
        else:
            na_prop_inds = tuple([index_dict[i] for i in non_avrg_prop.inds])

        prop: MolecularProperty = molsys_data.mol_props[triv_name]
        NON_AVRG = prop.vals[na_prop_inds]

        # = np.isclose(NON_AVRG, 0., atol=zero_tol, rtol=0.) for one number, without its cost per call (F3)
        if abs(NON_AVRG) <= zero_tol:
            return 0.
        else:
            product_all *= NON_AVRG
    
    return product_all


# EVALUATION OF TERM PARTS FOR ONE INDEX SET
def eval_avrg_per_indexdict(avrg_tensor: 'AvrgTensor', index_dict: dict, zero_tol: float = 1e-18):
    """
    The averaged part at one index set, read from the term's tensor (calculate_avrg_tensor_einsum).
    |value| <= zero_tol counts as 0.
    ---
    Averaging papers
    https://pubs.aip.org/aip/jcp/article/67/11/5026/788630/On-three-dimensional-rotational-averages
    https://pubs.aip.org/aip/jcp/article/141/20/204103/193500/Rotational-averaging-of-multiphoton-absorption
    """
    result = avrg_tensor.at(index_dict)

    # = np.isclose(result, 0., atol=zero_tol, rtol=0.) for one number, and faster (F3)
    return 0. if abs(result) <= zero_tol else result


def _make_func_to_compute_avrg(*,
                              avrg_expression: 'PropsCollection',
                              polarization_linear_comb: dict
                              ) -> Callable[[dict, 'MolPropsCollection'], float]:
    """
    for an expression with properties data values,
    compute average with given polarization setup for a choice of normal mode indices

    The pipeline uses calculate_avrg_tensor_einsum (every index set at once). This loop stays as its
    reference in the tests.
    """
    props_and_names = tuple(zip(avrg_expression, avrg_expression.names))     # names once, not per call

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

        total = 0.

        for cart_axes in polarization_linear_comb:

            # Comment (MR): Noting that I considered if there would be any issues with this in generalized routine,
            # couldn't think of any but want to discuss and double check for safety

            product = 1.

            for prop, prop_tuple_key in props_and_names:

                nm_inds = tuple([index_choices[i] for i in prop.inds]) # type: ignore
                cart_inds = tuple([cart_axes[i.o] for i in prop.ops])
                all_inds = (*nm_inds, *cart_inds)

                # retrieve data for property (prop_key) and idxs_key which is (tuple(mode inds), tuple(cart inds))
                product *= props_data[prop_tuple_key].vals[all_inds]

            total += product * polarization_linear_comb[cart_axes]

        return total

    return compute_for_idx_choice


# EVALUATION OF THE AVRG PART FOR EVERY INDEX SET AT ONCE - one einsum (todo.txt F1)
AVRG_SLOT_LETTERS = 'IJKLMN'    # einsum letter of cartesian slot 0, 1, 2, ... (ranks 2 to 6); slot: see calculate_avrg_tensor_einsum


@dataclass(frozen=True)
class AvrgTensor:
    """
    The averaged part of one term, for every choice of its mode labels (calculate_avrg_tensor_einsum).
        labels  the distinct mode labels of the averaged part, alphabetical, e.g. ('a', 'b', 'c')
        values  one axis per label, in the order of `labels`: values[0, 1, 2] is the value for a=0, b=1, c=2
    The labels travel with the values, so a lookup cannot read the axes in another order (the B6 trap).
    """
    labels: tuple[str, ...]
    values: np.ndarray

    def at(self, index_dict: dict) -> float:
        """
        The value for one index set. Labels that the averaged part does not have are ignored.
            e.g. labels ('a', 'b'), {'a': 3, 'b': 1, 'c': 0} -> values[3, 1]
        """
        return self.values[tuple(index_dict[label] for label in self.labels)] # type: ignore


def make_avrg_einsum_subscripts(avrg_expression: 'PropsCollection') -> str:
    """
    The averaged part as one einsum: one operand per property, then the recipe W, then the output.

    For one index set, the on-the-fly loop (_make_func_to_compute_avrg) computes
        sum over recipe keys (i_0, i_1, ...) of  W[i_0, i_1, ...] * product over properties of prop[modes..., carts...]
    Each property reads its mode indices first, then one Cartesian index per Cartesian axis (entry of prop.ops):
    the key's index at that axis's slot (op.o). einsum writes the same sum with one letter per index:
        mode label  ->  itself ('a', 'b', ...)                not summed: one axis of the output
        slot k      ->  AVRG_SLOT_LETTERS[k] ('I', 'J', ...)  summed: in W and in the property that has slot k

    e.g. term 8, polgrad[b] (slots 0, 3) * dipgrad[a] (slot 1) * dipgrad[c] (slot 2):
        polgrad[b, I, L]     'bIL'
        dipgrad[a, J]        'aJ'
        dipgrad[c, K]        'cK'
        W[I, J, K, L]        'IJKL'   W[i, j, k, l] = the recipe coefficient of key (i, j, k, l)
        T[a, b, c]           'abc'    the distinct mode labels, alphabetical
        -> 'bIL,aJ,cK,IJKL->abc':  T[a, b, c] = sum over I, J, K, L of polgrad[b, I, L] dipgrad[a, J] dipgrad[c, K] W[I, J, K, L]
    A label in two properties (term 10: dipgrad[a] twice) multiplies two values of the same mode. A label twice
    in one property (polhess[a, a]) reads its diagonal. Both are einsum's own rules, and both match the loop.

    raises ValueError if a mode label is not one lowercase letter (slots use upper case), or if the slots are
    not 0 .. rank-1, each once
    """
    slots = sorted(op.o for prop in avrg_expression for op in prop.ops)
    if slots != list(range(len(slots))):
        raise ValueError(f'averaged slots must be 0 .. {len(slots) - 1}, each once; got {slots}')
    if len(slots) > len(AVRG_SLOT_LETTERS):
        raise ValueError(f'at most {len(AVRG_SLOT_LETTERS)} averaged slots, got {len(slots)}')

    operands, labels = [], set()
    for prop in avrg_expression:
        mode_labels = list(prop.inds or ())
        not_a_letter = [lbl for lbl in mode_labels if len(lbl) != 1 or lbl not in string.ascii_lowercase]
        if not_a_letter:
            raise ValueError(f'mode labels must be one lowercase letter each, got {not_a_letter}')
        labels.update(mode_labels)
        operands.append(''.join(mode_labels) + ''.join(AVRG_SLOT_LETTERS[op.o] for op in prop.ops))

    return ','.join([*operands, AVRG_SLOT_LETTERS[:len(slots)]]) + '->' + ''.join(sorted(labels))


def calculate_avrg_tensor_einsum(avrg_expression: 'PropsCollection',
                                 props_data: 'MolPropsCollection',
                                 polarization_linear_comb: dict) -> AvrgTensor:
    """
    The averaged part for every index set at once: the values that the on-the-fly loop
    (_make_func_to_compute_avrg) gives one index set at a time, from one einsum.

    Slot: one interaction of the molecule with a light field, numbered by the field:
        slot 0 = the detected field,  slot k = the pulse with id k (pulse ids are 1, 2, ...; ElectricField checks)
    Each property has one Cartesian axis (x, y, z, in the molecule's axes) per field it couples to. In the code,
    prop.ops has one entry per Cartesian axis (derive stores it as a QOperator), and op.o is that axis's slot.
    derive writes op.o: interaction j gets the id of its pulse (dressWithPulseInteractions, response_terms.py).
        e.g. term 8, in the order of the expression:
            polgrad[b]  2 Cartesian axes, slots 0, 3   the detected field and pulse 3
            dipgrad[a]  1 Cartesian axis,  slot 1      pulse 1
            dipgrad[c]  1 Cartesian axis,  slot 2      pulse 2
    The order of the properties in the expression does not matter: op.o, not the position of the property, ties
    each Cartesian axis to its field. Term 8 with the properties reversed: 'cK,aJ,bIL,IJKL->abc', the same tensor.
    The order of the Cartesian axes inside one property does matter: it sets which data axis goes with which slot.
        e.g. polgrad[b] with ops (0, 3): in polgrad[b, i, j], i is the direction at slot 0, j at slot 3.
    For polgrad and polhess both orders give the same value: they are symmetric in i, j (formaldehyde data:
    exactly). dipgrad and diphess have one Cartesian axis, so no order.

    rank = the number of slots = the number of Cartesian axes of all properties together (term 8: 2 + 1 + 1 = 4).
    A recipe key has one direction per slot, in slot order:
        key (i_0, i_1, i_2, i_3) = (direction at slot 0, at slot 1, at slot 2, at slot 3), 0 = x, 1 = y, 2 = z
    The recipe comes from the lab side, with the same numbering: the experiment lists its polarizations as
    (detector, pulse 1, pulse 2, ...) (all_polarizations, experiment_abstractions.py), and laser_pol and the
    recipe are made from that list. So key position k and op.o = k both mean "the field of slot k".
        Not yet for ranks 3, 5 and 6: get_pol_tensor reverses the slots (todo.txt A15). Ranks 2 and 4 (EVV) are
        not affected.
        A pulse that acts twice would give two Cartesian axes the same op.o: make_avrg_einsum_subscripts raises.
    The molecule turns at random, so the average weighs every combination of directions with the recipe
    coefficient.

        1. subscripts from the expression (make_avrg_einsum_subscripts), e.g. term 8: 'bIL,aJ,cK,IJKL->abc'
        2. W: the recipe {(i_0, i_1, ...): coefficient} as a dense 3 x 3 x ... array, one axis per slot;
           0 where the recipe has no entry
        3. one operand per property: its data array, found by its name, as the loop finds it (PropsCollection.names)
        4. np.einsum: one axis per distinct mode label, alphabetical
    Size: n_modes ** (number of distinct labels), e.g. 42 modes, 3 labels: 74088 values, 0.6 MB.

    raises ValueError if a recipe key does not have one entry per slot
    """
    # 1. The einsum string, from the symbolic expression alone (no data yet).
    #    term 8: polgrad[b] (slots 0, 3) * dipgrad[a] (slot 1) * dipgrad[c] (slot 2)  ->  'bIL,aJ,cK,IJKL->abc'
    subscripts = make_avrg_einsum_subscripts(avrg_expression)

    # rank: the number of slots = the number of Cartesian axes of all properties together (entries of prop.ops).
    #    term 8: polgrad has 2 Cartesian axes, each dipgrad has 1  ->  rank 4.
    #    make_avrg_einsum_subscripts has checked that the slots are 0 .. rank-1, each once: slot k is axis k of W.
    rank = sum(len(prop.ops) for prop in avrg_expression)

    # 2. The recipe as a dense array W: one axis of length 3 (x, y, z) per slot.
    #    The recipe holds only the nonzero keys: rank 4 has 21 of the 3**4 = 81 keys.
    #    e.g. laser_pol (1, 1, 1): key (0, 0, 1, 1) has 1/15, so W[x, x, y, y] = 1/15.
    #    A key that is not in the recipe stays 0 in W, so einsum adds nothing for it (the loop never visits it).
    #    A key of the wrong length would not raise in numpy: W[(0, 0)] = c fills the whole block W[0, 0, :, :].
    #    So the length is checked here.
    W = np.zeros((3,) * rank)
    for key, coeff in polarization_linear_comb.items():
        if len(key) != rank:
            raise ValueError(f'recipe key {key} has {len(key)} entries, the averaged part has {rank} slots')
        W[key] = coeff

    # 3. One data array per property, in the order of the expression: the order of the operands in the string.
    #    The name comes from the property's orders (PropsCollection.names, as in the loop):
    #        1 Cartesian axis, 1 mode -> 'dipgrad' (n, 3)        2 Cartesian axes, 1 mode -> 'polgrad' (n, 3, 3)
    #        1 Cartesian axis, 2 modes -> 'diphess' (n, n, 3)    2 Cartesian axes, 2 modes -> 'polhess' (n, n, 3, 3)
    #    n = the number of modes. Mode axes first, then x, y, z: the layout the string assumes.
    #    term 8: [polgrad, dipgrad, dipgrad]. The same dipgrad array comes twice: once read at a, once at c.
    operands = [np.asarray(props_data[name].vals) for name in avrg_expression.names]

    # The output axes: the letters after '->', e.g. 'abc' -> ('a', 'b', 'c').
    # Read from the string, so the labels are the axis order that einsum really uses.
    labels = tuple(subscripts.split('->')[1])

    # 4. One call gives every index set. numpy multiplies the operands and sums over each letter that is not
    #    in the output: the slot letters I, J, K, L. The mode letters a, b, c are in the output: not summed.
    #    term 8, shape (n, n, n):
    #        values[a, b, c] = sum over I, J, K, L of  polgrad[b, I, L] * dipgrad[a, J] * dipgrad[c, K] * W[I, J, K, L]
    #    For one (a, b, c), this is the sum the loop computes: one product per recipe key, times its coefficient.
    #    optimize=True: numpy multiplies two arrays at a time, in a cheap order, instead of one loop over all
    #    7 letters. Only the order of the additions changes, so the values match the loop up to rounding
    #    (8e-14 relative on ACDM, todo.txt F1).
    return AvrgTensor(labels, np.einsum(subscripts, *operands, W, optimize=True))


# EVALUATION OF TERM PARTS FOR ONE INDEX SET - on the fly
def otf_vibdiffdenom(freqterms: 'FreqTermsCollection', index_dict: dict, molsys_data: 'MolSystemData', *,
                     states_choice: str):
    """
    product of 1/(E_left - E_right), in Eh
    energies from the state set molsys_data.states[states_choice]
    """
    product_all = 1.
    states = molsys_data.states[states_choice]

    for vibdiff in freqterms:
        vib_diff_w_value = VibDiff.from_symbolic(vibdiff, index_dict, states)

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
    A motif without conditions (a term without resonance conditions, a flat background, todo.txt E12) has no axes:
    its location is ResLocPoint({}), and the feature is the same on every grid point.

    raises np.linalg.LinAlgError if the location is not a point (fewer independent conditions
    than axes, e.g. a single condition on A + B) or the conditions are inconsistent.
    Compiled terms always pass (CompiledTerm.from_VibPertTerm checks the motif, B7); these
    checks guard motifs built by hand.
    """
    if len(motif) == 0:
        return ResLocPoint({})

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

def included_modes(molsys_data: MolSystemData, exclude_modes: Iterable[int] = ()) -> tuple[int, ...]:
    """
    The modes the evaluation uses: 0 .. n_modes-1 without exclude_modes, n_modes = len(molsys_data.eigenvals).
        e.g. 6 modes, exclude_modes (5,) -> (0, 1, 2, 3, 4)
    An excluded mode is removed everywhere (todo.txt Decisions, 2026-10-13): it is no resonance label (no row,
    no peak) and no value of a summed label, as if the molecule did not have it. So the other peaks change too:
    their sums lose the excluded mode. (VibAnaSetup.exclude_modes in the old code: no peaks either, but the sums
    dropped the mode only where the summed label is in the averaged part.)

    raises ValueError for a mode that does not exist (modes count from 0), or if no mode is left
    """
    if molsys_data.eigenvals is None:
        raise ValueError('molsys_data.eigenvals is absent - number of modes is required')
    n_modes = len(molsys_data.eigenvals)
    excluded = set(exclude_modes)
    unknown = [m for m in excluded if m not in range(n_modes)]
    if unknown:
        raise ValueError(f'exclude_modes {unknown}: no such mode, the modes are 0 .. {n_modes - 1}')
    modes = tuple(m for m in range(n_modes) if m not in excluded)
    if not modes:
        raise ValueError(f'exclude_modes removes all {n_modes} modes')
    return modes


def make_idx_sets(modes: Sequence[int], mode_labels: Sequence) -> list[dict[str, int]]:
    """
    Every assignment of the given modes to mode_labels, e.g. modes (0, 1), labels ['a', 'b']:
        [{'a': 0, 'b': 0}, {'a': 0, 'b': 1}, {'a': 1, 'b': 0}, {'a': 1, 'b': 1}]
    """
    import itertools
    return [dict(zip(mode_labels, combo)) for combo in itertools.product(modes, repeat=len(mode_labels))]


def build_contributions(terms: Sequence['CompiledTerm'], molsys_data: MolSystemData,
                        polarization_linear_comb: dict, *, states_choice: str,
                        exclude_modes: Iterable[int] = ()
                        ) -> tuple[ContributionTable, list[tuple[int, ParameterSet]]]:
    """
    One row per (term, index set) that adds to the spectrum. Index sets fix the term's
    resonance labels; evaluate_term_coeff_sumover sums the rest. Locations are in cm-1.
    The number of modes comes from molsys_data.eigenvals, the state energies from
    molsys_data.states[states_choice], e.g. states_choice='anharmonic' (no default, see MolSystemData.states).
    polarization_linear_comb: the orientation-average recipe (pipeline.make_polarization_linear_comb).
    exclude_modes: modes removed everywhere, e.g. (5,): no index set and no summed value uses them (included_modes).
    Checks molsys_data first: the frequencies, the chosen states, and values for every property the terms use
    (CompiledTerm.prop_names), however molsys_data was made (MolSystemData.load or from_datadict).

    returns (rows, zero), so that len(rows) + len(zero) == number of pairs:
        zero   - (term_id, params) with coeff == 0: no row, no location solved

    Every nonzero pair gets a location: CompiledTerm.from_VibPertTerm checks that the motif fixes
    one point (B7). A term built by hand skips that check; if its motif has no single point,
    solve_LSE_motif raises and the run stops.
    A term without resonance conditions (a flat background, todo.txt E12) has no label to fix: one index set, {},
    whose coefficient sums every label, at the location without axes (ResLocPoint({})).
    """
    modes = included_modes(molsys_data, exclude_modes)
    molsys_data.check_states_choice(states_choice)
    molsys_data.check_props(name for term in terms for name in term.prop_names)
    states = molsys_data.states[states_choice]

    locations: dict[tuple[ResonanceMotif, ParameterSet], ResLocPoint] = {}
    rows, zero = [], []
    for term_id, term in enumerate(terms):
        motif = term.cmp_resmotf
        # the averaged part for every index set: one einsum per term, not one per index set
        avrg_tensor = calculate_avrg_tensor_einsum(term.avrg_props, molsys_data.mol_props, polarization_linear_comb)
        for idxset in make_idx_sets(modes, sorted(term.idx_nonsumm)):
            ps = ParameterSet(idxset)

            # no leaves: FeatureResult.leaves computes a row's leaves again when asked
            coeff, _ = evaluate_term_coeff_sumover(term, idxset, molsys_data, polarization_linear_comb,
                                                   avrg_tensor=avrg_tensor, keep_leaves=False,
                                                   states_choice=states_choice, modes=modes)
            # not saving 0 coeffs
            if coeff == 0.:
                zero.append((term_id, ps))
                continue
            key = (motif, ps)

            # one solve per (motif, params)
            if key not in locations:
                locations[key] = solve_LSE_motif(motif, ps, states, unit='cm-1')
            rows.append(ContributionRow(term_id, motif, ps, locations[key], coeff))
    return ContributionTable(rows), zero


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
        (deleted in E9; d93cc7b is the last commit with it).
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
    propagator_param - vd -> the propagator's parameter in Eh, one call per condition. The Eh form of
                       feature.propagator_param (cm-1, a number or a (vd, freqs) function), which
                       eval_feature_on_grid converts to this.
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
                         propagator: Callable=lorentzian_propagator) -> complex | np.ndarray:
    """
    amplitude_coeff * resonance factor of the feature's motif. grid in cm-1, result in au.

    The propagator's parameter is feature.propagator_param, cm-1 (converted to Eh here):
        a number   -> the same on every condition
        a function -> propagator_param(vd, grid_cm) for each condition (features.PropagatorParam):
                      a number per condition, or an array of the grid's shape for one that changes with
                      the frequency, e.g. lambda vd, freqs: 2. + 0.002 * np.abs(freqs['A'])
    The box and get_intensity take the same propagator_param, at the peak (SpectralFeature.peak_widths).
    A flat feature (no conditions, todo.txt E12): amplitude_coeff on every grid point, an array of the grid's shape.
    """
    param = feature.propagator_param
    if param is None:
        raise ValueError(f'feature {feature} has no propagator_param')
    if callable(param):
        param_au = lambda vd: convNu2Ene(param(vd, grid_cm)) # type: ignore
    else:
        gamma_au = convNu2Ene(param)
        param_au = lambda _vd: gamma_au

    if feature.motif is None:
        raise ValueError(f'feature {feature} has no motif')
    
    if feature.amplitude_coeff is None:
        raise ValueError(f'feature {feature} has no amplitude_coeff')

    if feature.is_flat:
        shape = np.broadcast_shapes(*(np.shape(v) for v in grid_cm.values()))
        return np.full(shape, complex(feature.amplitude_coeff))

    grid_au = {ax: convNu2Ene(v) for ax, v in grid_cm.items()}

    resonance_factor = eval_resonance_factor(feature.motif, feature.rows[0].params, vibstates_data,
                                             grid_au, propagator, propagator_param=param_au)

    return feature.amplitude_coeff * resonance_factor


# every feature on every grid point: no boxes, no domains
def all_features_on_grid(features: Iterable['SpectralFeature'],
             vibstates_data: 'VibStatesData',
             coords_cm: dict[str, np.ndarray]) -> np.ndarray:
    """
    Sum of eval_feature_on_grid over all features, each on the whole grid. Nothing is cut off, so this is
    the reference for the evaluation box by box (boxed_features_on_grid). Slow for many features on big grids.

    coords_cm - one 1D array per axis, cm-1, e.g. the first output of Box.make_grid.
    Returns the complex amplitude in au; array dimension i is the i-th axis of coords_cm (ij indexing),
    e.g. coords_cm A with 5 points, B with 3 -> shape (5, 3). intensity = abs(amplitude)**2
    """
    mesh = dict(zip(coords_cm, np.meshgrid(*coords_cm.values(), indexing='ij')))
    total = np.zeros(tuple(len(points) for points in coords_cm.values()), dtype=complex)
    for feature in features:
        total += eval_feature_on_grid(feature, vibstates_data, mesh)
    return total


# the window's features, each on its own box
def boxed_features_on_grid(window: 'SpectralWindow',
                vibstates_data: 'VibStatesData',
                coords_cm: dict[str, np.ndarray]) -> np.ndarray:
    """
    Every feature of the window (full_features and contrib_features) on its own box only: the part of the grid
    under its box (box_slices). Each box is a cut-out of the same grid points, so where boxes overlap their
    amplitudes add, as in all_features_on_grid; squaring comes later, on the whole grid. Only each feature's tail
    beyond its box is left out.

    Size of the cut: set by the box sizes. With amplitude_half_widths (features.py) the intensity error stays near
    eps * the plot's floor (measured against all_features_on_grid: tests/e10_boxes.py, todo.txt E10). To check a case,
    compare with all_features_on_grid.
    Until 2026-10-14 features were evaluated by domains (boxes that overlap, each feature on the whole domain box):
    that hid part of the cut of the old boxes, at up to 20 times the cost. With amplitude boxes the own box is enough.

    window    - e.g. from SpectralWindow.from_features, so every feature has its box (a flat feature: the window)
    coords_cm - one 1D array per window axis, cm-1, e.g. the first output of window.sample_grid
    Returns the complex amplitude in au, laid out as in all_features_on_grid. A box outside the grid adds nothing.
    """
    total = np.zeros(tuple(len(points) for points in coords_cm.values()), dtype=complex)
    for feature in window.full_features + window.contrib_features:
        if feature.feat_box is None:
            raise ValueError(f'feature {feature} has no box: SpectralWindow.from_features gives one')
        slices = box_slices(feature.feat_box, coords_cm)
        if slices is None:
            continue
        box_coords = {ax: points[s] for (ax, points), s in zip(coords_cm.items(), slices)}
        total[slices] += all_features_on_grid([feature], vibstates_data, box_coords)
    return total

