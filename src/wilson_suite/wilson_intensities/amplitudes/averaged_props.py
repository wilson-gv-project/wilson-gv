"""
PROPERTIES in VibPerturbedTerm ---- #TODO still
"""
from typing import Callable
import copy

import numpy as np
from wilson_suite.wilson_intensities.amplitudes.term_parts import PropsCollection
from wilson_suite.wilson_intensities.amplitudes.utils import generate_index_choices_general
from wilson_suite.wilson_main.abstractions import MolPropsCollection

import logging
logger = logging.getLogger("wilson")


def group_PropsColls_by_numerator(list_props_collections: list['PropsCollection']) -> dict[PropsCollection, list[PropsCollection]]:
    """
    [x] DONE
    returns groups of avrg props motifs by numerator
    """
    groups_here: dict['PropsCollection', list] = {}
    for props_collection in list_props_collections:
        # props_collection.identify_avrg_motif() returns props stripped from nm indices
        groups_here.setdefault(props_collection.identify_avrg_motif(), []).append(props_collection)
    return groups_here


def make_gen_func_to_compute_avrg(*,
                              avrg_expression: 'PropsCollection',
                              pulse_polarization_vector: list) -> Callable[[dict, 'MolPropsCollection'], float]:
    """
    for an expression with properties data values,
    compute average with given polarization setup for a choice of normal mode indices
    """
    num_pulses = len(avrg_expression.get_cart_axes())  # should this be a set?
    from .averaging import getGeneralPolarizationAveragingExpression

    polarization_linear_comb = getGeneralPolarizationAveragingExpression(rank = num_pulses,
                                                                        laser_pol = pulse_polarization_vector)

    def compute_for_idx_choice(index_choices: dict, props_data: 'MolPropsCollection') -> float:
        """
        index_choices: dict, props_data: 'MolPropsCollection'
        """
        # if not isinstance(props_data, MolPropsCollection):
        #     if isinstance(props_data, list):
        #         if isinstance(props_data[0], MolPropsCollection):
        #             props_data = MolPropsCollection(props_data)
        #         else:
        #             raise TypeError(f'props_data is not an instance of MolPropsCollection: {type(props_data)} - {props_data}')
        #     else:
        #         raise TypeError(f'props_data is not an instance of MolPropsCollection: {type(props_data)} - {props_data}')

        if not isinstance(props_data, MolPropsCollection):
            raise TypeError(
                f"props_data must be a MolPropsCollection, got {type(props_data).__name__}"
            )

        # Validate index_choices has all required keys
        required_inds = {i for prop in avrg_expression for i in prop.inds}
        missing = required_inds - index_choices.keys()
        if missing:
            raise KeyError(
                f"index_choices is missing required mode indices: {missing}"
            )

        # from ..utils.spectrum_utils import greek_list, num_Greek
        from wilson_suite.wilson_utils.prop_trivname import prop_trivname

        total = 0.

        for cart_axes in polarization_linear_comb:

            # Comment (MR): Noting that I considered if there would be any issues with this in generalized routine,
            # couldn't think of any but want to discuss and double check for safety
            # greek_dict = {L: n for L, n in zip(greek_list[:len(cart_axes)], cart_axes)}

            product = 1.

            for prop in avrg_expression:

                prop_tuple_key = prop_trivname(ord_el=len(prop.ops), ord_geo=prop.dord)

                nm_inds = tuple([index_choices[i] for i in prop.inds])
                # cart_inds = tuple([greek_dict[num_Greek[i.o]] for i in prop.ops])
                cart_inds = tuple([cart_axes[i.o] for i in prop.ops])
                # assert cart_inds == cart_inds1
                all_inds = (*nm_inds, *cart_inds)

                # retrieve data for preperty (prop_key) and idxs_key which is (tuple(mode inds), tuple(cart inds))
                product *= props_data.get(prop_tuple_key).vals[all_inds]

            if product != 0.:
                logger.debug(f"Avrg prop contribution for indices {index_choices} and cart axes {cart_axes} with coefficient {polarization_linear_comb[cart_axes]}: {product}")

            total += product * polarization_linear_comb[cart_axes]

        return total

    return compute_for_idx_choice


def calculate_avrg_tensor(avrg_expression: 'PropsCollection',
                          pulse_polarization_vector: list,
                          props_data: 'MolPropsCollection',
                          number_of_nmodes: int,
                          nm_inds_choices: list[int]):
    """
    Precalculating the full tensor for given avrg_expression

    nm_inds_choices - could be generated with for all normal modes with:
        nm_inds_choices: list[dict[str, int]] = generate_index_choices_general(indlabels_in_motif=mode_inds, labels=list(range(number_of_nmodes)))

    """
    # so indices are in alphabetical order in full_tensor below
    mode_inds = sorted(set(avrg_expression.get_mode_indices()))  # list, deterministic order
    ind_choices: list[dict[str, int]] = generate_index_choices_general(indlabels_in_motif=mode_inds, labels=nm_inds_choices)

    # Indicating generalized version for updating
    func_general = make_gen_func_to_compute_avrg(avrg_expression=avrg_expression, pulse_polarization_vector=pulse_polarization_vector)

    full_tensor = np.zeros((number_of_nmodes,)*len(mode_inds))

    for idx in ind_choices:
        # so indices are in alphabetical order
        full_tensor[tuple(idx[k] for k in mode_inds)] = func_general(idx, props_data)

    return full_tensor


def group_PropsColls_by_repetition_pattern(avrg_expressions: list[PropsCollection]):
    """
    [x] DONE
    For a list of averaged properties expressions already grouped by numerator motifs

    """
    max_nm_inds = 0
    all_encoded: dict[PropsCollection, list[PropsCollection]] = {}

    for prop_coll in avrg_expressions:
        # number of max of uniques nm indices is equal to number of boxes for derivatives 
        #           - all indices are different and all props are 1st order ders
        nm_indx = prop_coll.get_mode_indices()
        num_unique_nm_idx = len(set(nm_indx))

        # if max number of unique indices is found then 
        # that would be the model expression for the whole group with this numerator motif
        if num_unique_nm_idx == len(nm_indx):
            all_encoded[nm_indices_repetition_reduce_deriv_symmetry(prop_coll)] = []
            for pr_coll in avrg_expressions:
                all_encoded[nm_indices_repetition_reduce_deriv_symmetry(prop_coll)].append(pr_coll)
            return all_encoded
        
        max_nm_inds = max(num_unique_nm_idx, max_nm_inds)
        all_encoded.setdefault(nm_indices_repetition_reduce_deriv_symmetry(prop_coll), []).append(prop_coll)
    
    return all_encoded



def make_unique_avrg_tensors_mapping(avrg_expressions: list[PropsCollection]):
    """

    """
    numerator_groups = group_PropsColls_by_numerator(avrg_expressions)

    numer_upd = {k:group_PropsColls_by_repetition_pattern(v) for k,v in numerator_groups.items()}

    flat_dict = {}
    for num_group in numer_upd:
        for pattern in numer_upd[num_group]:
            new_inds = nm_indices_repetition_decoding(pattern)
            model_expr = reconstruct_unique_avrg_expression(numerator_group=num_group, nm_indices=new_inds)
            for expression in numer_upd[num_group][pattern]:
                flat_dict[expression] = model_expr
                
    return flat_dict

def nm_indices_repetition_encoding(nm_indices: list[str]):
    """
    [a, b, c, d, b] - (0, 2, 0, 0, 2)
    [a, a, b, c, d] - (1, 1, 0, 0, 0)
    [a, b, c, d, d] - (0, 0, 0, 4, 4)
    """
    counts_dict = {i:nm_indices.count(i) for i in nm_indices}
    repeated = {k:i+1 for i,k in enumerate(counts_dict.keys()) if counts_dict[k]>1}

    encoded = [0] * len(nm_indices)
    for i, ind in enumerate(nm_indices):
        encoded[i] = repeated.get(ind, 0)
    return tuple(encoded)

def nm_indices_repetition_decoding(encoded_idx: tuple[int]):
    """
    (0, 2, 0, 0, 2) - [a, b, c, d, b]
    (1, 1, 0, 0, 0) - [a, a, b, c, d]
    (0, 0, 0, 4, 4) - [a, b, c, d, d]
    """
    lat_letters_for_zeros = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    lat_letters_for_ones = copy.deepcopy(lat_letters_for_zeros)
    
    result = []

    for coded in encoded_idx:
        curr_letter = 0
        if coded == 0:
            result.append(lat_letters_for_zeros.pop(curr_letter))

        else:
            result.append(lat_letters_for_ones[coded-1])
            if lat_letters_for_ones[coded-1] in lat_letters_for_zeros:
                lat_letters_for_zeros.remove(lat_letters_for_ones[coded-1])
    return result

def nm_indices_repetition_reduce_deriv_symmetry(props: PropsCollection) -> tuple[int]:
    """
    returns a sorted encoding, unlike nm_indices_repetition_encoding
    """
    nm_group_template = props.get_mode_indices_group_template()
    nm_indices_encoded = nm_indices_repetition_encoding(props.get_mode_indices())
    grouped_coded = group_nm_indices(nm_indices_encoded, nm_group_template)

    for g in grouped_coded:
        g.sort()
    return tuple([el for group in grouped_coded for el in group])

def group_nm_indices(nm_indices, grouping_template) -> list[list]:
    """
    nm_indices - coded or not list of nm indices
    
    ['d', 'd', 'a', 'c', 'b'], [2, 1, 1, 1] --> [['d', 'd'], ['a'], ['c'], ['b']]
    """
    result = []
    curr = 0
    
    for gr in grouping_template:
        result.append(list(nm_indices[curr: curr+gr]))
        curr += gr
    return result

def reconstruct_unique_avrg_expression(numerator_group: 'PropsCollection',
                                       nm_indices: list[str]) -> 'PropsCollection':
    """
    dipNone[0] * dipNone[1] * dipNone[2] * dipNone[3] * dipNone[4]
    hypNone[0, 2, 4] * dipNone[1] * dipNone[3] * dipNone[5]

    """
    upd_props = []
    index_tracker = 0
    
    for prop in numerator_group:
        prop = copy.deepcopy(prop)
        prop.inds = nm_indices[index_tracker: index_tracker + prop.dord]
        index_tracker += prop.dord
        upd_props.append(prop)
    return PropsCollection(props=upd_props)

def identify_unique_avrg_tensors(avrg_expressions: list[PropsCollection]) -> list[PropsCollection]:
    """
        
    """
    return set(make_unique_avrg_tensors_mapping(avrg_expressions).values())


def get_ind_tuple_from_base(expr: PropsCollection, base_expr: PropsCollection, index_dict: dict):
    """Map expr to indices according to base expression's unique symbols."""
    base_unique = sorted(list(set(base_expr.get_mode_indices())))
    expr_inds = expr.get_mode_indices()

    if len(base_unique) < len(expr_inds):
        # walk through only base_unique ind labels
        return tuple(index_dict[sym] for sym in base_unique)
    elif len(base_unique) == len(expr_inds):
        # walk through all expr_inds labels, there are repeated labels
        return tuple(index_dict[sym] for sym in expr_inds)
    else:
        # this generally should not be possible
        raise ValueError('This base_expr cannot be a base expression for this expr')
