"""
PROPERTIES in VibPerturbedTerm ---- #TODO still
"""
from typing import Callable
import copy

import numpy as np
from wilson_suite.wilson_derive.abstractions import PolProp
from wilson_suite.wilson_intensities.amplitudes.term_parts import PropsCollection, VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.utils import generate_index_choices_general
from wilson_suite.wilson_main.abstractions import MolPropsCollection

import logging
logger = logging.getLogger("wilson")

def simple_prop_ID(property: 'PolProp') -> tuple[tuple, int]:
    """
    !USING TUPLES OF TUPLES
    """
    operators = tuple([op.o for op in property.ops])
    return (operators, property.dord)


def make_avrg_props_motif(props: list['PolProp']) -> set[tuple]:
    """
    !USING TUPLES OF TUPLES

    indices below are concrete, after '|' but could be others, main part of ID is in the numerator
    {((0, 3), 1),  ---- \\frac{\\partial\\alpha_{\\alpha\\delta}} | e.g. {\\partial Q_{b}}
     ((2,), 1),    ---- \\frac{\\partial\\mu_{\\gamma}} | e.g. {\\partial Q_{b}}
     ((1,), 1)}    ---- \\frac{\\partial\\mu_{\\beta}} | e.g. {\\partial Q_{a}}
    """
    num_unique_inds = len(set([ind for prop in props for ind in prop.inds if prop.ops]))
    return tuple(simple_prop_ID(prop) for prop in props if prop.ops) + (num_unique_inds,)


def identify_unique_avrgmotifs(list_of_terms: list['VibPerturbedTerm']) -> set[PropsCollection]:
    """
    motif contains props and total number of unique indices in them together
    ??? --- not usefull now?
    """
    return set(PropsCollection(term.props).identify_avrg_motif() for term in list_of_terms)

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

def make_func_to_compute_avrg(*,
                     avrg_expression: 'PropsCollection',
                     polarization: str = 'ZZZZ') -> Callable[[dict, 'MolPropsCollection'], float]:
    """
    [x] DONE
    for an expression with properties data values, 
    compute average with given polarization setup for a choice of normal mode indices
    """
    num_pulses = len(avrg_expression.get_cart_axes()) # should this be a set?
    from .averaging import getPolarizationAveragingExpression

    # polarization='ZZZZ' - only this one is possible now
    polarization_avrg_terms, prefactor = getPolarizationAveragingExpression(num_pulses=num_pulses, polarization=polarization)

    def compute_for_idx_choice(index_choices: dict, props_data: 'MolPropsCollection') -> float:
        from ..utils.spectrum_utils import greek_list, num_Greek
        from wilson_suite.wilson_utils.prop_trivname import prop_trivname

        total = 0.

        for cart_axes in polarization_avrg_terms:
            greek_dict = {L: n for L, n in zip(greek_list[:len(cart_axes)], cart_axes)}
            product = 1.

            for prop in avrg_expression:
                el_operators = prop.ops
                differentiation_order = prop.dord

                prop_tuple_key = prop_trivname(ord_el=len(el_operators), ord_geo=differentiation_order)

                nm_inds = tuple([index_choices[i] for i in prop.inds])
                cart_inds = tuple([greek_dict[num_Greek[i.o]] for i in prop.ops])
                all_inds = (*nm_inds, *cart_inds)
                # retrieve data for preperty (prop_key) and idxs_key which is (tuple(mode inds), tuple(cart inds))
                product *= props_data.get(prop_tuple_key).vals[all_inds]

            if product != 0.:
                logger.debug(f"Avrg prop contribution for indices {index_choices} and cart axes {cart_axes}: {product}")
                
            total += product

        return total * prefactor
    return compute_for_idx_choice


def calculate_avrg_tensor(avrg_expression: 'PropsCollection',
                          polarization: str, number_of_nmodes: int,
                          props_data: 'MolPropsCollection'):
    """
    [x] DONE

    Precalculating the full tensor for given avrg_expression
    """
    mode_inds = set(avrg_expression.get_mode_indices())
    ind_choices: list[dict[str, int]] = generate_index_choices_general(indlabels_in_motif=mode_inds, labels=list(range(number_of_nmodes)))
    func = make_func_to_compute_avrg(avrg_expression=avrg_expression, polarization=polarization)

    full_tensor = np.zeros((number_of_nmodes,)*len(mode_inds))

    for idx in ind_choices:
        full_tensor[tuple(dict(sorted(idx.items())).values())] = func(idx, props_data)

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


def get_avrg_motif_relation(avrg_expr_main: PropsCollection, avrg_expr_sub: PropsCollection, index_dict: dict):
    sub_encoded = nm_indices_repetition_reduce_deriv_symmetry(avrg_expr_sub)
    main_encoded = nm_indices_repetition_reduce_deriv_symmetry(avrg_expr_main)
    
    fill_list = [0] * len(main_encoded)
    for i, actual_ind in enumerate(tuple(avrg_expr_sub.get_mode_indices())):
        fill_list[i] = index_dict[actual_ind]
    
    return fill_list

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
