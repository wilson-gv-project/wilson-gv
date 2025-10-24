"""
PROPERTIES in VibPerturbedTerm ---- #TODO still
"""
from typing import Callable

import numpy as np
from wilson_suite.wilson_derive.abstractions import PolProp
from wilson_suite.wilson_intensities.amplitudes.term_parts import PropsCollection, VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.utils import generate_index_choices_general
from wilson_suite.wilson_main.abstractions import MolPropsCollection


def simple_prop_ID(property: 'PolProp') -> tuple[tuple, int]:
    """
    USING TUPLES OF TUPLES
    """
    operators = tuple([op.o for op in property.ops])
    return (operators, property.dord)


def make_avrg_props_motif(props: list['PolProp']) -> set[tuple]:
    """
    USING TUPLES OF TUPLES

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
    """
    lst = [PropsCollection(term.props).identify_avrg_motif() for term in list_of_terms]
    for l in lst:
        print(l)
    return set(PropsCollection(term.props).identify_avrg_motif() for term in list_of_terms)


def make_func_to_compute_avrg(*,
                     avrg_expression: 'PropsCollection',
                     polarization: str = 'ZZZZ') -> Callable[[dict, 'MolPropsCollection'], float]:
    """
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
                # print(prop_tuple_key, "nm_inds", nm_inds, "cart_inds", cart_inds, 'cart_axes', cart_axes)

                all_inds = (*nm_inds, *cart_inds)

                # retrieve data for preperty (prop_key) and idxs_key which is (tuple(mode inds), tuple(cart inds))
                product *= props_data.get(prop_tuple_key)[all_inds]
            # print('...')
            total += product

        return total * prefactor
    return compute_for_idx_choice


def precalculate_avrg_tensor(avrg_expression: 'PropsCollection',
                             polarization: str, number_of_nmodes: int,
                             props_data: 'MolPropsCollection'):
    """
    Precalculating the full tensor for given avrg_expression
    """
    mode_inds = set(avrg_expression.get_mode_indices())
    ind_choices: list[dict[str, int]] = generate_index_choices_general(indlabels_in_motif=mode_inds, labels=list(range(number_of_nmodes)))

    func = make_func_to_compute_avrg(avrg_expression=avrg_expression, polarization=polarization)

    full_tensor = np.zeros((number_of_nmodes,)*len(mode_inds))
    print('full_tensor.shape', full_tensor.shape)
    print(ind_choices)

    for idx in ind_choices:
        full_tensor[tuple(dict(sorted(idx.items())).values())] = func(idx, props_data)

    return full_tensor

def precalc_averages_for_terms():
    return