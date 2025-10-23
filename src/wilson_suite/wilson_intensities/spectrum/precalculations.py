import numpy as np
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm
    from .term_parts import PropsCollection
    from ...wilson_main.abstractions import MolPropsCollection

def precalc_unique_coeff_parts(data_to_precalc: dict):
    return


def make_func_to_compute_avrg(*,
                     avrg_expression: 'PropsCollection', 
                     polarization: str = 'ZZZZ') -> Callable:
    """
    for an expression with properties data values, 
    compute average with given polarization setup for a choice of normal mode indices
    """
    num_pulses = len(avrg_expression.get_cart_axes()) # should this be a set?
    from .averaging import getPolarizationAveragingExpression
    
    # polarization='ZZZZ' - only this one is possible now
    polarization_avrg_terms, prefactor = getPolarizationAveragingExpression(num_pulses=num_pulses, polarization=polarization)

    def compute_for_idx_choice(index_choices: dict, props_data: 'MolPropsCollection'):
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
                             polarization: str, 
                             props_data: 'MolPropsCollection'):
    """
    Precalculating the full tensor for given avrg_expression
    """
    return